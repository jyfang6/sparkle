import math 
import random 
from copy import deepcopy 
from tqdm import trange, tqdm
from dataclasses import dataclass
from typing import List, Optional, Union 

import torch
import torch.distributed as dist
import torch.nn.functional as F

from openrlhf.models.utils import compute_approx_kl, masked_mean, compute_reward, compute_reward_use_recall
from openrlhf.trainer.ppo_utils.experience_maker import NaiveExperienceMaker, Experience, to
from openrlhf.trainer.ppo_utils.experience_maker import Samples as NaiveSamples 
from openrlhf.models.actor_kg_adaptive_rag import ReasoningModel 


@dataclass
class KGAdaptiveRAGExperience:

    trajectories: List[List[dict]] 
    action_mask: torch.Tensor 
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor 
    values: torch.Tensor 
    info: Optional[dict]
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    kl: Optional[torch.Tensor] = None 
    action_type: Optional[torch.Tensor] = None
    action_per_token_log_probs: Optional[torch.Tensor] = None
    action_per_token_log_probs_mask: Optional[torch.Tensor] = None 

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.action_mask = to(self.action_mask, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.action_type = to(self.action_type, device)
        self.action_per_token_log_probs = to(self.action_per_token_log_probs, device) 
        self.action_per_token_log_probs_mask = to(self.action_per_token_log_probs_mask, device)
        self.values = to(self.values, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        return self


@dataclass
class Samples:

    trajectories: List[List[dict]]
    labels: List[str]
    outputs: List[str]
    retrieval_trajectories: List[List[dict]] 

class KGAdaptiveRAGExperienceMaker(NaiveExperienceMaker):
    
    @torch.no_grad()
    def make_experience_list(self, all_prompts: List[str], all_labels: List[str], **generate_kwargs) -> List[KGAdaptiveRAGExperience]:

        args = self.strategy.args
        # generate response
        samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)
        # import pickle 
        # pickle.dump(samples_list, open("/nfs/samples_list.pkl", "wb"))
        # samples_list = pickle.load(open("/nfs/samples_list.pkl", "rb"))

        torch.distributed.barrier()
        torch.cuda.synchronize()

        experiences = []
        for sample in tqdm(samples_list, desc="make_experience", disable=not self.strategy.is_rank_0()):
            experiences.append(self.make_experience(sample).to_device("cpu"))

        if not self.strategy.args.rm_use_recall:
            experiences, rewards = self.process_experiences(experiences)
        else:
            rewards = [[] for i in range(len(experiences))] # dummy placeholder 
            assert self.strategy.args.advantage_estimator == "gae", "When using recall-based reward, only GAE is supported for advantage estimation. Please set advantage_estimator to 'gae'." 

        # calculate returns and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            num_actions = experience.info["num_actions"]
            if not self.strategy.args.rm_use_recall:
                reward = reward.to(device="cuda")
                # version 1: 原始版本
                # reward = compute_reward(
                #     reward, 
                #     self.kl_ctl.value, 
                #     experience.kl, 
                #     action_mask=experience.action_mask, 
                #     num_actions=num_actions,
                #     reward_clip_range=args.reward_clip_range,
                # )
                # version 2: 增加对长度的惩罚
                # L = experience.action_mask.sum(dim=-1).float()
                # L0 = torch.median(L).detach()
                # beta_eff = (self.kl_ctl.value * (L0/L)).unsqueeze(-1)
                # reward = compute_reward(
                #     reward, 
                #     beta_eff, 
                #     experience.kl, 
                #     action_mask=experience.action_mask, 
                #     num_actions=num_actions,
                #     reward_clip_range=args.reward_clip_range,
                # )
                # version 3: 使用token级别的KL 
                L_step = experience.action_per_token_log_probs_mask.sum(-1).float() # [n_traj, num_actions] 
                L0 = torch.median(L_step).detach() 
                beta_eff = (self.kl_ctl.value * (L0 / L_step.clamp_min(1.0))).clamp(max=3.0)
                reward = compute_reward(
                    reward, 
                    beta_eff, 
                    experience.kl, 
                    action_mask=experience.action_mask, 
                    num_actions=num_actions, 
                    reward_clip_range=args.reward_clip_range
                )
            else:
                # L = experience.action_mask.sum(dim=-1).float()
                # L0 = torch.median(L).detach() 
                # beta_eff = (self.kl_ctl.value * (L0/L)).unsqueeze(-1)
                # reward = compute_reward_use_recall(
                #     experience.info["sequence_reward"], 
                #     experience.info["step_reward"], 
                #     beta_eff, 
                #     experience.kl, 
                #     action_mask=experience.action_mask, 
                #     num_actions=num_actions,
                #     reward_clip_range=args.reward_clip_range, 
                # )
                # version 2: 
                L_step = experience.action_per_token_log_probs_mask.sum(-1).float() # [n_traj, num_actions] 
                L0 = torch.median(L_step).detach() 
                beta_eff = (self.kl_ctl.value * (L0 / L_step.clamp_min(1.0))).clamp(max=3.0)
                reward = compute_reward_use_recall(
                    experience.info["sequence_reward"], 
                    experience.info["step_reward"], 
                    beta_eff, 
                    experience.kl, 
                    action_mask=experience.action_mask, 
                    num_actions=num_actions,
                    reward_clip_range=args.reward_clip_range, 
                )
                del experience.info["sequence_reward"]
                del experience.info["step_reward"]

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    values=experience.values,
                    rewards=reward, 
                    action_mask=experience.action_mask,
                    gamma=generate_kwargs["gamma"],
                    lambd=generate_kwargs["lambd"],
                )
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")
            
            # NOTE: 我自己添加的
            experience.advantages = self.normalize_adv_by_type(
                adv=experience.advantages, 
                step_type_ids=experience.action_type, 
                action_mask=experience.action_mask, 
                L_step=L_step, 
                alpha=0.5, n_types=4, pad_type_id=0, min_count=8, 
                clip_val=5.0, soft_clip=False, recenter_after_clip=False 
            )
            # calculate the return info 
            return_sums = reward.sum(dim=-1) # 真的是reward吗？
            experience.info["return"] = return_sums
            # remove unnecessary info 
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], all_labels: List[str], **generate_kwargs) -> List[Samples]:
        torch.cuda.empty_cache()
        args = self.strategy.args
        # self.actor._reasoning_model.to("cuda")
        # load reasoning model 
        self.actor._reasoning_model = ReasoningModel(reasoning_model=args.reasoning_model, use_flash_attention_2=args.flash_attn, bf16=args.bf16).to("cuda")

        self.actor.eval()

        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        # samples_list = []
        # for i in trange(0, len(all_prompts), args.micro_rollout_batch_size, desc="Making experience", disable=not self.strategy.is_rank_0()):
        #     prompts = all_prompts[i : i + args.micro_rollout_batch_size]
        #     labels = all_labels[i : i + args.micro_rollout_batch_size]
        #     trajectories, outputs = self.actor.generate(prompts, **generate_kwargs)
        #     samples_list.append(Samples(trajectories=trajectories, labels=labels, outputs=outputs))
        all_trajectories, all_outputs, all_retrieval_trajectories = [], [], [] 
        make_experience_batch_size = 4
        for i in trange(0, len(all_prompts), make_experience_batch_size, desc="Generating samples", disable=not self.strategy.is_rank_0()):
            prompts = all_prompts[i : i + make_experience_batch_size]
            trajectories, outputs, retrieval_trajectories = self.actor.generate(prompts, **generate_kwargs)
            all_trajectories.extend(trajectories)
            all_outputs.extend(outputs)
            all_retrieval_trajectories.extend(retrieval_trajectories)
        
        # 转换成args.micro_rollout_batch_size的samples 
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            samples_list.append(
                Samples(
                    trajectories=all_trajectories[i : i + args.micro_rollout_batch_size], 
                    labels=all_labels[i : i + args.micro_rollout_batch_size], 
                    outputs=all_outputs[i : i + args.micro_rollout_batch_size],
                    retrieval_trajectories=all_retrieval_trajectories[i : i + args.micro_rollout_batch_size]
                )
            )
        
        # self.actor._reasoning_model.to("cpu")
        reasoning_model = self.actor._reasoning_model 
        self.actor._reasoning_model = None
        del reasoning_model 
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples):

        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        # if self.reward_model is not None: # 注释掉是因为reward_model不是nn.Module()
        #     self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # log probs
        action_log_probs, action_mask, action_type, action_per_token_log_probs, action_per_token_log_probs_mask = \
            self.actor(trajectories=samples.trajectories)
        torch.cuda.empty_cache()

        if self.initial_model is not None:
            # base_action_log_probs = self.initial_model(trajectories=samples.trajectories)[0]
            base_action_log_probs, _, _, base_action_per_token_log_probs, _ = self.initial_model(trajectories=samples.trajectories)
            torch.cuda.empty_cache()
        else:
            base_action_log_probs, base_action_per_token_log_probs = None, None 

        # value 
        if self.critic is not None:
            value = self.critic(
                trajectories=samples.trajectories, 
                max_num_actions=self.actor.max_num_actions, 
                max_prompt_length=self.actor.prompt_max_len+self.actor.generate_max_len
            )[0]
            torch.cuda.empty_cache()
        else:
            value = None 
        
        # rewards 
        # r = self.reward_model(predictions=samples.outputs, labels=samples.labels, retrieval_trajectories=samples.retrieval_trajectories).to(action_log_probs.device)
        if not self.strategy.args.rm_use_recall:
            r = self.reward_model(predictions=samples.outputs, labels=samples.labels).to(action_log_probs.device)
        else:
            r = self.reward_model(predictions=samples.outputs, labels=samples.labels, retrieval_trajectories=samples.retrieval_trajectories)
            r = {key: value.to(action_log_probs.device) for key, value in r.items()}

        #! 使用use_kl_loss表明用的是GROP算法，用PPO的话use_kl_loss是False，需要计算kl
        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl_per_token = compute_approx_kl(
                action_per_token_log_probs,
                base_action_per_token_log_probs, 
                action_mask=action_per_token_log_probs_mask,
                kl_estimator=self.strategy.args.kl_estimator 
            ) # [n_traj, num_actions, num_tokens]
            kl = kl_per_token.sum(dim=-1) # [n_traj, num_actions]

            if self.strategy.args.kl_estimator == "k1":

                # cap kl to prevent 
                L_step = action_per_token_log_probs_mask.sum(-1).float()
                cap = torch.clamp(0.5 * math.sqrt(2.0 * self.strategy.args.kl_target) * L_step, min=1.0, max=8.0)
                kl = torch.clamp(kl, -cap, cap)

                kl_per_token_for_logging = compute_approx_kl(
                    action_per_token_log_probs,
                    base_action_per_token_log_probs, 
                    action_mask=action_per_token_log_probs_mask,
                    kl_estimator="k2" 
                )
            else:
                kl_per_token_for_logging = kl_per_token
        else:
            kl_per_token = torch.zeros_like(action_per_token_log_probs, dtype=action_per_token_log_probs.dtype, device=action_per_token_log_probs.device)
            kl = kl_per_token.sum(dim=-1)

        if not self.strategy.args.rm_use_recall:
            # info = {
            #     "kl": masked_mean(kl, action_mask, dim=-1), # logging的时候记录per-token的KL
            #     "reward": r, 
            #     "response_length": action_mask.float().sum(dim=-1), 
            #     "num_actions": action_mask.shape[1],
            # }
            info = {
                "kl": (kl_per_token_for_logging*action_per_token_log_probs_mask).sum(dim=(1,2)) / action_per_token_log_probs_mask.sum(dim=(1,2)).clamp_min(1),
                "reward": r, 
                "response_length": action_per_token_log_probs_mask.sum(dim=(1,2)).clamp_min(1),
                "num_actions": action_mask.shape[1]
            }
        else:
            # info = {
            #     "kl": masked_mean(kl, action_mask, dim=-1),
            #     "reward": r["total_reward"], 
            #     "response_length": action_mask.float().sum(dim=-1), 
            #     "num_actions": action_mask.shape[1],
            #     "sequence_reward": r["sequence_reward"], 
            #     "step_reward": r["step_reward"], 
            # }
            info = {
                "kl": (kl_per_token_for_logging*action_per_token_log_probs_mask).sum(dim=(1,2)) / action_per_token_log_probs_mask.sum(dim=(1,2)).clamp_min(1),
                "reward": r["total_reward"], 
                "response_length": action_per_token_log_probs_mask.sum(dim=(1,2)).clamp_min(1),
                "num_actions": action_mask.shape[1], 
                "sequence_reward": r["sequence_reward"], 
                "step_reward": r["step_reward"], 
            }

        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train() 
        
        return KGAdaptiveRAGExperience(
            trajectories=samples.trajectories,
            action_log_probs=action_log_probs, 
            action_mask=action_mask,
            action_type=action_type, 
            action_per_token_log_probs = action_per_token_log_probs, 
            action_per_token_log_probs_mask=action_per_token_log_probs_mask,
            base_action_log_probs=base_action_log_probs,
            values=value, 
            kl=kl,
            info=info,
            returns=None,
            advantages=None
        )
    
    @torch.no_grad()
    def normalize_adv_by_type(
        self, 
        adv: torch.Tensor,                  # [B, S]  step-level advantages
        step_type_ids: torch.Tensor,        # [B, S]  0=padding, 1/2/3=真实动作
        action_mask: torch.Tensor,          # [B, S]  有效步=1, 否则=0
        L_step: torch.Tensor = None,        # [B, S]  每步token数（可选，用于长度均衡）
        alpha: float = 0.0,                 # 长度均衡指数：0=不用；0.5温和；1.0强等权
        n_types: int = 4,                   # 总类型数，含 padding=0
        pad_type_id: int = 0,
        min_count: int = 8,
        eps: float = 1e-8,
        clip_val: float = 5.0,              # <<< 新增：裁剪阈值
        soft_clip: bool = False,            # <<< 新增：True=tanh，False=硬裁剪
        recenter_after_clip: bool = False,  # <<< 可选：裁剪后再去均值
    ):
        """
        返回：组内标准化（并可选长度均衡）后的 advantages, 形状同 adv。
        仅用于 policy gradient; returns 不要归一化。
        """
        assert adv.shape == step_type_ids.shape == action_mask.shape
        B, S = adv.shape
        device = adv.device
        adv = adv.float()
        action_mask = action_mask.float()

        # 仅对“有效且非 padding”的步做归一
        valid = (action_mask > 0.5) & (step_type_ids != pad_type_id)    # [B,S]

        # one-hot 分组，padding 那一组会在 valid=0 下被自动排除
        onehot = F.one_hot(step_type_ids.long(), num_classes=n_types).float().to(device)   # [B,S,K]
        group_mask = onehot * valid.unsqueeze(-1)                                          # [B,S,K]

        # 组内统计（E[x] 与 Var[x]）
        counts = group_mask.sum(dim=(0,1))                                                 # [K]
        sum_per_group   = (adv.unsqueeze(-1)    * group_mask).sum(dim=(0,1))               # [K]
        sumsq_per_group = ((adv**2).unsqueeze(-1) * group_mask).sum(dim=(0,1))             # [K]

        mean_per_group = torch.where(
            counts >= min_count, sum_per_group / counts.clamp_min(1.0), torch.zeros_like(counts)
        )
        var_per_group  = torch.clamp(sumsq_per_group / counts.clamp_min(1.0) - mean_per_group**2, min=0.0)
        std_per_group  = torch.where(
            counts >= min_count, torch.sqrt(var_per_group + eps), torch.ones_like(counts)
        )

        # 广播到 [B,S]
        mean_g = mean_per_group.gather(0, step_type_ids.view(-1)).view(B, S)
        std_g  = std_per_group.gather(0, step_type_ids.view(-1)).view(B, S)

        # 标准化，仅对 valid 位置；其余置 0
        adv_norm = torch.where(valid, (adv - mean_g) / std_g, torch.zeros_like(adv))

        # 可选：长度均衡（让长/短步更公平）
        if (L_step is not None) and (alpha > 0.0):
            # 参考长度只在 valid 上取中位数
            L0 = torch.median(L_step[valid]).detach()
            w_len = (L0 / L_step.clamp_min(1.0))**alpha
            adv_norm = torch.where(valid, adv_norm * w_len, adv_norm)
        
        # clip advantages 
        if clip_val is not None and clip_val > 0:
            if soft_clip:
                # 软裁剪：更平滑，常用 c=5
                clipped = clip_val * torch.tanh(adv_norm / clip_val)
            else:
                # 硬裁剪：绝对值不超过 clip_val
                clipped = torch.clamp(adv_norm, -clip_val, clip_val)
            adv_norm = torch.where(valid, clipped, torch.zeros_like(adv_norm))

        # （可选）裁剪后再去一次均值，避免引入偏置
        if recenter_after_clip:
            m = (adv_norm * valid).sum() / valid.sum().clamp_min(1.0)
            adv_norm = torch.where(valid, adv_norm - m, adv_norm)

        return adv_norm
    

@dataclass
class TreeRolloutSamples:

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: list[str]
    pad_len: Optional[int]
    reward: torch.Tensor


class KGAdaptiveRAGExperienceMakerTreeRollout(NaiveExperienceMaker):
    
    @torch.no_grad()
    def make_experience_list(self, all_prompts: List[str], all_labels: List[str], **generate_kwargs) -> List[KGAdaptiveRAGExperience]:

        args = self.strategy.args
        samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)

        experiences = [] 
        for samples in tqdm(samples_list, desc="make_experience", disable=not self.strategy.is_rank_0()):
            experiences.append(self.make_experience(samples).to_device("cpu"))
        
        experiences, rewards = self.process_experiences(experiences)
        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")

        return experiences


    @torch.no_grad()
    def generate_samples(self, all_prompts, all_labels, **generate_kwargs):
        
        torch.cuda.empty_cache()
        args = self.strategy.args
        self.actor._reasoning_model = ReasoningModel(reasoning_model=args.reasoning_model, use_flash_attention_2=args.flash_attn, bf16=args.bf16).to("cuda")
        self.actor.eval()

        samples_list = []
        for i in trange(len(all_prompts)):
            sequences, attention_mask, action_mask, reward = self.actor.tree_rollout(
                question = all_prompts[i], 
                reward_model = self.reward_model, 
                label=all_labels[i],
                **generate_kwargs
            )

            if dist.is_initialized():
                dist.barrier() 
            
            for j in range(0, len(sequences), args.micro_rollout_batch_size):

                batch_sequences = sequences[j: j+args.micro_rollout_batch_size]
                batch_attention_mask = attention_mask[j: j+args.micro_rollout_batch_size]
                batch_action_mask = action_mask[j: j+args.micro_rollout_batch_size]
                batch_reward = reward[j: j+args.micro_rollout_batch_size] 
                num_actions = batch_action_mask.size(1)

                if len(batch_sequences) != args.micro_rollout_batch_size:
                    padding_length = args.micro_rollout_batch_size - len(batch_sequences)
                    padding_sequences = sequences[:padding_length] 
                    padding_attention_mask = attention_mask[:padding_length]
                    padding_action_mask = action_mask[:padding_length]
                    padding_reward = reward[:padding_length]
                    batch_sequences = torch.cat([batch_sequences, padding_sequences], dim=0)
                    batch_attention_mask = torch.cat([batch_attention_mask, padding_attention_mask], dim=0)
                    batch_action_mask = torch.cat([batch_action_mask, padding_action_mask], dim=0)
                    batch_reward = torch.cat([batch_reward, padding_reward], dim=0)

                batch_prompts, batch_labels = [], [] 
                for k in range(args.micro_rollout_batch_size):
                    batch_prompts.append(
                        self.actor.tokenizer.decode(
                            torch.masked_select(
                                batch_sequences[k, :-num_actions], 
                                batch_attention_mask[k, :-num_actions].bool()
                            )
                        )
                    )
                    batch_labels.append(
                        self.actor.tokenizer.decode(
                            torch.masked_select(
                                batch_sequences[k, -num_actions:], 
                                batch_attention_mask[k, -num_actions:].bool()
                            )
                        )
                    )

                samples = TreeRolloutSamples(
                    sequences=batch_sequences, 
                    attention_mask=batch_attention_mask,
                    action_mask=batch_action_mask,
                    num_actions=num_actions, 
                    packed_seq_lens=None,
                    response_length=batch_action_mask.float().sum(dim=-1), 
                    total_length=batch_attention_mask.float().sum(dim=-1), 
                    prompts=batch_prompts, 
                    labels=batch_labels, 
                    pad_len=None, 
                    reward=batch_reward
                )
                
                samples_list.append(samples)
        
        reasoning_model = self.actor._reasoning_model 
        self.actor._reasoning_model = None
        del reasoning_model 
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()

        if dist.is_initialized():
            # 多卡的时候，保证不同卡上的samples_list长度一样
            num_samples = 256 
            if len(samples_list) > num_samples:
                samples_list = random.sample(samples_list, num_samples) 
            elif len(samples_list) < num_samples:
                samples_list = samples_list + random.choices(samples_list, k=num_samples - len(samples_list)) 

        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: TreeRolloutSamples) -> Experience:

        self.actor.eval() 
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.critic is not None:
            self.critic.eval()
        
        # extract values from samples
        actor_model_device = self.actor.model.device
        sequences = samples.sequences.to(actor_model_device)
        attention_mask = samples.attention_mask.to(actor_model_device)
        action_mask = samples.action_mask.to(actor_model_device)
        num_actions = samples.num_actions
        r = samples.reward

        # log probs
        action_log_probs = self.actor.tree_rollout_forward(sequences, num_actions, attention_mask)

        if self.initial_model is not None:
            base_action_log_probs = self.initial_model.tree_rollout_forward(sequences, num_actions, attention_mask)
        else:
            base_action_log_probs = None 
        
        # values 
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )
