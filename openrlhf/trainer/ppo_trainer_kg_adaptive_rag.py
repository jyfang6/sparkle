import os 
import numpy as np 
from tqdm import tqdm 
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from openrlhf.models.actor import Actor
from openrlhf.trainer.ppo_trainer import PPOTrainer 
from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.utils import compute_approx_kl, masked_mean 
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.trainer.ppo_utils.kl_controller import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker_kg_adaptive_rag import KGAdaptiveRAGExperienceMaker, KGAdaptiveRAGExperience, KGAdaptiveRAGExperienceMakerTreeRollout 
from openrlhf.trainer.ppo_utils.replay_buffer_kg_adaptive_rag import KGAdaptiveRAGReplayBuffer, KGAdaptiveRAGReplayBufferTreeRollout, NaiveReplayBuffer

from baselines.search_o1 import extract_answer 
from openrlhf.models.model_kg_adaptive_rag import extract_non_reasoning_model_answer 
from setup.setup import COMMON_FOLDER
import sys 
sys.path.append(COMMON_FOLDER)
from my_evaluation import f1_score 


class PPOTrainerKGAdaptiveRAG(PPOTrainer):

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 256, # 250, # 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super(PPOTrainer, self).__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.tokenizer = self.actor.tokenizer
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = KGAdaptiveRAGExperienceMaker(
            actor, critic, reward_model, initial_model, tokenizer, prompt_max_len, 
            self.kl_ctl, strategy, remote_rm_url, reward_fn
        )
        packing_samples = getattr(self.args, "packing_samples", False)

        self.replay_buffer = KGAdaptiveRAGReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples
        )

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)


    def fit(self, args, train_dataloader, test_dataloader, consumed_samples, num_update_steps_per_episodes=1):

        # num_update_steps_per_episodes = len(train_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader 
        self.best_score = -1 

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.train_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts, labels in self.train_dataloader:
                experience_list = self.experience_maker.make_experience_list(rand_prompts, labels, **self.generate_kwargs)
                # import pickle # TODO delete 
                # experience_list = pickle.load(open("/nfs/experience_list.pkl", "rb"))

                for experience in experience_list:
                    self.replay_buffer.append(experience)
                
                if self.args.advantage_estimator != "group_norm":
                    self.replay_buffer.normalize("advantages", self.strategy)
                
                status = self.ppo_train(steps)
                self.replay_buffer.clear()

                if "kl" in status:
                    # self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt) # PPO原本的代码
                    n_steps = int(status["response_length"]) #! 没用的话记得改回去
                    self.kl_ctl.update(status["kl"], n_steps)
                pbar.set_postfix(status)

                # logs/checkpoints 
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1
        
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
    
    def ppo_train(self, global_steps=0):

        torch.cuda.empty_cache()
        device = torch.cuda.current_device()
        dataloader = DataLoader(
            self.replay_buffer, 
            batch_size = self.replay_buffer.sample_batch_size, 
            shuffle = True, 
            drop_last = True, 
            collate_fn = self.replay_buffer.collate_fn
        )

        status_list = [] 
        status_mean = {} 
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]
                
                short_status = {} 
                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        # "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list: 
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        torch.cuda.empty_cache()
        # 清空梯度？ 
        return status_mean
    
    def training_step_actor_orig(self, experience: KGAdaptiveRAGExperience) -> Dict[str, float]:

        self.actor.train()

        trajectories = experience.trajectories
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        if self.args.use_kl_loss and experience.base_action_log_probs is not None:
            base_action_log_probs = experience.base_action_log_probs
        
        # actor loss 
        action_log_probs, action_mask = self.actor(trajectories)
        actor_loss = self.actor_loss_fn(
            action_log_probs, 
            old_action_log_probs, 
            advantages, 
            action_mask=action_mask
        )

        if self.args.use_kl_loss: 
            if self.initial_model is not None:
                kl = compute_approx_kl(
                    action_log_probs, 
                    base_action_log_probs, 
                    action_mask, 
                    kl_estimator=self.args.kl_estimator 
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
            
            kl_mean = masked_mean(kl, action_mask, dim=-1)
            kl_loss = kl_mean.mean()
            experience.info["kl"] = kl_loss.item()
        else:
            kl_loss = 0.0 

        aux_loss = 0 # 这里我简化了，没有使用aux_loss 
        loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value
        self.strategy.backward(loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
        
        # status 
        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status

    def _zero_touch(self, module: torch.nn.Module) -> torch.Tensor:
        # 产生一个“与该 module 参数相连但数值为 0”的标量，用于虚拟 backward
        # 注意：不要写成 torch.tensor(0., device=...)，那样与参数图不相连
        terms = []
        for p in module.parameters(recurse=True):
            if p.requires_grad:
                # 用 sum() 建图，再乘以 0，让梯度为 0 但参与通信
                terms.append(p.float().sum())
        if not terms:
            # 没有可训练参数时返回 0
            return torch.zeros((), device=next(module.parameters()).device)
        return sum(terms) * 0.0

    def training_step_actor(self, experience: KGAdaptiveRAGExperience, chunk_size=8) -> Dict[str, float]:

        self.actor.train()

        trajectories = experience.trajectories
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs 
        # if self.args.use_kl_loss and experience.base_action_log_probs is not None:
        #     base_action_log_probs = experience.base_action_log_probs
        # else:
        #     base_action_log_probs = None 
                
        num_actions = old_action_log_probs.shape[1] 
        total_actor_loss = 0.0 
        total_kl_loss = 0.0 

        max_num_actions = min(16, int(self.strategy.all_reduce(num_actions, "max")))

        # print("Optimising actor model ...")
        # print(f"Rank: {0 if self.strategy.is_rank_0() else 1} (actor): trajectory length: {len(trajectories[0])}")
        #>>>>>> for logging purposes 
        # seq_kl_accum = torch.zeros(len(trajectories), device=old_action_log_probs.device)
        eps_clip = getattr(self.args, "eps_clip", 0.2)
        clip_actions = 0 
        valid_actions = 0 
        # >>>>>>

        for i in range(0, max_num_actions, chunk_size):

            if i >= num_actions:
                sync_loss = self._zero_touch(self.actor.model)
                self.strategy.backward(sync_loss, self.actor, self.actor_optim)
                continue

            traj_chunk = [trajectory[i:i+chunk_size] for trajectory in trajectories]
            old_action_log_probs_chunk = old_action_log_probs[:, i:i+chunk_size]
            advantages_chunk = advantages[:, i:i+chunk_size]
            if base_action_log_probs is not None:
                base_action_log_probs_chunk = base_action_log_probs[:, i:i+chunk_size]
            
            action_log_probs_chunk, action_mask_chunk, *other = self.actor(trajectories=traj_chunk)
            actor_loss_chunk = self.actor_loss_fn.forward_chunk(
                action_log_probs_chunk, 
                old_action_log_probs_chunk, 
                advantages_chunk,
                action_mask=action_mask_chunk,
                full_action_mask=experience.action_mask
            )
            if self.args.use_kl_loss:
                raise ValueError(f"PPO Training should not set `--use_kl_loss` to True")
                # if self.initial_model is not None: # 这个计算有错误
                #     kl = compute_approx_kl(
                #         action_log_probs_chunk, 
                #         base_action_log_probs_chunk, 
                #         action_mask_chunk, 
                #         kl_estimator=self.args.kl_estimator 
                #     )
                # else:
                #     kl = torch.zeros_like(action_log_probs_chunk, dtype=action_log_probs_chunk.dtype, device=action_log_probs_chunk.device)
                # # kl_mean = masked_mean(kl, action_mask_chunk, dim=-1)
                # # kl_loss_chunk = kl_mean.mean()
                # kl_loss_chunk = ((kl.double()*action_mask_chunk.double()).sum(dim=-1) / (experience.action_mask.sum(dim=-1)+1e-8)).mean().float()
            else:
                kl_loss_chunk = 0.0
            
            aux_loss_chunk = 0 # 这里我简化了，没有使用aux_loss
            loss_chunk = actor_loss_chunk + aux_loss_chunk * self.args.aux_loss_coef + kl_loss_chunk * self.kl_ctl.value
            self.strategy.backward(loss_chunk, self.actor, self.actor_optim) # retain_graph=True) # 只计算梯度，同时会根据梯度累积的值来对loss进行scale
            # print(f"Rank: {0 if self.strategy.is_rank_0() else 1} (actor): Successfully backward using chunk {i} -- {i+chunk_size}")

            # >>>>>> for logging purposes
            ratio_chunk = torch.exp(action_log_probs_chunk - old_action_log_probs_chunk)
            clip_mask_chunk = ((ratio_chunk > (1.0 + eps_clip)) | (ratio_chunk < (1.0 - eps_clip))) & (action_mask_chunk > 0.5)
            clip_actions += clip_mask_chunk.sum().item()
            valid_actions += (action_mask_chunk > 0.5).sum().item()
            # kl = compute_approx_kl(action_log_probs_chunk, base_action_log_probs_chunk, action_mask_chunk, kl_estimator=self.args.kl_estimator)
            # seq_kl_accum += (kl * action_mask_chunk).sum(dim=1)
            # >>>>>>

            total_actor_loss += actor_loss_chunk.item()
            total_kl_loss += kl_loss_chunk.item() if torch.is_tensor(kl_loss_chunk) else kl_loss_chunk

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor") # 没调用一次会内部更新micro_steps，如果累积的micro_steps达到了梯度累积的值，那么就会更新参数，同时清空梯度
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
        
        status = {"policy_loss": total_actor_loss,"actor_lr": self.actor_scheduler.get_last_lr()[0]}
        # if self.args.use_kl_loss:
        #     experience.info["kl"] = total_kl_loss
        
        # >>>>> for logging purposes
        # status["sequence_kl_mean"] = seq_kl_accum.mean().item()
        # status["sequence_beta_kl_mean"] = (self.kl_ctl.value * seq_kl_accum).mean().item()
        status["ratio_clip_frac"] = clip_actions / max(1, valid_actions)
        # >>>>>>

        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.float().mean().item()
        # print("# >>>>> actor status: ", status)
        return status
    
    def training_step_critic_orig(self, experience: KGAdaptiveRAGExperience) -> Dict[str, float]:

        self.critic.train() 

        trajectories = experience.trajectories
        old_values = experience.values 
        returns = experience.returns 

        values, action_mask = self.critic(
            trajectories = trajectories, 
            max_num_actions = self.actor.max_num_actions, 
            max_prompt_length = self.actor.prompt_max_len + self.actor.generate_max_len 
        )

        critic_loss = self.critic_loss_fn(values, old_values, returns, action_mask)

        aux_loss = 0 # 这里我简化了，没有使用aux_loss 
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        
        return status
    
    def training_step_critic(self, experience: KGAdaptiveRAGExperience, chunk_size=8) -> Dict[str, float]:

        self.critic.train() 

        trajectories = experience.trajectories
        old_values = experience.values 
        returns = experience.returns 

        num_actions = old_values.shape[1]
        total_critic_loss = 0.0 
        values_list = [] 

        max_num_actions = min(16, int(self.strategy.all_reduce(num_actions, "max")))

        # print("Optimising critic model ...")  
        # print(f"Rank: {0 if self.strategy.is_rank_0() else 1} (critic): trajectory length: {len(trajectories[0])}")
        for i in range(0, max_num_actions, chunk_size):

            if i >= num_actions:
                sync_loss = self._zero_touch(self.critic)
                self.strategy.backward(sync_loss, self.critic, self.critic_optim)
                continue

            traj_chunk = [trajectory[i:i+chunk_size] for trajectory in trajectories]
            old_values_chunk = old_values[:, i:i+chunk_size]
            returns_chunk = returns[:, i:i+chunk_size]
            values_chunk, action_mask_chunk = self.critic(
                trajectories = traj_chunk, 
                max_num_actions = self.actor.max_num_actions, 
                max_prompt_length = self.actor.prompt_max_len + self.actor.generate_max_len 
            )
            values_list.append(values_chunk.detach())

            critic_loss_chunk = self.critic_loss_fn.forward_chunk(
                values_chunk, 
                old_values_chunk, 
                returns_chunk, 
                action_mask=action_mask_chunk,
                full_action_mask=experience.action_mask
            )

            aux_loss = 0 # 这里我简化了，没有使用aux_loss 
            loss_chunk = critic_loss_chunk + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(loss_chunk, self.critic, self.critic_optim) # , retain_graph=True)
            # print(f"Rank: {0 if self.strategy.is_rank_0() else 1} (critic): Successfully backward using chunk {i} -- {i+chunk_size}")

            total_critic_loss += critic_loss_chunk.item()

        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
        
        # status 
        values = torch.cat(values_list, dim=1)
        status = {
            "critic_loss": total_critic_loss,
            "values": masked_mean(values, experience.action_mask[:, :values.shape[1]]).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }

        return status
        
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):

        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            score, recall = self.evaluate(self.test_dataloader)
            score = self.strategy.all_reduce(score) 
            recall = self.strategy.all_reduce(recall) 
            if self._wandb is not None and self.strategy.is_rank_0():
                print(f"##>>>>>>>>> F1 Score: {score}, Recall: {recall} <<<<<<<<<<<<<<<< ## ")
                self._wandb.log({"eval/f1_score": score, "eval/recall": recall, "eval/global_step": global_step})
            
            if score > self.best_score:
                self._save_checkpoint(args, f"best_dev_step{global_step}", client_states)
                self.best_score = score 
        
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def evaluate(self, dataloader, **kwargs):
        
        self.actor.eval()
        generation_config = {"do_sample": False, "temperature": None}
        all_prompts, all_labels = [], [] 
        for batch_prompts, batch_labels in dataloader:
            all_prompts.extend(batch_prompts)
            all_labels.extend(batch_labels)
        all_samples = self.experience_maker.generate_samples(all_prompts, all_labels, **generation_config)
        
        all_labels, all_outputs = [], []
        for sample in all_samples:
            for label, output in zip(sample.labels, sample.outputs):
                all_labels.append(label)
                all_outputs.append(output)
        
        # Evaluation
        scores = [] 
        for output, labels in zip(all_outputs, all_labels):
            if "<think>" in output or "</think>" in output:
                pred_answer = extract_non_reasoning_model_answer(output)
            else:
                pred_answer = extract_answer(output, mode="qa")
            f1 = max(f1_score(pred_answer, gold_answer)[0] for gold_answer in labels)
            scores.append(f1)
        
        # compute recall 
        recalls = [] 
        for sample in all_samples:
            for retrieval_trajectory in sample.retrieval_trajectories:
                question = retrieval_trajectory[-1]["question"]
                retrieved_ctxids = set(retrieval_trajectory[-1]["final_context_ids"])
                gold_ctxids = self.reward_model.question_2_gold_ctxids[question] 
                recalls.append(self.reward_model.compute_recall(gold_ctxids, retrieved_ctxids))
        
        return np.mean(scores), np.mean(recalls)
        

class PPOTrainerKGAdaptiveRAGTreeRollout(PPOTrainer):

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 256, # 250, # 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super(PPOTrainer, self).__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.tokenizer = self.actor.tokenizer
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = KGAdaptiveRAGExperienceMakerTreeRollout(
            actor, critic, reward_model, initial_model, tokenizer, prompt_max_len, 
            self.kl_ctl, strategy, remote_rm_url, reward_fn
        )
        packing_samples = getattr(self.args, "packing_samples", False)

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples
        )

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)
    
    def fit(self, args, train_dataloader, test_dataloader, consumed_samples, num_update_steps_per_episodes=1):

        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader 
        self.best_score = -1 

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.train_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts, labels in self.train_dataloader:
                experience_list = self.experience_maker.make_experience_list(rand_prompts, labels, **self.generate_kwargs)

                for i, experience in enumerate(experience_list):
                    self.replay_buffer.append(experience) 
                
                if self.args.advantage_estimator != "group_norm":
                    self.replay_buffer.normalize("advantages", self.strategy)
                status = self.ppo_train(steps)
                self.replay_buffer.clear()

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
    
    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()

        sequences = experience.sequences
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        num_actions = experience.action_mask.size(1)
        packed_seq_lens = None
        attention_mask = experience.attention_mask
        if self.args.use_kl_loss and experience.base_action_log_probs is not None:
            base_action_log_probs = experience.base_action_log_probs

        # actor loss
        action_log_probs, output = self.actor.tree_rollout_forward(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )

        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
        )

        if self.args.use_kl_loss:
            if self.initial_model is not None:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    experience.action_mask,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
            
            kl_mean = masked_mean(kl, experience.action_mask, dim=-1)
            kl_loss = kl_mean.mean()
            experience.info["kl"] = kl_loss.item()
        else:
            kl_loss = 0

        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value
        self.strategy.backward(loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        # status
        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}

        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status
    
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            metrics = self.evaluate(self.test_dataloader)
            
            # 同步不同卡的信息
            metrics = self.strategy.all_reduce(metrics)
            f1_score = metrics["f1_score"]
            recall = metrics["recall"]
            
            # 记录到wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                print(f"##>>>>>>>>> F1 Score: {f1_score}, Recall: {recall} <<<<<<<<<<<<<<<< ## ")
                logs = {
                    "eval/%s" % k: v
                    for k, v in {
                        **metrics,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
            
            # 保存最佳模型
            if f1_score > self.best_score:
                self._save_checkpoint(args, f"best_dev_step{global_step}", client_states)
                self.best_score = f1_score 
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def evaluate(self, dataloader, **kwargs):
        
        self.actor.eval()
        generation_config = {"do_sample": False, "temperature": None}
        
        # 收集所有prompts和labels
        all_prompts, all_labels = [], [] 
        for batch_prompts, batch_labels in dataloader:
            all_prompts.extend(batch_prompts)
            all_labels.extend(batch_labels)
        
        # 使用actor.generate方法生成trajectory和outputs
        # 需要加载reasoning model
        from openrlhf.models.actor_kg_adaptive_rag import ReasoningModel
        args = self.strategy.args
        self.actor._reasoning_model = ReasoningModel(
            reasoning_model=args.reasoning_model, 
            use_flash_attention_2=args.flash_attn, 
            bf16=args.bf16
        ).to("cuda")
        
        # 生成samples
        all_trajectories, all_outputs, all_retrieval_trajectories = [], [], []
        make_experience_batch_size = 4
        pbar = tqdm(
            range(0, len(all_prompts), make_experience_batch_size),
            desc="Evaluating",
            disable=not self.strategy.is_rank_0(),
        )
        for i in pbar:
            prompts = all_prompts[i : i + make_experience_batch_size]
            trajectories, outputs, retrieval_trajectories = self.actor.generate(prompts, **generation_config)
            all_trajectories.extend(trajectories)
            all_outputs.extend(outputs)
            all_retrieval_trajectories.extend(retrieval_trajectories)
            pbar.set_postfix({"processed": len(all_outputs), "total": len(all_prompts)})
        
        # 清理reasoning model
        reasoning_model = self.actor._reasoning_model 
        self.actor._reasoning_model = None
        del reasoning_model 
        torch.cuda.empty_cache()
        
        # Evaluation - 计算F1分数
        scores = [] 
        for output, labels in zip(all_outputs, all_labels):
            if "<think>" in output or "</think>" in output:
                pred_answer = extract_non_reasoning_model_answer(output)
            else:
                pred_answer = extract_answer(output, mode="qa")
            f1 = max(f1_score(pred_answer, gold_answer)[0] for gold_answer in labels)
            scores.append(f1)
        
        # 计算recall
        recalls = [] 
        for retrieval_trajectory in all_retrieval_trajectories:
            question = retrieval_trajectory[-1]["question"]
            retrieved_ctxids = set(retrieval_trajectory[-1]["final_context_ids"])
            gold_ctxids = self.reward_model.question_2_gold_ctxids[question] 
            recalls.append(self.reward_model.compute_recall(gold_ctxids, retrieved_ctxids))
        
        f1_score_mean = np.mean(scores)
        recall_mean = np.mean(recalls)
        
        return {"f1_score": f1_score_mean, "recall": recall_mean}
