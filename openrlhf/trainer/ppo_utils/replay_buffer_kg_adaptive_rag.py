import torch 
from dataclasses import dataclass
from typing import List, Optional

from openrlhf.trainer.ppo_utils.replay_buffer import NaiveReplayBuffer, zero_pad_sequences
from openrlhf.trainer.ppo_utils.experience_maker_kg_adaptive_rag import KGAdaptiveRAGExperience


@dataclass
class BufferItem:

    trajectories: List[dict]
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]


def split_experience_batch(experience: KGAdaptiveRAGExperience) -> List[BufferItem]:
    
    batch_size = len(experience.trajectories)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "trajectories",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}

    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert vv.numel() == 1, f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def remove_padding_in_sequences(items: List[BufferItem]) -> List[BufferItem]:

    for item in items:

        act_log_prob, base_act_log_prob, value, ret, adv, act_mask = (
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        item.action_log_probs = act_log_prob[:right_pad] 
        item.base_action_log_probs = base_act_log_prob[:right_pad] if base_act_log_prob is not None else None
        item.values = value[:right_pad] if value is not None else None 
        item.returns = ret[:right_pad]
        item.advantages = adv[:right_pad]
        item.action_mask = act_mask[:right_pad]
        
    return items 


def make_experience_batch(items: List[BufferItem]) -> KGAdaptiveRAGExperience:

    kwargs = {} 
    keys = (
        "trajectories",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "action_mask",
    )

    for key in keys:
        vals = [getattr(item, key) for item in items]
        if vals[0] is not None and isinstance(vals[0], torch.Tensor):
            batch_data = zero_pad_sequences(vals, "left")
        else:
            batch_data = vals 
        kwargs[key] = batch_data 
    
    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    
    return KGAdaptiveRAGExperience(**kwargs)
        

class KGAdaptiveRAGReplayBuffer(NaiveReplayBuffer):

    @torch.no_grad()
    def append(self, experience: KGAdaptiveRAGExperience):
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        
        items = split_experience_batch(experience)
        # the packed samples comes with no padding
        if not self.packing_samples:
            items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]
    
    def collate_fn(self, batch):
        experience = make_experience_batch(batch)
        return experience
    
class KGAdaptiveRAGReplayBufferTreeRollout(NaiveReplayBuffer):
    pass 


