import os 
import re 
import math 
import torch
import torch.nn as nn 
from typing import List 
from transformers import AutoConfig, AutoTokenizer
from prompts.adaptive_rag import * 
from baselines.search_o1 import extract_answer 
from openrlhf.models.actor_kg_adaptive_rag import KGRetriever
from generator.generator import Generator 

from setup.setup import COMMON_FOLDER, HF_TOKEN
from utils.pipeline_utils import load_llm_tokenizer_and_model

import sys 
sys.path.append(COMMON_FOLDER)
from my_utils import load_json 
from my_evaluation import f1_score 


def extract_non_reasoning_model_answer(output: str) -> str:
    
    original_output = output
    
    # 第一步：非贪心查找 <answer> </answer> 对
    pattern1 = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern1, output, re.DOTALL)
    for match in matches:
        answer = match.replace("<answer>", "").replace("</answer>", "").strip()
        if answer:  # 非空
            return answer
    
    # 第二步：替换标签后查找
    ANSWER_TAG = "ANSWER_TAG"
    output = output.replace("<answer>", ANSWER_TAG).replace("</answer>", ANSWER_TAG)
    segments = re.split(ANSWER_TAG, output)
    for segment in segments:
        segment = segment.strip()
        if segment:
            return segment # 返回第一个非空段落 

    # 都不成功，print错误并返回原始output
    print(f"Warning: Failed to extract answer from output: {original_output}")
    return original_output


def get_kg_adaptive_critic_model(model_name_or_path, base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):

    class KGAdaptiveCriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):

            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, token=HF_TOKEN)
            self.determine_retrieval_action = "Determine Retrieval"
            self.formulate_retrieval_queries_action = "Formulate Retrieval Query"
            self.identify_relevant_triples_action = "Identify Relevant Triples" 

        def get_action_prompt(self, action: str, state: dict) -> str:

            PROMPT_MAP = {
                # self.determine_retrieval_action: (DETERMINE_RETRIEVAL_INSTRUCTION, DETERMINE_RETRIEVAL_INPUTS),
                self.determine_retrieval_action: (DETERMINE_RETRIEVAL_INSTRUCTION_WITH_CONTEXT, DETERMINE_RETRIEVAL_INPUTS_WITH_CONTEXT), 
                self.formulate_retrieval_queries_action: (QUERY_FORMULATION_INSTRUCTION, QUERY_FORMULATION_INPUT),
                self.identify_relevant_triples_action: (RELEVANT_TRIPLES_INSTRUCTION, RELEVANT_TRIPLES_INPUT),
            }
            instruction, input_format = PROMPT_MAP[action]
            try:
                chat = [{"role": "system", "content": instruction}, {"role": "user", "content": input_format.format(**state)}]
                return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            except:
                print(f"Critic model tokenizer {type(self.tokenizer).__name__} does not support chat template! Use normal tokenization instead.") 
                return instruction + "\n\n" + input_format.format(**state)
        
        def get_prompt_response_from_trajectory(self, trajectories: List[List[dict]], num_actions: int):

            prompts, responses = [], []
            action_mask = [] 
            for i, trajectory in enumerate(trajectories):
                trajectory_action_mask = [] 
                for action_item in trajectory[:num_actions]:
                    prompts.append(self.get_action_prompt(action_item["action"], action_item["state"]))
                    responses.append(action_item["response"])
                    trajectory_action_mask.append(1)
                while len(prompts) < (i+1) * num_actions:
                    # add action padding 
                    prompts.append("")
                    responses.append("")
                    trajectory_action_mask.append(0)
                action_mask.append(trajectory_action_mask)
            
            return prompts, responses, action_mask 

        def forward(self, trajectories: List[List[dict]], max_num_actions: int, max_prompt_length: int):

            batch_size = len(trajectories) 
            num_actions = min(max_num_actions, max([len(trajectory) for trajectory in trajectories]))
            prompts, responses, action_mask = self.get_prompt_response_from_trajectory(trajectories, num_actions)
            input_texts = [prompt + response + self.tokenizer.eos_token for prompt, response in zip(prompts, responses)]
            inputs = self.tokenizer(input_texts, max_length=max_prompt_length, padding=True, truncation=True, return_tensors="pt", padding_side="right") 

            model_device = getattr(getattr(self, self.base_model_prefix), "device")
            input_ids: torch.Tensor = inputs["input_ids"].to(model_device)
            attention_mask: torch.Tensor = inputs["attention_mask"].to(model_device)
            position_ids: torch.Tensor = attention_mask.long().cumsum(-1) - 1
            action_mask: torch.Tensor = torch.tensor(action_mask, dtype=torch.long, device=model_device)

            # outputs = getattr(self, self.base_model_prefix)(
            #     input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True
            # )
            # last_hidden_state: torch.Tensor = outputs.last_hidden_state # (batch_size * num_actions, max_prompt_length, hidden_size)

            # 划分小的batch size
            mini_batch_size = 4
            last_hidden_state_list = []
            for i in range(0, input_ids.shape[0], mini_batch_size):
                mini_batch_input_ids = input_ids[i:i+mini_batch_size]
                mini_batch_attention_mask = attention_mask[i:i+mini_batch_size]
                mini_batch_position_ids = position_ids[i:i+mini_batch_size]
                outputs = getattr(self, self.base_model_prefix)(
                    input_ids=mini_batch_input_ids, attention_mask=mini_batch_attention_mask, position_ids=mini_batch_position_ids, return_dict=True
                )
                last_hidden_state_list.append(outputs.last_hidden_state)
            last_hidden_state = torch.cat(last_hidden_state_list, dim=0)

            # extract eos token hidden state 
            eos_indices = attention_mask.shape[1] - attention_mask.long().fliplr().argmax(dim=1, keepdim=True) - 1 
            eos_hidden_states = last_hidden_state.gather(dim=1, index=eos_indices.unsqueeze(-1).expand(-1, -1, last_hidden_state.shape[-1])) # (batch_size * num_actions, 1, hidden_size)
            eos_hidden_states = eos_hidden_states.squeeze(1) # (batch_size * num_actions, hidden_size)
            
            # calculate value 
            values = getattr(self, self.value_head_prefix)(eos_hidden_states).reshape(batch_size, num_actions)

            # normalize value 
            if self.normalize_reward:
                values = (values - self.mean) / self.std  
            
            return values, action_mask
    
    return KGAdaptiveCriticModel


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s.startswith("yes"):
            return True
        if s.startswith("no"):
            return False
    return bool(v)


class QARewardModel(nn.Module):

    def __init__(self, data_dir: str, use_recall: bool = False, retriever: KGRetriever = None, **kwargs):
        super().__init__()
        self.data = self._load_data(data_dir)
        self.question_2_answers = self._load_question_2_answers()

        self.use_recall = use_recall
        if self.use_recall:
            self.question_2_gold_ctxids = self._load_question_2_gold_ctxids()
            self.determine_retrieval_action = "Determine Retrieval"
            self.formulate_retrieval_queries_action = "Formulate Retrieval Query"
            self.identify_relevant_triples_action = "Identify Relevant Triples" 
            # self.alpha, self.beta, self.lamb_r, self.lamb_q, self.lamb_s = 1.0, 0.5, 0.1, 0.3, 0.3 
            # self.retrieval_cost = 0.02 
            # self.miss_penalty = 0.05 
            self.alpha, self.beta = 0.05, 0.40 
            self.lamb_r, self.lamb_q, self.lamb_s = 0.30, 0.7, 0.3 
            self.retrieval_cost, self.miss_penalty = 0.02, 0.1 
            assert retriever is not None, "retriever must be provided when use_recall is True" 
            self.retriever = retriever 
        
        self.register_buffer("mean", torch.zeros(1), persistent=False)
        self.register_buffer("std", torch.ones(1), persistent=False)

    def _load_data(self, data_dir: str):
        data = []
        for file in ["train.json", "test.json"]:
            file_path = os.path.join(data_dir, file) 
            if not os.path.exists(file_path):
                continue 
            print(f"Loading Reward Model Data from {file_path} ...")
            data.extend(load_json(file_path))
        return data
    
    def _load_question_2_answers(self):
        question_2_answers = {}
        for example in self.data:
            question_2_answers[example["question"]] = example["answers"]
        return question_2_answers
    
    def _load_question_2_gold_ctxids(self): 
        question_2_gold_ctxids = {}
        for example in self.data:
            question = example["question"] 
            question_2_gold_ctxids[question] = set(ctx["id"] for ctx in example["gold_ctxs"])
        return question_2_gold_ctxids

    def get_answers(self, question: str):
        return self.question_2_answers[question]
    
    def compute_f1(self, predictions: List[str], labels: List[str]) -> torch.Tensor: 
        scores = [] 
        for prediction, label in zip(predictions, labels):
            if "<think>" in predictions[0] or "</think>" in predictions[0]:
                prediction_answer = extract_non_reasoning_model_answer(prediction)
            else:
                prediction_answer = extract_answer(prediction, mode="qa")
            f1 = max(f1_score(prediction_answer, answer)[0] for answer in label)
            scores.append(f1)
        return torch.tensor(scores)
    
    def compute_recall(self, gold_ctxids: set, retrieved_ctxids: set) -> float: 
        if len(gold_ctxids) == 0:
            return 0.0 
        return len(gold_ctxids.intersection(retrieved_ctxids)) / float(len(gold_ctxids))
    
    def forward_tree_rollout(self, prediction: str, label: List[str], question: str, ctxids: List[str]):

        f1 = self.compute_f1([prediction], [label])[0].item()
        gold_ctxids: set = self.question_2_gold_ctxids[question] 
        recall = self.compute_recall(gold_ctxids, set(ctxids))
        return 0.3 * f1 + 0.7 * recall 

    def forward(self, predictions: List[str], labels: List[List[str]], **kwargs):

        if not self.use_recall:
            return self.compute_f1(predictions, labels) 
        else:
            f1_score = self.compute_f1(predictions, labels)
            retrieval_trajectories = kwargs["retrieval_trajectories"]

            batch_size = len(retrieval_trajectories)
            max_num_actions = max(len(traj) - 1 for traj in retrieval_trajectories) # -1 是因为traj最后一个item是最终的context
            
            recall, step_reward = torch.zeros(batch_size).float(), torch.zeros((batch_size, max_num_actions)).float()

            for i, traj in enumerate(retrieval_trajectories):
                assert len(traj) >= 2 and traj[-1]["action"] == "final_context" 

                question = traj[0]["question"]
                gold_ctxids: set = self.question_2_gold_ctxids[question]
                final_ctxids = set(traj[-1]["final_context_ids"])
                recall[i] = self.compute_recall(gold_ctxids, final_ctxids) # traj最后一个item是最终的context

                retrieved_ctxids_on_question = set(ctx["id"] for ctx in self.retriever.retrieve([question], topk=5)[0])

                j = 0 
                while j < len(traj) - 1: 
                    item = traj[j] 
                    if item["action"] == self.determine_retrieval_action:
                        current_ctxids = set(item["current_context_ids"])
                        coverage_before = self.compute_recall(gold_ctxids, current_ctxids)
                        require_retrieval = to_bool(item["require_retrieval"])

                        k = j + 1 
                        last_query_retrieved_ids = [] 
                        while k < len(traj) - 1 and traj[k]["action"] != self.determine_retrieval_action:

                            if traj[k]["action"] == self.formulate_retrieval_queries_action:
                                query = traj[k]["query"]
                                if query is None:
                                    query = question # dummy placeholder
                                remaining_gold_ctxids = gold_ctxids - current_ctxids  
                                retrieved_ctxids = [ctx["id"] for ctx in self.retriever.retrieve([query], topk=5)[0]]
                                last_query_retrieved_ids = retrieved_ctxids
                                r0 = self.compute_recall(remaining_gold_ctxids, retrieved_ctxids_on_question)
                                rq = self.compute_recall(remaining_gold_ctxids, set(retrieved_ctxids))
                                step_reward[i, k] = self.lamb_q * math.sqrt(max(0, rq - r0))
                            
                            elif traj[k]["action"] == self.identify_relevant_triples_action:
                                selected_ctxid = traj[k]["selected_context_id"] 
                                remaining_gold_ctxids = gold_ctxids - current_ctxids 
                                if selected_ctxid in remaining_gold_ctxids:
                                    rank = None 
                                    if last_query_retrieved_ids:
                                        try:
                                            rank = last_query_retrieved_ids.index(selected_ctxid) + 1 
                                        except:
                                            rank = None 
                                    if rank is not None:
                                        gain = 1.0 / math.log2(rank+1)
                                    else:
                                        gain = 1.0 
                                    step_reward[i, k] = self.lamb_s * gain 
                                else:
                                    step_reward[i, k] = -self.miss_penalty 
                            
                            k += 1 

                        if k < len(traj) - 1 and traj[k]["action"] == self.determine_retrieval_action:
                            next_ctxids = set(traj[k]["current_context_ids"])
                        else:
                            next_ctxids = final_ctxids 
                        coverage_after = self.compute_recall(gold_ctxids, next_ctxids)
                        gain = coverage_after - coverage_before 
                        cost = self.retrieval_cost if require_retrieval else 0.0 
                        step_reward[i, j] = self.lamb_r * gain - cost 
                        j = k 
                    
                    else:
                        j += 1 
            
            sequence_reward = self.alpha * f1_score + self.beta * recall
            total_reward = sequence_reward + step_reward.sum(dim=-1)
            return {"sequence_reward": sequence_reward, "step_reward": step_reward, "total_reward": total_reward} 


