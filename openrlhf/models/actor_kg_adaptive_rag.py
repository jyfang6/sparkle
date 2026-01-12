import os 
import json  
import regex
import pickle 
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Dict
from copy import deepcopy 
from flash_attn.utils.distributed import all_gather
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openrlhf.models.actor import Actor
from openrlhf.models.utils import log_probs_from_logits 
from prompts.adaptive_rag import * 
from generator.generator import Generator 
from utils.utils import escape_braces, add_eos_token_at_the_end
from retriever.e5 import get_e5_embeddings_for_query, get_e5_embeddings_for_document 
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from torch import Tensor
from setup.setup import HF_TOKEN

import sys 
sys.path.append("/nfs/common")
from my_evaluation import f1_score 

retrieval_query_output_format = """Step 1: Reasoning Chain:
{reasoning_chain}

Step 2: Knowledge Gaps:
{knowledge_gaps}

Step 3: Retrieval Query:
{retrieval_query}"""


def get_tokenizer(model_name):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        padding_side="left",
        token=HF_TOKEN,
    )
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


class ReasoningModel(Generator):

    def __init__(
        self, 
        reasoning_model: str, 
        use_flash_attention_2: bool = False, 
        bf16: bool = True, 
        device_map: dict = None,
        max_reasoning_steps: int = 12, 
        max_retrieval_count: int = 5, 
        max_request_times: int = 8, 
        topk: int = 10, 
        num_candidate_triples: int = 25,
        max_num_context: int = 5, 
        # max_num_actions: int = 18, 
        **kwargs 
    ):
        # super().__init__()
        attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
        if reasoning_model in ["deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"]:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, 
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        else:
            quantization_config = None
        tokenizer = get_tokenizer(reasoning_model)
        model = AutoModelForCausalLM.from_pretrained(
            reasoning_model, 
            trust_remote_code=True, 
            attn_implementation=attn_implementation, 
            quantization_config=quantization_config, 
            # quantization_config=BitsAndBytesConfig(
            #     load_in_8bit=True, 
            #     llm_int8_threshold=6.0,
            #     llm_int8_has_fp16_weight=False
            # ),
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            # ),
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map=device_map, 
            token=HF_TOKEN,
        )
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        super().__init__(
            tokenizer=tokenizer, 
            generator=model, 
            # max_length=2048, 
            max_length=4096, 
            max_new_tokens=256, 
            batch_size=4, 
            **kwargs
        )
        # self.to_device("cpu")
        self.max_reasoning_steps = max_reasoning_steps
        self.max_retrieval_count = max_retrieval_count
        self.max_request_times = max_request_times
        self.topk = topk
        self.num_candidate_triples = num_candidate_triples 
        self.max_num_context = max_num_context
        # self.max_num_actions = max_num_actions 
        if reasoning_model in ["deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"]:
            self.is_long_reasoning_model = True 
            self.generation_config = {"top_k": 20, "top_p": 0.8, "do_sample": True, "temperature": 0.7}
        elif reasoning_model in ["Qwen/Qwen2.5-7B-Instruct"]:
            self.is_long_reasoning_model = False 
            self.generation_config = {} # 空的话默认就是greedy search
        else:
            raise ValueError(f"Can't determine whether it is a long reasoning model or not for {reasoning_model}!") 

    
    def generate(self, prompts: List[str], generation_config: dict, stop_words: List[str] = None) -> List[str]:
        
        inputs = self.tokenizer_encode(prompts=prompts)
        # outputs = super().generate(inputs, stop_words=stop_words, **generation_config)[0]
        outputs = super().generate(inputs, stop_words=stop_words, **self.generation_config)[0]
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts 
    
    # def to_device(self, device: torch.device):
    #     self.generator.to(device)


class KGRetriever:

    def __init__(self, corpus_path: str, max_query_length: int = 256, max_doc_length: int = 256):

        self.corpus_path = corpus_path 
        # 加载corpus
        print(f"Loading corpus from {self.corpus_path}")
        with open(os.path.join(self.corpus_path, "corpus.json"), encoding="utf-8") as f:
            self.corpus = json.load(f)
        print(f"Corpus loaded with {len(self.corpus)} documents")
        self.id2doc = {doc["id"]: doc for doc in self.corpus}
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

        # 获取KG Triples 
        print(f"Loading KG Triples from {self.corpus_path}")
        self.cached_kg_triples = pickle.load(open(os.path.join(self.corpus_path, "cached_kg_triples.pkl"), "rb"))
        
        # 计算documents的embedding
        self.documents_embeddings = self._calculate_documents_embeddings() 
    
    def _calculate_documents_embeddings(self):

        def get_document_text(doc: dict) -> str:
            text = doc.get("text", None)
            if text is not None:
                return text
            return " ".join([sent.strip() for sent in doc.get("sentences", [])])
        
        cache_embeddings_path = os.path.join(self.corpus_path, "cached_documents_embeddings.pkl")
        if os.path.exists(cache_embeddings_path):
            return pickle.load(open(cache_embeddings_path, "rb"))
        
        documents_texts = ["Title: {} Text: {}".format(doc["title"], get_document_text(doc)) for doc in self.corpus]
        embeddings = get_e5_embeddings_for_document(doc_list=documents_texts, max_length=self.max_doc_length, batch_size=64)
        pickle.dump(embeddings, open(cache_embeddings_path, "wb"))
        return embeddings 
    
    def _calculate_kg_triples_embeddings(self):

        cache_embeddings_path = os.path.join(self.corpus_path, "cached_kg_triples_embeddings.pkl")
        if os.path.exists(cache_embeddings_path):
            return pickle.load(open(cache_embeddings_path, "rb"))
        
        triples_texts = [triple["text"] for triple in self.kg_triples]
        embeddings = get_e5_embeddings_for_document(doc_list=triples_texts, max_length=self.max_doc_length, batch_size=64)
        pickle.dump(embeddings, open(cache_embeddings_path, "wb"))

        return embeddings

    def retrieve(self, queries: List[str], topk: int = 10) -> List[List[dict]]:

        query_embeddings = get_e5_embeddings_for_query(query_list=queries, max_length=self.max_query_length, batch_size=4)
        similarities = torch.matmul(query_embeddings, self.documents_embeddings.T)
        topk_values, topk_indices = torch.topk(similarities, k=topk, dim=-1)
        topk_values = topk_values.tolist()
        topk_indices = topk_indices.tolist()
        retrieval_results = []
        for values, indices in zip(topk_values, topk_indices):
            one_retrieval_result = [] 
            for value, idx in zip(values, indices):
                doc = deepcopy(self.corpus[idx])
                doc["score"] = value
                one_retrieval_result.append(doc)
            retrieval_results.append(one_retrieval_result)
        return retrieval_results

    def get_candidate_triples_from_documents(self, documents: List[dict]) -> List[dict]:
        """
        Input: 
            documents: [{"id": str, "title": str, "text": str / "sentence": str, "triples": ["text": str, "sentence": int]}]
        Output:
            triples: [{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...]
        """
        triples = [] 
        for doc in documents:
            docid = doc["id"]
            title = doc["title"]
            for triple in self.cached_kg_triples[docid]["triples"]:
                triple = {
                    "title": title,
                    "text": triple["text"],
                    "reference": [docid, triple["sentence"]]
                }
                triples.append(triple)
        
        return triples
    
    def filter_triples(self, question: str, query: str, triples: List[dict], topk: int = 25) -> List[dict]:

        num_triples = len(triples)
        triples_texts = [triple["text"] for triple in triples]
        triples_embeddings = get_e5_embeddings_for_document(doc_list=triples_texts, max_length=self.max_doc_length, batch_size=4)
        query_embeddings = get_e5_embeddings_for_query(query_list=[question + " " + query], max_length=self.max_query_length, batch_size=4)
        query_triple_similarities = torch.matmul(query_embeddings, triples_embeddings.T)
        topk_relevant_triples_scores, topk_relevant_triples_indices = torch.topk(
            query_triple_similarities, k = min(topk, num_triples), dim=1
        )

        # [0]是因为只有一个query
        topk_relevant_triples_indices = topk_relevant_triples_indices.tolist()[0]
        topk_relevant_triples_scores = topk_relevant_triples_scores.tolist()[0]
        
        # return topk_relevant_triples_indices, topk_relevant_triples_scores
        filtered_triples = [triples[i] for i in topk_relevant_triples_indices]
        return filtered_triples, topk_relevant_triples_scores

    def get_documents_from_triples(self, triples: List[List[dict]], source_documents: List[List[dict]]) -> List[List[dict]]:

        documents = []
        for one_triple_list, one_document_list in zip(triples, source_documents):
            docids = [triple["reference"][0] for triple in one_triple_list] 
            id2doc = {doc["id"]: doc for doc in one_document_list}
            documents.append([id2doc[docid] for docid in docids])
        return documents 


class TreeNode:
    """树节点类，用于存储rollout过程中的状态"""
    def __init__(self, result: dict, parent=None, branch_type: str = "root"):
        self.result = deepcopy(result)  # 当前节点的状态
        self.parent = parent
        self.children = []
        self.branch_type = branch_type  # "root", "retrieval_yes", "retrieval_no", "query_branch", "triple_branch"
        self.is_expanded = False
        self.is_finished = False

        self.is_pruned = False 
        self.pruned_similarity = None  
        self.corresponding_non_retrieval_node = None 
        
    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)
        
    def get_all_leaf_nodes(self):
        """获取所有叶子节点"""
        if not self.children or self.is_finished:
            return [self]
            
        leaf_nodes = []
        for child in self.children:
            if child.is_pruned:
                continue 
            leaf_nodes.extend(child.get_all_leaf_nodes())
        return leaf_nodes
        
    def get_depth(self):
        """获取节点深度"""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth


class ActorKGAdaptiveRAG(Actor):

    def __init__(
        self, 
        pretrain_or_model: str,
        reasoning_model: ReasoningModel,
        retriever: KGRetriever, 
        prompt_max_len: int = 2048,
        generate_max_len: int = 160,
        use_flash_attention_2: bool = False,
        bf16: bool = True,
        load_in_4bit: bool = False,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        target_modules: list[str] = None,
        ds_config: dict = None,
        device_map: dict = None,
        packing_samples: bool = False,
        max_num_actions: int = 20, 
        **kwargs,
    ):
        super(Actor, self).__init__()
        attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

        if load_in_4bit:
            assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            nf4_config = None
        
        self.tokenizer = get_tokenizer(pretrain_or_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrain_or_model,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            device_map=device_map,
            token=HF_TOKEN,
        )

        self._reasoning_model = reasoning_model
        self._retriever = retriever 
        self.prompt_max_len = prompt_max_len
        self.generate_max_len = generate_max_len
        self.max_num_actions = max_num_actions 
        self.enable_retrieval_symbol = "Yes (retrieval is required)" # "Yes"
        self.disable_retrieval_symbol = "No (retrieval is not required)" # "No" 
        self.determine_retrieval_action = "Determine Retrieval"
        self.formulate_retrieval_queries_action = "Formulate Retrieval Query"
        self.identify_relevant_triples_action = "Identify Relevant Triples" 
        self.action_type_map = { # 0表示用于padding的action
            self.determine_retrieval_action: 1, 
            self.formulate_retrieval_queries_action: 2, 
            self.identify_relevant_triples_action: 3,
        }

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def get_action_prompt(self, action: str, state: dict) -> str:

        PROMPT_MAP = {
            # self.determine_retrieval_action: (DETERMINE_RETRIEVAL_INSTRUCTION, DETERMINE_RETRIEVAL_INPUTS),
            self.determine_retrieval_action: (DETERMINE_RETRIEVAL_INSTRUCTION_WITH_CONTEXT, DETERMINE_RETRIEVAL_INPUTS_WITH_CONTEXT),
            self.formulate_retrieval_queries_action: (QUERY_FORMULATION_INSTRUCTION, QUERY_FORMULATION_INPUT),
            self.identify_relevant_triples_action: (RELEVANT_TRIPLES_INSTRUCTION, RELEVANT_TRIPLES_INPUT),
        }
        instruction, input_format = PROMPT_MAP[action]
        chat = [{"role": "system", "content": instruction}, {"role": "user", "content": input_format.format(**state)}]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    def get_prompt_response_from_trajectory(self, trajectories: List[List[dict]], num_actions: int):

        prompts, responses = [], []
        action_mask, action_type = [], [] 
        for i, trajectory in enumerate(trajectories):
            trajectory_action_mask, trajectory_action_type = [], [] 
            for action_item in trajectory[:num_actions]:
                prompts.append(self.get_action_prompt(action_item["action"], action_item["state"]))
                responses.append(action_item["response"])
                trajectory_action_mask.append(1)
                trajectory_action_type.append(self.action_type_map[action_item["action"]])
            while len(prompts) < (i+1) * num_actions:
                # add action padding 
                prompts.append("")
                responses.append("")
                trajectory_action_mask.append(0)
                trajectory_action_type.append(0)
            action_mask.append(trajectory_action_mask)
            action_type.append(trajectory_action_type) 
        
        return prompts, responses, action_mask, action_type

    def collate_func(self, trajectories: List[List[dict]], num_actions: int):

        # prompts: (batch_size * num_actions, ); responses: (batch_size * num_actions, ); action_mask: (batch_size, num_actions); action_type: (batch_size, num_actions) 
        prompts, responses, action_mask, action_type = self.get_prompt_response_from_trajectory(trajectories, num_actions)
        # right padding
        inputs = self.tokenizer(prompts, max_length=self.prompt_max_len, padding=True, truncation=True, return_tensors="pt", padding_side="right")
        input_ids, input_attention_mask = inputs["input_ids"], inputs["attention_mask"]

        # right padding
        outputs = self.tokenizer(responses, max_length=self.generate_max_len, padding=True, truncation=True, return_tensors="pt", padding_side="right", add_special_tokens=False)
        output_ids, output_attention_mask = add_eos_token_at_the_end(outputs["input_ids"], outputs["attention_mask"], self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)

        # NOTE: Qwen2.5的Flash Attention 2 只支持 left padding
        sequences = torch.zeros((input_ids.shape[0], input_ids.shape[1] + output_ids.shape[1]), dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(sequences, dtype=torch.long)
        response_mask = torch.zeros_like(sequences, dtype=torch.long)
        for i, (one_input_ids, one_input_attention_mask, one_output_ids, one_output_attention_mask) in enumerate(
            zip(input_ids, input_attention_mask, output_ids, output_attention_mask)
        ):
            input_token_ids = one_input_ids[one_input_attention_mask == 1]
            output_token_ids = one_output_ids[one_output_attention_mask == 1]
            token_ids = torch.cat([input_token_ids, output_token_ids], dim=0)
            # sequences[i, :len(token_ids)] = token_ids
            # attention_mask[i, :len(token_ids)] = 1
            # response_mask[i, len(input_token_ids): len(token_ids)] = 1
            sequences[i, -len(token_ids):] = token_ids
            attention_mask[i, -len(token_ids):] = 1
            response_mask[i, -len(output_token_ids):] = 1
        
        # truncate sequences to remove additional padding
        # end_idx = attention_mask.sum(0).nonzero(as_tuple=True)[0][-1].item() + 1
        # sequences = sequences[:, :end_idx]
        # attention_mask = attention_mask[:, :end_idx]
        # response_mask = response_mask[:, :end_idx]
        start_idx = attention_mask.sum(0).nonzero(as_tuple=True)[0][0].item()
        sequences = sequences[:, start_idx:]
        attention_mask = attention_mask[:, start_idx:]
        response_mask = response_mask[:, start_idx:]

        action_mask: Tensor = torch.tensor(action_mask, dtype=torch.long)
        action_type: Tensor = torch.tensor(action_type, dtype=torch.long) 
        return sequences, attention_mask, response_mask, action_mask, action_type

    def forward(self, trajectories: List[List[dict]]):

        batch_size = len(trajectories)
        num_actions = min(self.max_num_actions, max([len(trajectory) for trajectory in trajectories]))

        # prepare inputs 
        # sequences, attention_mask, response_mask: (n_traj * num_actions, num_tokens) 
        # action_mask, action_type: (n_traj, num_actions)
        sequences, attention_mask, response_mask, action_mask, action_type = self.collate_func(trajectories, num_actions)
        device = self.model.device 
        position_ids = attention_mask.long().cumsum(-1) - 1 
        sequences, attention_mask, position_ids = sequences.to(device), attention_mask.to(device), position_ids.to(device)
        response_mask, action_mask, action_type = response_mask.to(device), action_mask.to(device), action_type.to(device) 

        # model_output = self.model(input_ids=sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True)
        # logits = model_output.logits.to(torch.float32)

        # 对每个trajectory单独前向传播，并且划分小的batch size来计算action的对数概率
        all_action_log_probs_list = [] 
        all_action_per_token_log_probs_list = [] 
        for traj_idx in range(batch_size):
            traj_sequences = sequences[traj_idx*num_actions: (traj_idx+1)*num_actions]
            traj_attention_mask = attention_mask[traj_idx*num_actions: (traj_idx+1)*num_actions]
            traj_position_ids = position_ids[traj_idx*num_actions: (traj_idx+1)*num_actions] 
            traj_response_mask = response_mask[traj_idx*num_actions: (traj_idx+1)*num_actions]
            mini_batch_size = 4 

            traj_action_log_probs_list = []
            for i in range(0, traj_sequences.shape[0], mini_batch_size):
                mini_batch_traj_sequences = traj_sequences[i:i+mini_batch_size]
                mini_batch_traj_attention_mask = traj_attention_mask[i:i+mini_batch_size]
                mini_batch_traj_position_ids = traj_position_ids[i:i+mini_batch_size]
                mini_batch_traj_response_mask = traj_response_mask[i:i+mini_batch_size, 1:]
                traj_model_output = self.model(
                    input_ids = mini_batch_traj_sequences,
                    attention_mask = mini_batch_traj_attention_mask, 
                    position_ids = mini_batch_traj_position_ids,
                    return_dict = True
                )
                # logits_list.append(traj_model_output.logits.to(torch.float32))
                mini_batch_logits = traj_model_output.logits.to(torch.float32)
                mini_batch_log_probs = log_probs_from_logits(mini_batch_logits[:, :-1, :], mini_batch_traj_sequences[:, 1:])
                
                # alpha = 1.0 
                # mini_batch_response_length = mini_batch_traj_response_mask.sum(-1).clamp_min(1)
                # mini_batch_action_log_probs = torch.sum(mini_batch_log_probs * mini_batch_traj_response_mask, dim=-1) / (mini_batch_response_length.float()**alpha)

                mini_batch_action_log_probs = torch.sum(mini_batch_log_probs * mini_batch_traj_response_mask, dim=-1)
                traj_action_log_probs_list.append(mini_batch_action_log_probs)
                all_action_per_token_log_probs_list.append(mini_batch_log_probs * mini_batch_traj_response_mask)

                if not self.model.training:
                    del mini_batch_logits, mini_batch_log_probs, traj_model_output
                    torch.cuda.empty_cache() 
        
            traj_action_log_probs = torch.cat(traj_action_log_probs_list, dim=0).reshape(1, num_actions)
            all_action_log_probs_list.append(traj_action_log_probs)
        
        action_log_probs = torch.cat(all_action_log_probs_list, dim=0)
        action_per_token_log_probs = torch.cat(all_action_per_token_log_probs_list, dim=0).reshape(batch_size, num_actions, -1)
        action_per_token_log_probs_mask = response_mask[:, 1:].reshape(batch_size, num_actions, -1) 

        """
        mini_batch_size = 4
        logits_list = []
        for i in range(0, sequences.shape[0], mini_batch_size):
            mini_batch_sequences = sequences[i:i+mini_batch_size]
            mini_batch_attention_mask = attention_mask[i:i+mini_batch_size]
            mini_batch_position_ids = position_ids[i:i+mini_batch_size]
            model_output = self.model(input_ids=mini_batch_sequences, attention_mask=mini_batch_attention_mask, position_ids=mini_batch_position_ids, return_dict=True)
            logits_list.append(model_output.logits.to(torch.float32))
        logits = torch.cat(logits_list, dim=0)

        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:]) # (batch_size x num_actions, seq_len - 1)
        action_log_probs = log_probs.reshape(batch_size, num_actions, -1)
        response_mask = response_mask[:, 1:].reshape(batch_size, num_actions, -1).to(device=device)
        
        # version 1:直接对token的对数概率求和
        # action_log_probs = torch.sum(action_log_probs * response_mask, dim=-1) # (batch_size, num_actions) 

        # version 2: 下面的计算中考虑到不同action输出长度的影响, alpha设置为1的时候严格平均，0.5的时候长序列的影响会减小但不至于过小，为0的时候相当于直接求和
        alpha = 1.0 
        response_length = response_mask.sum(dim=-1).clamp_min(1) 
        action_log_probs = torch.sum(action_log_probs * response_mask, dim=-1) / (response_length.float()**alpha)
        """
        # action_log_probs, action_mask, action_type: (n_traj, num_actions) 
        # action_per_token_log_probs, action_per_token_log_probs_mask: (n_traj, num_actions, num_tokens)
        return action_log_probs, action_mask, action_type, action_per_token_log_probs, action_per_token_log_probs_mask

    @torch.no_grad()
    def actor_generate(self, instructions: List[str], prompts: List[str], generation_config: dict):
        chats = [
            [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}]
            for instruction, prompt in zip(instructions, prompts)
        ]
        prompts = self.tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompts, max_length=self.prompt_max_len, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        outputs = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=self.generate_max_len,
            **generation_config 
        )
        generated_ids = outputs[:, input_ids.shape[1]:].detach().cpu()
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return texts 

    @torch.no_grad()
    def is_retrieval_required(self, questions: List[str], thoughts: List[str], generation_config: dict) -> List[str]:

        instructions = [DETERMINE_RETRIEVAL_INSTRUCTION] * len(questions)
        prompts = [
            DETERMINE_RETRIEVAL_INPUTS.format(question=question, thought=thought) 
            for question, thought in zip(questions, thoughts)
        ]
        generated_texts = self.actor_generate(instructions, prompts, generation_config)
        # 解析生成的文本
        outputs = [self.enable_retrieval_symbol if "yes" in text.lower() else self.disable_retrieval_symbol for text in generated_texts]
        return outputs 
    
    @torch.no_grad()
    def is_retrieval_required_with_context(self, questions: List[str], thoughts: List[str], current_thoughts: List[str], contexts: List[List[str]], generation_config:dict) -> List[str]:
        
        instructions = [DETERMINE_RETRIEVAL_INSTRUCTION_WITH_CONTEXT] * len(questions) 
        # 已经在get_context中truncate了
        # contexts = [
        #     [text if len(text.split())<=160 else " ".join(text.split()[:250]) for text in context]
        #     for context in contexts
        # ]
        prompts = [
            DETERMINE_RETRIEVAL_INPUTS_WITH_CONTEXT.format(
                retrieved_context = "\n\n".join(context), 
                question=question, 
                reasoning_history=thought, 
                current_reasoning_step=current_thought
            )
            for question, thought, current_thought, context in zip(questions, thoughts, current_thoughts, contexts)
        ]
        generated_texts = self.actor_generate(instructions, prompts, generation_config)
        # 解析生成的文本 
        # print(f"Determine Retrieval Response: {generated_texts}")
        for prompt, text in zip(prompts, generated_texts):
            if text.strip().lower() not in ["yes (retrieval is required)", "no (retrieval is not required)"]:
                print(f"Prompt for unusual outputs:\n{prompt}")
        outputs = [self.enable_retrieval_symbol if "yes" in text.lower() else self.disable_retrieval_symbol for text in generated_texts]
        return outputs 
    
    @torch.no_grad()
    def formulate_retrieval_queries(self, questions: List[str], thoughts: List[str], generation_config: dict) -> List[str]:

        instructions = [QUERY_FORMULATION_INSTRUCTION] * len(questions)
        prompts = [
            QUERY_FORMULATION_INPUT.format(question=question, thought=thought)
            for question, thought in zip(questions, thoughts)
        ]
        generated_texts = self.actor_generate(instructions, prompts, generation_config)
        # 解析生成的文本
        parsed_items = [self.parse_retrieval_query_response(question, text) for question, text in zip(questions, generated_texts)]
        reasoning_chains = [item["reasoning_chain"] for item in parsed_items]
        queries = [item["query"] for item in parsed_items]
        responses = [item["response"] for item in parsed_items]
        return reasoning_chains, queries, responses
    
    @torch.no_grad()
    def identify_relevant_triples(self, questions: List[str], reasoning_chains: List[str], queries: List[str], triples: List[List[dict]], generation_config: dict):

        triples_texts = []
        for one_triple_list in triples:
            one_triple_text_list = []
            for triple in one_triple_list:
                one_triple_text_list.append(f"Source Title: {triple['title']}\nText: {triple['text']}")
            triples_texts.append("\n".join([f"{i+1}. {text}" for i, text in enumerate(one_triple_text_list)]))

        instructions = [RELEVANT_TRIPLES_INSTRUCTION] * len(questions)
        prompts = [
            RELEVANT_TRIPLES_INPUT.format(question=question, reasoning_chain=reasoning_chain, query=query, candidate_triples=triples_text) 
            for question, reasoning_chain, query, triples_text in zip(questions, reasoning_chains, queries, triples_texts) 
        ]
        generated_texts = self.actor_generate(instructions, prompts, generation_config)

        relevant_triples, responses = [], []
        for text, one_triple_list in zip(generated_texts, triples):
            selected_triples, response = self.parse_relevant_triples_response(text, one_triple_list)
            relevant_triples.append(selected_triples)
            responses.append(response)
        
        return relevant_triples, triples_texts, responses

    def parse_retrieval_query_response(self, question: str, response: str) -> dict:

        default_result = {"reasoning_chain": "", "knowledge_gaps": "", "query": "", "response": ""}

        response = response.replace("**", "")
        failed = False
        # 匹配Step 1到下一个Step之间的所有内容
        reasoning_pattern = r"Step 1: Reasoning Chain:?\s*(.*?)(?=\n\s*Step|$)"
        reasoning_match = regex.search(reasoning_pattern, response, regex.DOTALL)
        if reasoning_match:
            reasoning_chain = reasoning_match.group(1).strip()
            default_result["reasoning_chain"] = reasoning_chain
        else:
            failed = True
            default_result["reasoning_chain"] = "Failed to extract reasoning chain: " + response

        # 匹配Step 2到下一个Step之间的所有内容
        knowledge_gaps_pattern = r"Step 2: Knowledge Gaps:?\s*(.*?)(?=\n\s*Step|$)"
        knowledge_gaps_match = regex.search(knowledge_gaps_pattern, response, regex.DOTALL)
        if knowledge_gaps_match:
            knowledge_gaps = knowledge_gaps_match.group(1).strip()
            default_result["knowledge_gaps"] = knowledge_gaps
        else:
            failed = True
            default_result["knowledge_gaps"] = "Failed to extract knowledge gaps: " + response
        
        # 匹配Step 3到结尾的内容
        query_pattern = r"Step 3:.*?Query:?\s*(.*?)$"
        query_match = regex.search(query_pattern, response, regex.DOTALL)
        if query_match:
            query = query_match.group(1).strip()
            default_result["query"] = query
            if "none" in query.lower():
                default_result["query"] = None
        else:
            failed = True
            default_result["query"] = f"Question: {question}\n{response}"

        if failed:
            default_result["response"] = response 
        else:
            default_result["response"] = retrieval_query_output_format.format(
                reasoning_chain=default_result["reasoning_chain"],
                knowledge_gaps=default_result["knowledge_gaps"],
                retrieval_query=default_result["query"]
            )
        return default_result
    
    def parse_relevant_triples_response(self, response: str, triples: List[dict]) -> List[dict]:

        triples_texts_list = []
        angle_bracket_pattern = r'<([^>]+)>'
        matches = regex.finditer(angle_bracket_pattern, response)
        for mat in matches:
            triple_text = mat.group(1).strip()
            if triple_text:
                triples_texts_list.append("<" + triple_text + ">")
        
        if not triples_texts_list:
            triples_texts_list.append(response)
        
        selected_triples = []
        selected_triples_indices = set()
        for triple_text in triples_texts_list:
            similarities = [f1_score(triple_text, triple["text"])[0] for triple in triples]
            max_similarity_index = similarities.index(max(similarities))
            if max_similarity_index not in selected_triples_indices:
                selected_triples.append(triples[max_similarity_index])
                selected_triples_indices.add(max_similarity_index)
        
        # 得到response 
        response = triples_texts_list[0] 

        return selected_triples, response

    def get_context(self, document: dict) -> str:
        text = document.get("text", None)
        if not text:
            text = " ".join([s.strip() for s in document.get("sentences", [])])
        # truncate text: 
        text = text if len(text.split())<=120 else " ".join(text.split()[:120])
        return "Title: {}\nText: {}".format(document["title"], text)
    
    def get_context_from_docids(self, docids_to_scores: Dict[str, float], result: dict) -> List[str]:

        sorted_docids = [docid for docid, _ in sorted(docids_to_scores.items(), key=lambda x: x[1], reverse=True)]
        sorted_docids = sorted_docids[:self._reasoning_model.max_num_context]
        sorted_documents = [self._retriever.id2doc[docid] for docid in sorted_docids]
        context = [self.get_context(document) for document in sorted_documents]
        
        # 根据context window动态地调整context的数量，同时也加入一点随机性在里面
        final_context = [] 
        for num_doc in range(len(sorted_documents), 0, -1):
            candidate_prompt_text = "\n\n".join(context[:num_doc]) + "\n\n" + result["question"] + "\n\n" + result["existing_thought"] + "\n\n" + result["current_thought"]
            if len(self.tokenizer.encode(candidate_prompt_text)) < 600:
                final_context = context[:num_doc]
                break 
        return final_context
    
    def get_context_ids(self, docids_to_scores):
        sorted_docids = [docid for docid, _ in sorted(docids_to_scores.items(), key=lambda x: x[1], reverse=True)]
        sorted_docids = sorted_docids[:self._reasoning_model.max_num_context]
        return sorted_docids
    
    def convert_non_reasoning_model_thought_to_thought(self, thought: str) -> str:

        segments = thought.split("</think>")
        segments = [seg.replace("<think>", "").strip() for seg in segments if seg.strip()]
        # return "\n\n".join(segments) + "\n\n"
        output = "\n\n".join(segments)
        if output:
            output += "\n\n"
        return output 
    
    """
    @torch.no_grad()
    def generate(self, questions: List[str], **kwargs):

        generation_args = {
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 1.0)
        }

        if self._reasoning_model.is_long_reasoning_model:
            prompts = self._reasoning_model.prompt(
                inputs=[ANSWER_PROMPT.format(question=question) for question in questions],
                use_chat=True, 
            )
        else:
            prompts = self._reasoning_model.prompt(
                inputs=[ANSWER_PROMPT_FOR_NON_REASONING_MODEL.format(question=question) for question in questions], 
                use_chat=True
            )

        results = []
        for question, prompt in zip(questions, prompts):
            results.append(
                {
                    "question": question, 
                    "prompt": prompt, 
                    "output": "", 
                    "context": [], 
                    "finished": False, 
                    "trajectory": [],
                    "docids_to_scores": {}, 
                    "step": 0, 
                    "request_times": 0, 
                    "retrieval_count": 0 
                }
            )

        while True:

            results_not_finished = []
            for result in results:
                if not result["finished"] and result["step"] < self._reasoning_model.max_reasoning_steps:
                    results_not_finished.append(result)
            
            if not results_not_finished:
                break 
            
            if results_not_finished[0]["step"] == 0:
                generated_texts = [
                    f"Alright, so I need to figure out {question} Let me start by breaking down the question.\n\n"
                    for question in questions
                ]
            else:
                input_prompts = []
                for result in results_not_finished:
                    input_prompt = result["prompt"].format(context="\n\n".join(result["context"]))
                    input_prompts.append(input_prompt)
                generated_texts = self._reasoning_model.generate(
                    prompts=input_prompts, 
                    generation_config=generation_args,
                    stop_words=["\n\n", self._reasoning_model.tokenizer.eos_token]
                )
                generated_texts = [escape_braces(text) for text in generated_texts]

            results_finish_thinking, texts_finish_thinking = [], []
            results_not_finish_thinking, texts_not_finish_thinking = [], [] 
            for result, generated_text in zip(results_not_finished, generated_texts):
                if "</think>" in generated_text:
                    results_finish_thinking.append(result)
                    texts_finish_thinking.append(generated_text)
                else:
                    results_not_finish_thinking.append(result)
                    texts_not_finish_thinking.append(generated_text)

            if results_finish_thinking:
                for result, text in zip(results_finish_thinking, texts_finish_thinking):
                    result["prompt"] += text 
                    result["output"] += text 
                input_prompts = [] 
                for result in results_finish_thinking:
                    input_prompt = result["prompt"].format(context="\n\n".join(result["context"]))
                    input_prompts.append(input_prompt)
                answer_texts = self._reasoning_model.generate(
                    prompts=input_prompts, 
                    generation_config=generation_args
                )
                for result, answer_text in zip(results_finish_thinking, answer_texts):
                    result["prompt"] += answer_text 
                    result["output"] += answer_text 
                    result["finished"] = True

            results_not_finished = results_not_finish_thinking
            generated_texts = texts_not_finish_thinking
            if not results_not_finished:
                break 
            for result, generated_text in zip(results_not_finished, generated_texts):
                result["prompt"] += generated_text 
                result["output"] += generated_text 
                result["step"] += 1

            results_can_retrieve = []
            for result in results_not_finished:
                if result["retrieval_count"] < self._reasoning_model.max_retrieval_count and \
                    result["request_times"] < self._reasoning_model.max_request_times:
                    results_can_retrieve.append(result)
            
            if not results_can_retrieve:
                continue

            is_retrieval_required = self.is_retrieval_required(
                questions = [result["question"] for result in results_can_retrieve], 
                thoughts = [result["output"] for result in results_can_retrieve], 
                generation_config=generation_args
            )

            for result in results_can_retrieve:
                result["request_times"] += 1 

            # 记录中间状态
            results_require_retrieval = []
            for result, require_retrieval in zip(results_can_retrieve, is_retrieval_required):
                result["trajectory"].append(
                    {
                        "action": self.determine_retrieval_action, 
                        "state": {
                            "question": result["question"], 
                            "thought": result["output"], 
                        },
                        "response": require_retrieval
                    }
                )
                if require_retrieval == self.enable_retrieval_symbol:
                    results_require_retrieval.append(result)

            if not results_require_retrieval:
                continue

            # 构造检索的query
            reasoning_chains, queries, responses = self.formulate_retrieval_queries(
                questions = [result["question"] for result in results_require_retrieval], 
                thoughts = [result["output"] for result in results_require_retrieval], 
                generation_config=generation_args
            )

            # 记录中间状态
            for result, response in zip(results_require_retrieval, responses):
                result["trajectory"].append(
                    {
                        "action": self.formulate_retrieval_queries_action, 
                        "state": {
                            "question": result["question"], 
                            "thought": result["output"], 
                        }, 
                        "response": response
                    }
                )
            
            # 过滤掉没有检索query的样本
            results_with_retrieval_query, filtered_reasoning_chains, filtered_queries = [], [], [] 
            for result, reasoning_chain, query in zip(results_require_retrieval, reasoning_chains, queries):
                if query is None:
                    continue 
                results_with_retrieval_query.append(result)
                filtered_reasoning_chains.append(reasoning_chain)
                filtered_queries.append(query)
            assert len(results_with_retrieval_query) == len(filtered_reasoning_chains) == len(filtered_queries) 

            if not results_with_retrieval_query:
                continue 
            # 检索documents并过滤得到相关的triples
            retrieved_documents = self._retriever.retrieve(queries=filtered_queries, topk=self._reasoning_model.topk)
            retrieved_triples = []
            for result, filtered_query, documents in zip(results_with_retrieval_query, filtered_queries, retrieved_documents):
                candidate_triples = self._retriever.get_candidate_triples_from_documents(documents)
                retrieved_triples.append(
                    self._retriever.filter_triples(
                        result["question"], 
                        filtered_query, 
                        candidate_triples,
                        self._reasoning_model.num_candidate_triples
                    )
                )

            # 识别相关的triples
            relevant_triples, triples_texts, responses = self.identify_relevant_triples(
                questions = [result["question"] for result in results_with_retrieval_query], 
                reasoning_chains = filtered_reasoning_chains, 
                queries = filtered_queries, 
                triples = retrieved_triples, 
                generation_config=generation_args 
            )

            # 记录中间状态
            for result, reasoning_chain, query, triples_text, response in zip(
                results_with_retrieval_query, filtered_reasoning_chains, filtered_queries, triples_texts, responses
            ):
                result["trajectory"].append(
                    {
                        "action": self.identify_relevant_triples_action, 
                        "state": {
                            "question": result["question"],
                            "reasoning_chain": reasoning_chain,
                            "query": query, 
                            "candidate_triples": triples_text, 
                        },
                        "response": response
                    }
                )
            
            # 得到relevant triples对应的documents
            relevant_documents = self._retriever.get_documents_from_triples(relevant_triples, retrieved_documents)
            for result, one_relevant_document_list in zip(results_with_retrieval_query, relevant_documents):
                for doc in one_relevant_document_list:
                    result["docids_to_scores"][doc["id"]] = max(result["docids_to_scores"].get(doc["id"], 0), doc["score"])

            # 更新context
            for result in results_with_retrieval_query:
                result["retrieval_count"] += 1 
                sorted_docids = [docid for docid, _ in sorted(result["docids_to_scores"].items(), key=lambda x: x[1], reverse=True)]
                sorted_docids = sorted_docids[:self._reasoning_model.max_num_context] 

                sorted_documents = [self._retriever.id2doc[docid] for docid in sorted_docids]
                context = [self.get_context(doc) for doc in sorted_documents]
                result["context"] = context

        return [result["trajectory"] for result in results], [result["output"] for result in results]
    """

    @torch.no_grad()
    def generate(self, questions: List[str], **kwargs):

        generation_args = {
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 1.0)
        }

        if self._reasoning_model.is_long_reasoning_model:
            prompts = self._reasoning_model.prompt(
                inputs=[ANSWER_PROMPT.format(question=question) for question in questions],
                use_chat=True, 
            )
        else:
            prompts = self._reasoning_model.prompt(
                inputs=[ANSWER_PROMPT_FOR_NON_REASONING_MODEL.format(question=question) for question in questions], 
                use_chat=True
            )

        results = []
        for question, prompt in zip(questions, prompts):
            results.append(
                {
                    "question": question, 
                    "prompt": prompt, 
                    "output": "", # 存储到目前reasoning step为止的所有输出
                    "existing_thought": "", # 存储在当前的reasoning step之前的thought
                    "current_thought": "", # 存储当前reasoning step生成的thought
                    "context": [], 
                    "finished": False, 
                    "trajectory": [],
                    "retrieval_trajectory": [], 
                    "docids_to_scores": {}, 
                    "step": 0, 
                    "request_times": 0, 
                    "retrieval_count": 0 
                }
            )

        while True:

            results_not_finished = []
            for result in results:
                if not result["finished"] and result["step"] < self._reasoning_model.max_reasoning_steps:
                    results_not_finished.append(result)
            
            if not results_not_finished:
                break 

            if results_not_finished[0]["step"] == 0:
                # retrieve at first step 
                questions = [result["question"] for result in results_not_finished]
                retrieved_documents: List[List[dict]] = self._retriever.retrieve(queries=questions, topk=self._reasoning_model.topk)
                for result, documents in zip(results_not_finished, retrieved_documents):
                    candidate_triples = self._retriever.get_candidate_triples_from_documents(documents)
                    retrieved_triples, retrieved_triples_scores = self._retriever.filter_triples(result["question"], "", candidate_triples, self._reasoning_model.num_candidate_triples)
                    for triple, score in zip(retrieved_triples, retrieved_triples_scores):
                        docid = triple["reference"][0]
                        docids_to_scores = result["docids_to_scores"]
                        docids_to_scores[docid] = max(docids_to_scores.get(docid, 0), score)
                    context = self.get_context_from_docids(docids_to_scores, result)
                    result["context"] = context 

            input_prompts = [] 
            for result in results_not_finished:
                input_prompt = result["prompt"].format(context="\n\n".join(result["context"]))
                input_prompts.append(input_prompt)
            generated_texts = self._reasoning_model.generate(
                prompts=input_prompts, 
                generation_config=generation_args,
                stop_words=["\n\n", self._reasoning_model.tokenizer.eos_token] if self._reasoning_model.is_long_reasoning_model \
                    else ["</think>", self._reasoning_model.tokenizer.eos_token] 
            )
            generated_texts = [escape_braces(text) for text in generated_texts]

            # 判断是否已经完成了thinking
            results_finish_thinking, texts_finish_thinking = [], []
            results_not_finish_thinking, texts_not_finish_thinking = [], [] 
            for result, generated_text in zip(results_not_finished, generated_texts):
                if self._reasoning_model.is_long_reasoning_model:
                    if "</think>" in generated_text:
                        results_finish_thinking.append(result)
                        texts_finish_thinking.append(generated_text)
                    else:
                        results_not_finish_thinking.append(result)
                        texts_not_finish_thinking.append(generated_text)
                else:
                    if "</answer>" in generated_text or "<answer>" in generated_text:
                        results_finish_thinking.append(result)
                        texts_finish_thinking.append(generated_text)
                    else:
                        results_not_finish_thinking.append(result)
                        texts_not_finish_thinking.append(generated_text)

            if results_finish_thinking:
                for result, text in zip(results_finish_thinking, texts_finish_thinking):
                    result["prompt"] += text 
                    result["output"] += text 
                input_prompts = [] 
                for result in results_finish_thinking:
                    input_prompt = result["prompt"].format(context="\n\n".join(result["context"]))
                    input_prompts.append(input_prompt)
                answer_texts = self._reasoning_model.generate(
                    prompts=input_prompts, 
                    generation_config=generation_args
                )
                for result, answer_text in zip(results_finish_thinking, answer_texts):
                    result["prompt"] += answer_text 
                    result["output"] += answer_text 
                    result["finished"] = True

            results_not_finished = results_not_finish_thinking
            generated_texts = texts_not_finish_thinking
            if not results_not_finished:
                break 

            for result, generated_text in zip(results_not_finished, generated_texts):
                result["existing_thought"] = deepcopy(result["output"])
                result["current_thought"] = deepcopy(generated_text)
                result["prompt"] += generated_text 
                result["output"] += generated_text 
                result["step"] += 1

            results_can_retrieve = []
            for result in results_not_finished:
                if result["retrieval_count"] < self._reasoning_model.max_retrieval_count and \
                    result["request_times"] < self._reasoning_model.max_request_times:
                    results_can_retrieve.append(result)
            
            if not results_can_retrieve:
                continue

            is_retrieval_required = self.is_retrieval_required_with_context(
                questions = [result["question"] for result in results_can_retrieve], 
                thoughts = [
                    result["existing_thought"] if self._reasoning_model.is_long_reasoning_model else \
                        self.convert_non_reasoning_model_thought_to_thought(result["existing_thought"])
                    for result in results_can_retrieve
                ], 
                current_thoughts = [
                    result["current_thought"] if self._reasoning_model.is_long_reasoning_model else \
                        self.convert_non_reasoning_model_thought_to_thought(result["current_thought"])
                    for result in results_can_retrieve
                ],
                contexts = [result["context"] for result in results_can_retrieve], 
                generation_config=generation_args
            )

            for result in results_can_retrieve:
                result["request_times"] += 1 

            # 记录中间状态
            results_require_retrieval = []
            for result, require_retrieval in zip(results_can_retrieve, is_retrieval_required):
                # NOTE: trajectory中的state需要和prompt对应：它的key应该是prompt中声明的变量，value则是直接插入到prompt中的字符串
                result["trajectory"].append(
                    {
                        "action": self.determine_retrieval_action, 
                        "state": {
                            "question": result["question"], 
                            "reasoning_history": result["existing_thought"] if self._reasoning_model.is_long_reasoning_model else\
                                self.convert_non_reasoning_model_thought_to_thought(result["existing_thought"]), 
                            "current_reasoning_step": result["current_thought"] if self._reasoning_model.is_long_reasoning_model else \
                                self.convert_non_reasoning_model_thought_to_thought(result["current_thought"]), 
                            "retrieved_context": "\n\n".join(result["context"]), 
                        },
                        "response": require_retrieval
                    }
                )
                if require_retrieval == self.enable_retrieval_symbol:
                    results_require_retrieval.append(result)
                # log retrieval trajectory 
                result["retrieval_trajectory"].append(
                    {
                        "action": self.determine_retrieval_action, 
                        "question": result["question"], 
                        "current_context_ids": self.get_context_ids(result["docids_to_scores"]), 
                        "require_retrieval": require_retrieval 
                    }
                )

            if not results_require_retrieval:
                continue

            # 构造检索的query
            reasoning_chains, queries, responses = self.formulate_retrieval_queries(
                questions = [result["question"] for result in results_require_retrieval], 
                thoughts = [result["output"] for result in results_require_retrieval], 
                generation_config=generation_args
            )

            # 记录中间状态
            for result, response, query in zip(results_require_retrieval, responses, queries):
                result["trajectory"].append(
                    {
                        "action": self.formulate_retrieval_queries_action, 
                        "state": {
                            "question": result["question"], 
                            "thought": result["output"], 
                        }, 
                        "response": response
                    }
                )
                result["retrieval_trajectory"].append(
                    {
                        "action": self.formulate_retrieval_queries_action, 
                        "question": result["question"], 
                        "query": query, 
                    }
                )
            
            # 过滤掉没有检索query的样本
            results_with_retrieval_query, filtered_reasoning_chains, filtered_queries = [], [], [] 
            for result, reasoning_chain, query in zip(results_require_retrieval, reasoning_chains, queries):
                if query is None:
                    continue 
                results_with_retrieval_query.append(result)
                filtered_reasoning_chains.append(reasoning_chain)
                filtered_queries.append(query)
            assert len(results_with_retrieval_query) == len(filtered_reasoning_chains) == len(filtered_queries) 

            if not results_with_retrieval_query:
                continue 

            # 检索documents并过滤得到相关的triples
            retrieved_documents = self._retriever.retrieve(queries=filtered_queries, topk=self._reasoning_model.topk)
            retrieved_triples, retrieved_triples_scores = [], []
            for result, filtered_query, documents in zip(results_with_retrieval_query, filtered_queries, retrieved_documents):
                candidate_triples = self._retriever.get_candidate_triples_from_documents(documents)
                retrieved_triples_results = self._retriever.filter_triples(
                    result["question"], 
                    filtered_query, 
                    candidate_triples,
                    self._reasoning_model.num_candidate_triples
                )
                retrieved_triples.append(retrieved_triples_results[0])
                retrieved_triples_scores.append(retrieved_triples_results[1])

            # 识别相关的triples
            relevant_triples, triples_texts, responses = self.identify_relevant_triples(
                questions = [result["question"] for result in results_with_retrieval_query], 
                reasoning_chains = filtered_reasoning_chains, 
                queries = filtered_queries, 
                triples = retrieved_triples, 
                generation_config=generation_args 
            )

            # 记录中间状态
            for result, reasoning_chain, query, triples, triples_text, response in zip(
                results_with_retrieval_query, filtered_reasoning_chains, filtered_queries, relevant_triples, triples_texts, responses
            ):
                result["trajectory"].append(
                    {
                        "action": self.identify_relevant_triples_action, 
                        "state": {
                            "question": result["question"],
                            "reasoning_chain": reasoning_chain,
                            "query": query, 
                            "candidate_triples": triples_text, 
                        },
                        "response": response
                    }
                )
                result["retrieval_trajectory"].append(
                    {
                        "action": self.identify_relevant_triples_action, 
                        "question": result["question"], 
                        "selected_context_id": triples[0]["reference"][0],
                    }
                )
            
            # 得到relevant triples对应的documents
            for result, one_retrieved_triples, one_retrieved_triples_scores, one_relevant_triples in \
                zip(results_require_retrieval, retrieved_triples, retrieved_triples_scores, relevant_triples):
                triple_text_to_index = {triple["text"]: index for index, triple in enumerate(one_retrieved_triples)}
                for relevant_triple in one_relevant_triples:
                    idx = triple_text_to_index[relevant_triple["text"]]
                    one_retrieved_triples_scores[idx] += 0.5 
                for triple, score in zip(one_retrieved_triples, one_retrieved_triples_scores):
                    docid = triple["reference"][0]
                    result["docids_to_scores"][docid] = max(result["docids_to_scores"].get(docid, 0), score)
            
            # 更新context
            for result in results_with_retrieval_query:
                result["retrieval_count"] += 1 
                # sorted_docids = [docid for docid, _ in sorted(result["docids_to_scores"].items(), key=lambda x: x[1], reverse=True)]
                # sorted_docids = sorted_docids[:self._reasoning_model.max_num_context] 
                # sorted_documents = [self._retriever.id2doc[docid] for docid in sorted_docids]
                # context = [self.get_context(doc) for doc in sorted_documents]
                # result["context"] = context
                result["context"] = self.get_context_from_docids(docids_to_scores=result["docids_to_scores"], result=result)

        for result in results:
            result["retrieval_trajectory"].append(
                {
                    "action": "final_context", 
                    "question": result["question"], 
                    "final_context_ids": self.get_context_ids(result["docids_to_scores"]), 
                }
            )
        """
        results_not_finished = [] 
        for result in results:
            if "</think>" not in result["output"] or r"\boxed{" not in result["output"]:
                results_not_finished.append(result)
        
        if results_not_finished:
            prompts = self._reasoning_model.prompt(
                inputs=[ANSWER_PROMPT.format(question=result["question"]) for result in results_not_finished],
                use_chat=True, 
            )
            input_prompts = [] 
            for prompt, result in zip(prompts, results_not_finished):
                input_prompt = prompt.format(context="\n\n".join(result["context"]))
                input_prompts.append(input_prompt)
            generated_texts = self._reasoning_model.generate(
                prompts=input_prompts, 
                generation_config=generation_args
            )
            for result, prompt, generated_text in zip(results_not_finished, prompts, generated_texts):
                result["prompt"] = prompt 
                result["output"] = generated_text 
                result["finished"] = True 
        """
        
        return [result["trajectory"] for result in results], [result["output"] for result in results], [result["retrieval_trajectory"] for result in results] 
    

    @torch.no_grad()
    def tree_rollout(self, question: str, reward_model, label: str, max_tree_depth: int = 4, **kwargs):
        """
        Args:
            question: 单个问题
            reward_model: 奖励模型, 用于评估trajectory
            label: 正确答案标签
            max_tree_depth: 最大树深度，在此深度内进行分支扩展
            **kwargs: 生成参数
        """
        generation_args = {
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 1.0)
        }

        # 初始化根节点
        if self._reasoning_model.is_long_reasoning_model:
            prompt = self._reasoning_model.prompt(
                inputs=[ANSWER_PROMPT.format(question=question)],
                use_chat=True, 
            )[0]
        else:
            prompt = self._reasoning_model.prompt(
                inputs=[ANSWER_PROMPT_FOR_NON_REASONING_MODEL.format(question=question)], 
                use_chat=True
            )[0]

        initial_result = {
            "question": question, 
            "prompt": prompt, 
            "output": "",
            "existing_thought": "",
            "current_thought": "",
            "context": [], 
            "finished": False, 
            "trajectory": [],
            "retrieval_trajectory": [], 
            "docids_to_scores": {}, 
            "step": 0, 
            "request_times": 0, 
            "retrieval_count": 0 
        }

        root_node = TreeNode(initial_result, branch_type="root")
        
        # 第一步：初始检索
        retrieved_documents = self._retriever.retrieve(queries=[question], topk=self._reasoning_model.topk)[0]
        candidate_triples = self._retriever.get_candidate_triples_from_documents(retrieved_documents)
        retrieved_triples, retrieved_triples_scores = self._retriever.filter_triples(
            question, "", candidate_triples, self._reasoning_model.num_candidate_triples
        )
        
        for triple, score in zip(retrieved_triples, retrieved_triples_scores):
            docid = triple["reference"][0]
            docids_to_scores = root_node.result["docids_to_scores"]
            docids_to_scores[docid] = max(docids_to_scores.get(docid, 0), score)
        
        context = self.get_context_from_docids(docids_to_scores, root_node.result)
        root_node.result["context"] = context

        current_depth = 0
        
        while True:

            # 获取所有未完成的叶子节点
            leaf_nodes = [
                node for node in root_node.get_all_leaf_nodes() \
                    if not node.is_finished and \
                        node.result["step"] < self._reasoning_model.max_reasoning_steps
            ]
            
            if not leaf_nodes:
                break
            
            # 生成reasoning step
            # from pdb import set_trace; set_trace()
            leaf_nodes = self._generate_reasoning_step_for_nodes(leaf_nodes, generation_args)
            
            # 剪枝操作：移除相似的检索分支
            if current_depth < max_tree_depth:
                leaf_nodes = self._prune_similar_branches(leaf_nodes)
            
            # 过滤掉已完成thinking的节点
            active_nodes = [node for node in leaf_nodes if not node.is_finished]
            if not active_nodes:
                break
                
            # 更新节点状态
            for node in active_nodes:
                node.result["step"] += 1
            
            # 检查是否需要进行树状扩展
            if current_depth < max_tree_depth:
                # 树状扩展：对每个决策点生成分支
                self._expand_tree_nodes(active_nodes, generation_args, current_depth)
            else:
                # 单一rollout：正常处理
                self._process_single_rollout(active_nodes, generation_args)
            
            current_depth += 1
        
        # 收集所有轨迹
        all_trajectories = []
        all_outputs = []
        all_retrieval_trajectories = []
        
        # 收集所有叶子节点及其最终得分
        leaf_nodes = root_node.get_all_leaf_nodes()
        leaf_scores = {}  # node_id -> score

        for i, leaf_node in enumerate(leaf_nodes):

            if leaf_node.result["trajectory"]:  # 只收集有轨迹的叶子节点
                all_trajectories.append(leaf_node.result["trajectory"])
                all_outputs.append(leaf_node.result["output"])
                all_retrieval_trajectories.append(leaf_node.result["retrieval_trajectory"])
                
                # 使用reward_model对最终输出进行评分
                score = reward_model.forward_tree_rollout(
                    prediction = leaf_node.result["output"], 
                    label = label, 
                    question=leaf_node.result["question"], 
                    ctxids=self.get_context_ids(leaf_node.result["docids_to_scores"])
                )
                leaf_scores[id(leaf_node)] = score
        
        # 计算每个操作的评分（包含剪枝处理）
        # from pdb import set_trace; set_trace()
        all_operations = self._calculate_operation_scores_with_pruning(root_node, leaf_scores)

        # 对action进行采样
        sampled_operations = self._sample_operations(all_operations)
        
        # 将operations转换成训练所需的格式
        operations_sequences, operations_attention_mask, operations_action_mask, operations_rewards = self._collate_operations(sampled_operations)
        
        return operations_sequences, operations_attention_mask, operations_action_mask, operations_rewards

    def _sample_operations(self, all_operations: List[dict]) -> List[dict]:
        """
        对operations进行采样
        - 保留所有的formulate_query和select_triple的action
        - 随机采样和formulate_query+select_triple相同数量的determine_retrieval action
        """
        import random
        
        # 按action类型分组
        determine_retrieval_ops = []
        formulate_query_ops = []
        select_triple_ops = []
        
        for op in all_operations:
            action_type = op["action"]
            if action_type == self.determine_retrieval_action:
                determine_retrieval_ops.append(op)
            elif action_type == self.formulate_retrieval_queries_action:
                formulate_query_ops.append(op)
            elif action_type == self.identify_relevant_triples_action:
                select_triple_ops.append(op)
        
        # 保留所有的formulate_query和select_triple
        sampled_operations = formulate_query_ops + select_triple_ops
        
        # 计算需要采样的determine_retrieval数量
        target_determine_count = len(formulate_query_ops) + len(select_triple_ops)
        
        if target_determine_count > 0:
            # 只有在target_determine_count大于0的时候才随机采样determine_retrieval操作
            if len(determine_retrieval_ops) <= target_determine_count:
                # 如果determine_retrieval数量不够，全部保留
                sampled_determine_ops = determine_retrieval_ops
            else:
                # 随机采样指定数量
                sampled_determine_ops = random.sample(determine_retrieval_ops, target_determine_count)
        else:
            # 如果没有formulate_query和select_triple操作，则不采样determine_retrieval 
            sampled_determine_ops = determine_retrieval_ops 
        
        # 合并所有采样的操作
        sampled_operations.extend(sampled_determine_ops)
        
        return sampled_operations

    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的Jaccard相似度"""
        # 提取最后一行reasoning结果
        lines1 = text1.strip().split('\n')
        lines2 = text2.strip().split('\n')
        
        last_line1 = lines1[-1].strip() if lines1 else ""
        last_line2 = lines2[-1].strip() if lines2 else ""
        
        # 转换为词集合
        words1 = set(last_line1.split())
        words2 = set(last_line2.split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _prune_similar_branches(self, leaf_nodes: List[TreeNode]) -> List[TreeNode]:
        """
        剪枝相似的检索分支
        根据树结构：
        - Retrieval branch: determine_retrieval → query_formulation → triple_selector (叶子节点)
        - Non-retrieval branch: determine_retrieval (叶子节点)
        """
        if len(leaf_nodes) < 2:
            return leaf_nodes
        
        # 找到所有可能的剪枝对：retrieval chain的最后节点 vs non-retrieval节点
        nodes_to_remove = set()
        pruned_branches = set()  # 记录已处理的分支
                
        for i, node1 in enumerate(leaf_nodes):
            if id(node1) in pruned_branches:
                continue
                
            for j, node2 in enumerate(leaf_nodes):
                if i >= j or id(node2) in pruned_branches:
                    continue
                
                # 检查是否是来自同一个根分支的两个不同路径
                if not self._are_sibling_branches(node1, node2):
                    continue
                
                # 识别retrieval和non-retrieval节点
                retrieval_leaf, non_retrieval_leaf = self._identify_branch_types(node1, node2)
                
                if retrieval_leaf is None or non_retrieval_leaf is None:
                    continue
                
                # 获取reasoning结果
                output1 = retrieval_leaf.result.get("output", "")
                output2 = non_retrieval_leaf.result.get("output", "")
                
                # 计算Jaccard相似度
                similarity = self._calculate_jaccard_similarity(output1, output2)
                
                if similarity > 0.9:
                    # 找到要剪枝的retrieval分支的根节点
                    retrieval_root = self._find_corresponding_retrieval_node(retrieval_leaf)
                    
                    # 标记为被剪枝
                    retrieval_root.is_pruned = True
                    retrieval_root.pruned_similarity = similarity
                    retrieval_root.corresponding_non_retrieval_node = non_retrieval_leaf
                    
                    # 移除整个retrieval分支的所有节点
                    retrieval_nodes = self._get_all_retrieval_branch_nodes(retrieval_leaf)
                    for node in retrieval_nodes:
                        nodes_to_remove.add(id(node))
                    
                    # 标记已处理
                    pruned_branches.add(id(retrieval_leaf))
                    pruned_branches.add(id(non_retrieval_leaf))
                    
                    # print(f"Pruned retrieval branch with similarity {similarity:.3f}")
                    break
        
        # 返回未被剪枝的节点
        return [node for node in leaf_nodes if id(node) not in nodes_to_remove]

    def _are_sibling_branches(self, node1: TreeNode, node2: TreeNode) -> bool:
        """
        检查两个节点是否来自同一个分支决策点
        node1/node2只能是triple_branch或者retrieval_no类型
        """
        # 获取两个节点对应的retrieval节点
        retrieval_node1 = self._find_corresponding_retrieval_node(node1) 
        retrieval_node2 = self._find_corresponding_retrieval_node(node2) 
        if id(retrieval_node1) == id(retrieval_node2):
            # 说明是同个retrieval节点下的分支
            return False 
        
        # 获取两个节点的上一个retrieval节点
        prev_retrieval1 = self._find_previous_retrieval_or_root_node_from(retrieval_node1)
        prev_retrieval2 = self._find_previous_retrieval_or_root_node_from(retrieval_node2)
        
        # 如果有相同的上一个retrieval节点，说明是相邻的兄弟分支
        return prev_retrieval1 is not None and prev_retrieval2 is not None and id(prev_retrieval1) == id(prev_retrieval2)

    def _find_corresponding_retrieval_node(self, node: TreeNode) -> TreeNode: 
        """找到triple_branch节点对应的retrieval节点"""
        current = node 
        while "retrieval_" not in current.branch_type:
            current = current.parent  
        return current 

    def _find_previous_retrieval_or_root_node_from(self, start_node: TreeNode) -> TreeNode:
        """从指定节点开始往上找上一个包含determine_retrieval的节点"""
        current = start_node.parent  # 从父节点开始查找
        while "retrieval_" not in current.branch_type and "root" not in current.branch_type:
            current = current.parent
        return current

    def _identify_branch_types(self, node1: TreeNode, node2: TreeNode) -> tuple:
        """识别哪个是retrieval分支, 哪个是non-retrieval分支"""
        retrieval_node = None
        non_retrieval_node = None
        
        for node in [node1, node2]:
            corresponding_retrieval_node = self._find_corresponding_retrieval_node(node) 
            if "retrieval_yes" in corresponding_retrieval_node.branch_type:
                retrieval_node = node
            else:
                non_retrieval_node = node 
        
        return retrieval_node, non_retrieval_node

    def _get_all_retrieval_branch_nodes(self, retrieval_leaf: TreeNode) -> List[TreeNode]:
        """获取retrieval分支的所有节点"""
        nodes = [] 
        current = retrieval_leaf
        while current is not None:
            if "retrieval_yes" in current.branch_type:
                nodes.append(current)
                break 
            nodes.append(current)
            current = current.parent
        return nodes

    def _calculate_operation_scores(self, root_node: TreeNode, leaf_scores: dict) -> List[dict]:
        """
        计算每个操作的评分，基于从该操作延伸出去的所有分支的平均得分
        
        Args:
            root_node: 根节点
            leaf_scores: 叶子节点得分字典 {node_id: score}
            
        Returns:
            List[dict]: 每个操作的评分信息
        """
        all_operations = []
        
        # 递归遍历所有节点，收集操作信息
        def traverse_node(node: TreeNode, parent_trajectory_len: int = 0):
            
            # 只处理当前节点新增的操作（避免重复计算父节点的操作）
            current_trajectory = node.result.get("trajectory", [])
            new_operations = current_trajectory[parent_trajectory_len:]
            
            # 判断是否为叶子节点
            is_leaf = len(node.children) == 0
            
            if is_leaf:
                # 对于叶子节点，每个新增的操作都使用该叶子节点自己的得分
                node_score = leaf_scores.get(id(node), 0.0)
                
                for operation_idx, action_item in enumerate(new_operations):
                    action_type = action_item["action"]
                    
                    # 只记录三种主要操作
                    if action_type in [self.determine_retrieval_action, self.formulate_retrieval_queries_action, self.identify_relevant_triples_action]:
                        operation_info = {
                            "action": action_type,
                            "state": action_item["state"],
                            "response": action_item["response"],
                            "score": node_score,  # 叶子节点操作使用自己的得分
                            "num_branches": 1,  # 叶子节点只有一个分支
                            "branch_scores": [node_score],
                            "node_id": id(node),
                            "operation_index": parent_trajectory_len + operation_idx,
                            "branch_type": node.branch_type,
                            "trajectory_length": len(current_trajectory),
                            "is_leaf_operation": True  # 标记为叶子节点操作
                        }
                        all_operations.append(operation_info)
            else:
                # 对于非叶子节点，使用所有后代节点的平均得分
                descendant_leaves = node.get_all_leaf_nodes()
                descendant_scores = [leaf_scores.get(id(leaf), 0.0) for leaf in descendant_leaves if id(leaf) in leaf_scores] 

                avg_score = sum(descendant_scores) / len(descendant_scores)
                
                for operation_idx, action_item in enumerate(new_operations):
                    action_type = action_item["action"]
                    
                    # 只记录三种主要操作
                    if action_type in [self.determine_retrieval_action, self.formulate_retrieval_queries_action, self.identify_relevant_triples_action]:
                        operation_info = {
                            "action": action_type,
                            "state": action_item["state"],
                            "response": action_item["response"],
                            "score": avg_score,
                            "num_branches": len(descendant_scores),
                            "branch_scores": descendant_scores,
                            "node_id": id(node),
                            "operation_index": parent_trajectory_len + operation_idx,
                            "branch_type": node.branch_type,
                            "trajectory_length": len(current_trajectory),
                            "is_leaf_operation": False
                        }
                        all_operations.append(operation_info)
            
            # 递归处理子节点，传递当前节点的trajectory长度
            for child in node.children:
                traverse_node(child, len(current_trajectory))
        
        # 从根节点开始遍历
        traverse_node(root_node)
        
        return all_operations

    def _calculate_operation_scores_with_pruning(self, root_node: TreeNode, leaf_scores: dict) -> List[dict]:
        
        final_retrieve_penalty = 0.05 
        pruned_penalty = 0.1 

        all_operations = []
        final_determine_retrieval_operations = [] 
        
        # 递归遍历所有节点，收集操作信息
        def traverse_node(node: TreeNode, parent_trajectory_len: int = 0):
            
            # 只处理当前节点新增的操作（避免重复计算父节点的操作）
            current_trajectory = node.result.get("trajectory", [])
            new_operations = current_trajectory[parent_trajectory_len:]
            
            # 判断是否为叶子节点
            is_leaf = len(node.children) == 0
            
            if is_leaf:
                # 对于叶子节点，每个新增的操作都使用该叶子节点自己的得分
                node_score = leaf_scores.get(id(node), 0.0)
                
                for operation_idx, action_item in enumerate(new_operations):
                    action_type = action_item["action"]
                    
                    # 只记录三种主要操作
                    if action_type in [self.determine_retrieval_action, self.formulate_retrieval_queries_action, self.identify_relevant_triples_action]:
                        operation_info = {
                            "action": action_type,
                            "state": action_item["state"],
                            "response": action_item["response"],
                            "score": node_score,  # 叶子节点操作使用自己的得分
                            "num_branches": 1,  # 叶子节点只有一个分支
                            "branch_scores": [node_score],
                            "node_id": id(node),
                            "operation_index": parent_trajectory_len + operation_idx,
                            "branch_type": node.branch_type,
                            "trajectory_length": len(current_trajectory),
                            "is_leaf_operation": True  # 标记为叶子节点操作
                        }
                        all_operations.append(operation_info)
                
                # 在叶子节点的trajectory中得到最后一个determine retrieval的惩罚项分数
                final_determine_retrieval_operations.append(self._get_final_det_operation(node.result["trajectory"], penalty_score=final_retrieve_penalty))
            
            else:
                # 对于非叶子节点，使用所有后代节点的平均得分
                descendant_leaves = node.get_all_leaf_nodes()
                descendant_scores = [leaf_scores.get(id(leaf), 0.0) for leaf in descendant_leaves if id(leaf) in leaf_scores] 

                avg_score = sum(descendant_scores) / len(descendant_scores)
                
                for operation_idx, action_item in enumerate(new_operations):
                    action_type = action_item["action"]
                    
                    # 只记录三种主要操作
                    if action_type in [self.determine_retrieval_action, self.formulate_retrieval_queries_action, self.identify_relevant_triples_action]:
                        operation_info = {
                            "action": action_type,
                            "state": action_item["state"],
                            "response": action_item["response"],
                            "score": avg_score,
                            "num_branches": len(descendant_scores),
                            "branch_scores": descendant_scores,
                            "node_id": id(node),
                            "operation_index": parent_trajectory_len + operation_idx,
                            "branch_type": node.branch_type,
                            "trajectory_length": len(current_trajectory),
                            "is_leaf_operation": False
                        }
                        all_operations.append(operation_info)
            
            # 递归处理子节点，传递当前节点的trajectory长度
            has_pruned_child = any(child.is_pruned for child in node.children)
            current_operation_length = len(all_operations)
            for child in node.children:

                if child.is_pruned:
                    # 如果被剪枝了，那么记录被剪掉的determine retrieval的info，同时给它一个惩罚项
                    counterpart = child.corresponding_non_retrieval_node
                    counterpart_is_leaf = len(counterpart.children) == 0
                    if counterpart_is_leaf:
                        counterpart_score = leaf_scores.get(id(counterpart), 0.0)
                    else:
                        counterpart_descendant_leaves = counterpart.get_all_leaf_nodes()
                        counterpart_descendant_scores = [leaf_scores.get(id(leaf), 0.0) for leaf in counterpart_descendant_leaves if id(leaf) in leaf_scores] 
                        counterpart_score = sum(counterpart_descendant_scores) / min(len(counterpart_descendant_scores), 1)
                    
                    child_trajectory = child.result.get("trajectory", [])
                    child_new_operations = child_trajectory[len(current_trajectory):]
                    for action_item in child_new_operations:
                        action_type = action_item["action"] 
                        if action_type == self.determine_retrieval_action:
                            all_operations.append(
                                {
                                    "action": action_type, 
                                    "state": action_item["state"], 
                                    "response": action_item["response"], 
                                    "score": counterpart_score - pruned_penalty,
                                    "num_branches": 0, 
                                    "branch_scores": None, 
                                    "node_id": id(child), 
                                    "operation_index": None, 
                                    "branch_type": child.branch_type, 
                                    "trajectory_length": len(current_trajectory) + 1, 
                                    "is_leaf_operation": False 
                                }
                            )
                            current_operation_length += 1 
                            break
                else:
                    traverse_node(child, len(current_trajectory))
                    # 对于非剪枝的节点，对determine_retrieval增加额外的奖励
                    if has_pruned_child:
                        all_operations[current_operation_length]["score"] += pruned_penalty 
        
        # 从根节点开始遍历
        traverse_node(root_node)

        # 把final_determine_retrieval_operations的penalty_score加到对应的operation上
        for det_operation in final_determine_retrieval_operations:
            for operation in all_operations:
                if (det_operation["action"] == operation["action"]) and (det_operation["response"] == operation["response"]):
                    det_state = det_operation["state"]
                    operation_state = operation["state"]
                    if (det_state["question"] == operation_state["question"]) and \
                        (det_state["reasoning_history"] == operation_state["reasoning_history"]) and \
                            (det_state["current_reasoning_step"] == operation_state["current_reasoning_step"]) and \
                                (det_state["retrieved_context"] == operation_state["retrieved_context"]):
                        operation["score"] += det_operation["penalty_score"]
                        break 
        
        return all_operations

    def _get_final_det_operation(self, action_list: List[dict], penalty_score: int) -> dict:

        last_determine_retrieval_action= None
        for i in range(len(action_list)-1, -1, -1):
            if action_list[i]["action"] == self.determine_retrieval_action:
                last_determine_retrieval_action = action_list[i]
                break 
        
        last_determine_retrieval_action_info = {
            "action": last_determine_retrieval_action["action"], 
            "state": last_determine_retrieval_action["state"], 
            "response": last_determine_retrieval_action["response"], 
            "penalty_score": -penalty_score if "yes" in last_determine_retrieval_action["response"].lower() else penalty_score
        }

        return last_determine_retrieval_action_info

    def _collate_operations(self, operations: List[dict]):
        """
        将operations转换成类似actor.py中的sequences, attention_mask, action_mask和reward的形式
        
        Args:
            operations: 操作列表
            
        Returns:
            tuple: (sequences, attention_mask, action_mask, rewards)
        """

        # 提取prompts, responses和rewards
        prompts = []
        responses = []
        rewards = []
        
        for op in operations:
            # 构建prompt（基于state）
            prompt = self.get_action_prompt(op["action"], op["state"])
            prompts.append(prompt)
            responses.append(op["response"])
            rewards.append(op["score"])
        
        batch_size = len(operations)
        
        # 使用tokenizer处理prompts和responses
        # right padding for inputs
        inputs = self.tokenizer(prompts, max_length=self.prompt_max_len, padding=True, truncation=True, return_tensors="pt", padding_side="left") 
        input_ids, input_attention_mask = inputs["input_ids"], inputs["attention_mask"]

        # right padding for outputs
        outputs = self.tokenizer(responses, max_length=self.generate_max_len, padding=True, truncation=True, return_tensors="pt", padding_side="right", add_special_tokens=False)
        output_ids, output_attention_mask = add_eos_token_at_the_end(outputs["input_ids"], outputs["attention_mask"], self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
        
        # 先在dim=1维度拼接input_ids和output_ids
        sequences = torch.cat([input_ids, output_ids], dim=1)
        input_lens = input_attention_mask.size(1)  # 每个样本的input长度
        
        # 完全按照actor.py中process_sequences的逻辑处理
        # 直接调用process_sequences方法（与actor.py中generate方法的调用方式相同）
        sequences, attention_mask, action_mask = self.process_sequences(sequences, input_lens, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
        
        # 转换rewards为tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        return sequences, attention_mask, action_mask, rewards_tensor

    def _generate_reasoning_step_for_nodes(self, nodes: List[TreeNode], generation_args: dict) -> List:

        batch_size = 4
        updated_nodes = []
        
        for i in range(0, len(nodes), batch_size):

            batch_nodes = nodes[i:i+batch_size]
            
            # 准备输入
            input_prompts = []
            for node in batch_nodes:
                input_prompt = node.result["prompt"].format(context="\n\n".join(node.result["context"]))
                input_prompts.append(input_prompt)
            
            # 生成文本
            generated_texts = self._reasoning_model.generate(
                prompts=input_prompts, 
                generation_config=generation_args,
                stop_words=["\n\n", self._reasoning_model.tokenizer.eos_token] if self._reasoning_model.is_long_reasoning_model 
                    else ["</think>", self._reasoning_model.tokenizer.eos_token] 
            )
            generated_texts = [escape_braces(text) for text in generated_texts]
            
            # 处理生成结果
            for node, generated_text in zip(batch_nodes, generated_texts):
                # 检查是否完成thinking
                is_finished = False
                if self._reasoning_model.is_long_reasoning_model:
                    if "</think>" in generated_text:
                        is_finished = True
                else:
                    if "</answer>" in generated_text or "<answer>" in generated_text:
                        is_finished = True
                
                if is_finished:
                    # 完成thinking，生成最终答案
                    node.result["prompt"] += generated_text
                    node.result["output"] += generated_text
                    
                    input_prompt = node.result["prompt"].format(context="\n\n".join(node.result["context"]))
                    answer_text = self._reasoning_model.generate(
                        prompts=[input_prompt], 
                        generation_config=generation_args
                    )[0]
                    
                    node.result["prompt"] += answer_text
                    node.result["output"] += answer_text
                    node.is_finished = True
                else:
                    # 更新节点状态
                    node.result["existing_thought"] = deepcopy(node.result["output"])
                    node.result["current_thought"] = deepcopy(generated_text)
                    node.result["prompt"] += generated_text
                    node.result["output"] += generated_text
                
                updated_nodes.append(node)
        
        return updated_nodes

    def _expand_tree_nodes(self, nodes: List, generation_args: dict, current_depth: int):
        """扩展树节点，为每个决策点创建分支"""
        for node in nodes:
            if (node.result["retrieval_count"] >= self._reasoning_model.max_retrieval_count or 
                node.result["request_times"] >= self._reasoning_model.max_request_times):
                continue
            
            # 创建检索决策分支：Yes和No
            node.result["request_times"] += 1
            
            # 分支1：需要检索
            yes_node = TreeNode(node.result, parent=node, branch_type="retrieval_yes")
            yes_node.result["trajectory"].append({
                "action": self.determine_retrieval_action,
                "state": {
                    "question": node.result["question"],
                    "reasoning_history": node.result["existing_thought"] if self._reasoning_model.is_long_reasoning_model else
                        self.convert_non_reasoning_model_thought_to_thought(node.result["existing_thought"]),
                    "current_reasoning_step": node.result["current_thought"] if self._reasoning_model.is_long_reasoning_model else
                        self.convert_non_reasoning_model_thought_to_thought(node.result["current_thought"]),
                    "retrieved_context": "\n\n".join(node.result["context"]),
                },
                "response": self.enable_retrieval_symbol
            })
            yes_node.result["retrieval_trajectory"].append({
                "action": self.determine_retrieval_action,
                "question": node.result["question"],
                "current_context_ids": self.get_context_ids(node.result["docids_to_scores"]),
                "require_retrieval": self.enable_retrieval_symbol
            })
            node.add_child(yes_node)
            
            # 分支2：不需要检索
            no_node = TreeNode(node.result, parent=node, branch_type="retrieval_no")
            no_node.result["trajectory"].append({
                "action": self.determine_retrieval_action,
                "state": {
                    "question": node.result["question"],
                    "reasoning_history": node.result["existing_thought"] if self._reasoning_model.is_long_reasoning_model else
                        self.convert_non_reasoning_model_thought_to_thought(node.result["existing_thought"]),
                    "current_reasoning_step": node.result["current_thought"] if self._reasoning_model.is_long_reasoning_model else
                        self.convert_non_reasoning_model_thought_to_thought(node.result["current_thought"]),
                    "retrieved_context": "\n\n".join(node.result["context"]),
                },
                "response": self.disable_retrieval_symbol
            })
            no_node.result["retrieval_trajectory"].append({
                "action": self.determine_retrieval_action,
                "question": node.result["question"],
                "current_context_ids": self.get_context_ids(node.result["docids_to_scores"]),
                "require_retrieval": self.disable_retrieval_symbol
            })
            node.add_child(no_node)
            
            # 只对需要检索的分支继续扩展
            self._expand_retrieval_branch(yes_node, generation_args)

    def _expand_retrieval_branch(self, retrieval_node, generation_args: dict):
        """扩展检索分支, 生成query和triple选择的分支"""
        # 生成两个不同的query分支
        query_branches = self._generate_query_branches(retrieval_node, generation_args, num_branches=1) # 只生成一个query分支
        
        for i, (reasoning_chain, query, response) in enumerate(query_branches):
            if query is None:
                continue
                
            query_node = TreeNode(retrieval_node.result, parent=retrieval_node, branch_type=f"query_branch_{i}")
            query_node.result["trajectory"].append({
                "action": self.formulate_retrieval_queries_action,
                "state": {
                    "question": retrieval_node.result["question"],
                    "thought": retrieval_node.result["output"],
                },
                "response": response
            })
            query_node.result["retrieval_trajectory"].append({
                "action": self.formulate_retrieval_queries_action,
                "question": retrieval_node.result["question"],
                "query": query,
            })
            retrieval_node.add_child(query_node)
            
            # 为每个query分支生成triple选择分支
            self._expand_triple_branches(query_node, reasoning_chain, query, generation_args)

    def _generate_query_branches(self, node, generation_args: dict, num_branches: int = 2):
        """生成多个query分支"""
        # 使用不同的采样参数生成多样化的query
        branches = []
        
        for i in range(num_branches):
            # 调整采样参数以增加多样性
            modified_args = deepcopy(generation_args)
            modified_args["temperature"] = generation_args.get("temperature", 1.0) * (1.0 + i * 0.2)
            modified_args["do_sample"] = True
            
            reasoning_chains, queries, responses = self.formulate_retrieval_queries(
                questions=[node.result["question"]],
                thoughts=[node.result["output"]],
                generation_config=modified_args
            )
            
            branches.append((reasoning_chains[0], queries[0], responses[0]))
        
        return branches

    def _expand_triple_branches(self, query_node, reasoning_chain: str, query: str, generation_args: dict):
        """扩展triple选择分支"""
        # 检索documents和triples
        retrieved_documents = self._retriever.retrieve(queries=[query], topk=self._reasoning_model.topk)[0]
        candidate_triples = self._retriever.get_candidate_triples_from_documents(retrieved_documents)
        retrieved_triples, retrieved_triples_scores = self._retriever.filter_triples(
            query_node.result["question"], query, candidate_triples, self._reasoning_model.num_candidate_triples
        )
        
        # 生成两个不同的triple选择分支
        triple_branches = self._generate_triple_branches(
            query_node, reasoning_chain, query, [retrieved_triples], generation_args, num_branches=1 
        )
        
        for i, (relevant_triples, response) in enumerate(triple_branches):
            triple_node = TreeNode(query_node.result, parent=query_node, branch_type=f"triple_branch_{i}")
            
            # 构建triples_text用于trajectory
            triples_text = []
            for triple in retrieved_triples:
                triples_text.append(f"Source Title: {triple['title']}\nText: {triple['text']}")
            triples_text_str = "\n".join([f"{j+1}. {text}" for j, text in enumerate(triples_text)])
            
            triple_node.result["trajectory"].append({
                "action": self.identify_relevant_triples_action,
                "state": {
                    "question": query_node.result["question"],
                    "reasoning_chain": reasoning_chain,
                    "query": query,
                    "candidate_triples": triples_text_str,
                },
                "response": response
            })
            
            if relevant_triples:
                triple_node.result["retrieval_trajectory"].append({
                    "action": self.identify_relevant_triples_action,
                    "question": query_node.result["question"],
                    "selected_context_id": relevant_triples[0]["reference"][0],
                })
            
            query_node.add_child(triple_node)
            
            # 更新context
            self._update_context_for_node(triple_node, retrieved_triples, retrieved_triples_scores, relevant_triples)

    def _generate_triple_branches(self, node, reasoning_chain: str, query: str, triples: List[List[dict]], 
                                 generation_args: dict, num_branches: int = 2):
        """生成多个triple选择分支"""
        branches = []
        
        for i in range(num_branches):
            # 调整采样参数
            modified_args = deepcopy(generation_args)
            modified_args["temperature"] = generation_args.get("temperature", 1.0) * (1.0 + i * 0.2)
            modified_args["do_sample"] = True
            
            relevant_triples, triples_texts, responses = self.identify_relevant_triples(
                questions=[node.result["question"]],
                reasoning_chains=[reasoning_chain],
                queries=[query],
                triples=triples,
                generation_config=modified_args
            )
            
            branches.append((relevant_triples[0], responses[0]))
        
        return branches

    def _update_context_for_node(self, node, retrieved_triples: List[dict], 
                                retrieved_triples_scores: List[float], relevant_triples: List[dict]):
        """为节点更新context"""
        node.result["retrieval_count"] += 1
        
        # 更新docids_to_scores
        triple_text_to_index = {triple["text"]: index for index, triple in enumerate(retrieved_triples)}
        for relevant_triple in relevant_triples:
            if relevant_triple["text"] in triple_text_to_index:
                idx = triple_text_to_index[relevant_triple["text"]]
                retrieved_triples_scores[idx] += 0.5
        
        for triple, score in zip(retrieved_triples, retrieved_triples_scores):
            docid = triple["reference"][0]
            node.result["docids_to_scores"][docid] = max(node.result["docids_to_scores"].get(docid, 0), score)
        
        # 更新context
        node.result["context"] = self.get_context_from_docids(
            docids_to_scores=node.result["docids_to_scores"], 
            result=node.result
        )

    def _process_single_rollout(self, nodes: List, generation_args: dict):
        """对节点进行单一rollout处理不分支"""
        # 过滤出可以检索的节点
        nodes_can_retrieve = []
        for node in nodes:
            if (node.result["retrieval_count"] < self._reasoning_model.max_retrieval_count and 
                node.result["request_times"] < self._reasoning_model.max_request_times):
                nodes_can_retrieve.append(node)
        
        if not nodes_can_retrieve:
            return
        
        # 批量处理检索决策
        batch_size = 4
        for i in range(0, len(nodes_can_retrieve), batch_size):
            batch_nodes = nodes_can_retrieve[i:i+batch_size]
            
            # 判断是否需要检索
            is_retrieval_required = self.is_retrieval_required_with_context(
                questions=[node.result["question"] for node in batch_nodes],
                thoughts=[
                    node.result["existing_thought"] if self._reasoning_model.is_long_reasoning_model else
                        self.convert_non_reasoning_model_thought_to_thought(node.result["existing_thought"])
                    for node in batch_nodes
                ],
                current_thoughts=[
                    node.result["current_thought"] if self._reasoning_model.is_long_reasoning_model else
                        self.convert_non_reasoning_model_thought_to_thought(node.result["current_thought"])
                    for node in batch_nodes
                ],
                contexts=[node.result["context"] for node in batch_nodes],
                generation_config=generation_args
            )
            
            # 更新节点状态
            nodes_require_retrieval = []
            for node, require_retrieval in zip(batch_nodes, is_retrieval_required):
                node.result["request_times"] += 1
                
                # 记录trajectory
                node.result["trajectory"].append({
                    "action": self.determine_retrieval_action,
                    "state": {
                        "question": node.result["question"],
                        "reasoning_history": node.result["existing_thought"] if self._reasoning_model.is_long_reasoning_model else
                            self.convert_non_reasoning_model_thought_to_thought(node.result["existing_thought"]),
                        "current_reasoning_step": node.result["current_thought"] if self._reasoning_model.is_long_reasoning_model else
                            self.convert_non_reasoning_model_thought_to_thought(node.result["current_thought"]),
                        "retrieved_context": "\n\n".join(node.result["context"]),
                    },
                    "response": require_retrieval
                })
                
                node.result["retrieval_trajectory"].append({
                    "action": self.determine_retrieval_action,
                    "question": node.result["question"],
                    "current_context_ids": self.get_context_ids(node.result["docids_to_scores"]),
                    "require_retrieval": require_retrieval
                })
                
                if require_retrieval == self.enable_retrieval_symbol:
                    nodes_require_retrieval.append(node)
            
            if not nodes_require_retrieval:
                continue
            
            # 后续的检索处理逻辑
            self._process_retrieval_for_nodes(nodes_require_retrieval, generation_args)

    def _process_retrieval_for_nodes(self, nodes: List, generation_args: dict):
        """为节点处理检索逻辑"""
        batch_size = 4
        
        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i:i+batch_size]
            
            # 构造检索query
            reasoning_chains, queries, responses = self.formulate_retrieval_queries(
                questions=[node.result["question"] for node in batch_nodes],
                thoughts=[node.result["output"] for node in batch_nodes],
                generation_config=generation_args
            )
            
            # 更新trajectory
            for node, response, query in zip(batch_nodes, responses, queries):
                node.result["trajectory"].append({
                    "action": self.formulate_retrieval_queries_action,
                    "state": {
                        "question": node.result["question"],
                        "thought": node.result["output"],
                    },
                    "response": response
                })
                node.result["retrieval_trajectory"].append({
                    "action": self.formulate_retrieval_queries_action,
                    "question": node.result["question"],
                    "query": query,
                })
            
            # 过滤有效query的节点
            valid_nodes = []
            valid_reasoning_chains = []
            valid_queries = []
            
            for node, reasoning_chain, query in zip(batch_nodes, reasoning_chains, queries):
                if query is not None:
                    valid_nodes.append(node)
                    valid_reasoning_chains.append(reasoning_chain)
                    valid_queries.append(query)
            
            if not valid_nodes:
                continue
            
            # 检索和处理triples
            retrieved_documents = self._retriever.retrieve(queries=valid_queries, topk=self._reasoning_model.topk)
            retrieved_triples, retrieved_triples_scores = [], []
            
            for node, query, documents in zip(valid_nodes, valid_queries, retrieved_documents):
                candidate_triples = self._retriever.get_candidate_triples_from_documents(documents)
                retrieved_triples_results = self._retriever.filter_triples(
                    node.result["question"], query, candidate_triples, self._reasoning_model.num_candidate_triples
                )
                retrieved_triples.append(retrieved_triples_results[0])
                retrieved_triples_scores.append(retrieved_triples_results[1])
            
            # 识别相关triples
            relevant_triples, triples_texts, responses = self.identify_relevant_triples(
                questions=[node.result["question"] for node in valid_nodes],
                reasoning_chains=valid_reasoning_chains,
                queries=valid_queries,
                triples=retrieved_triples,
                generation_config=generation_args
            )
            
            # 更新节点状态
            for node, reasoning_chain, query, triples, triples_text, response, one_retrieved_triples, one_retrieved_triples_scores in zip(
                valid_nodes, valid_reasoning_chains, valid_queries, relevant_triples, triples_texts, responses,
                retrieved_triples, retrieved_triples_scores
            ):
                node.result["trajectory"].append({
                    "action": self.identify_relevant_triples_action,
                    "state": {
                        "question": node.result["question"],
                        "reasoning_chain": reasoning_chain,
                        "query": query,
                        "candidate_triples": triples_text,
                    },
                    "response": response
                })
                
                if triples:
                    node.result["retrieval_trajectory"].append({
                        "action": self.identify_relevant_triples_action,
                        "question": node.result["question"],
                        "selected_context_id": triples[0]["reference"][0],
                    })
                
                # 更新context
                self._update_context_for_node(node, one_retrieved_triples, one_retrieved_triples_scores, triples)
    
    def tree_rollout_forward(self, sequences, num_actions, attention_mask, return_output=False, **kwargs) -> torch.Tensor:

        position_ids = attention_mask.long().cumsum(-1) - 1 
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)  
        output["logits"] = output["logits"].to(torch.float32)

        if num_actions is None:
            assert return_output 
            return output
        
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])
        action_log_probs = log_probs[:, -num_actions:]
        if return_output:
            return action_log_probs, output
        else:
            return action_log_probs 

