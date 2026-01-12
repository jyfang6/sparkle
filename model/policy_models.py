import json 
import regex
import logging
import torch.nn as nn
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Union
from torch import Tensor 

from generator.generator import Generator 
from generator.api_generator import OpenAIGPTGenerator
from prompts.adaptive_rag import * 

from utils.utils import HParams, parse_json_from_text 
from utils.pipeline_utils import (
    get_llm_params, 
    load_llm_tokenizer_and_model, 
    get_llm_generation_params
)

import sys 
sys.path.append("/nfs/common")
sys.path.append("/nfs/clayx/main_branch/EvoAgentX")
from my_evaluation import f1_score 


logger = logging.getLogger(__file__)


def load_policy_model(policy_model_config: HParams):
    config: dict = policy_model_config.get_hparams()
    tok, llm = load_llm_tokenizer_and_model(**get_llm_params(config=config))
    model = Generator(tokenizer=tok, generator=llm, **config)
    return model 


class KGAdaptivePolicyModel(nn.Module):

    def __init__(self, config: HParams):
        super().__init__()
        self.config = config
        self.model: Generator = load_policy_model(config)
        self.use_demo = config.get_hparams().get("use_demo", True)
        #! 根据generator的名字来判断是否使用chat form
        self.use_chat = True #! TODO  
        """
        model_name_or_path: str = config.get_hparams()["model_name_or_path"]
        if "qwen2.5_3b_instruct" in model_name_or_path.lower():
            print(f"Setting `use_chat=True` for KGAdaptivePolicyModel with `model_name_or_path={model_name_or_path}`")
            self.use_chat = True 
        elif "qwen2.5_3b" in model_name_or_path:
            print(f"Setting `use_chat=False` for KGAdaptivePolicyModel with `model_name_or_path={model_name_or_path}`")
            self.use_chat = False
        else:
            raise ValueError(f"{model_name_or_path} is an unknown model for KGAdaptivePolicyModel!")
        """

    def get_model_outputs(self, prompts: List[str]) -> Tuple[Tensor, Tensor]:

        generation_params = get_llm_generation_params(self.config.get_hparams())
        inputs = self.model.tokenizer_encode(prompts=prompts)
        token_ids, logits = self.model.generate(inputs, **generation_params)
        return token_ids, logits
    
    def get_is_retrieval_required_outputs(self, questions: List[str], thoughts: List[str]) -> Tuple[Tensor, Tensor]:

        # prompts = [
        #     DETERMINE_RETRIEVAL_PROMPT.format(question=question, thought=thought)
        #     for question, thought in zip(questions, thoughts)
        # ]
        # prompts = self.model.prompt(inputs=prompts, use_chat=True)
        if self.use_demo:
            instructions = [DETERMINE_RETRIEVAL_INSTRUCTION + "\n\n" + DETERMINE_RETRIEVAL_DEMO] * len(questions)
        else:
            instructions = [DETERMINE_RETRIEVAL_INSTRUCTION] * len(questions)
        inputs = [
            DETERMINE_RETRIEVAL_INPUTS.format(question=question, thought=thought)
            for question, thought in zip(questions, thoughts)
        ]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=True)
        # print("-"*100, f"\nDetermine Retrieval Required Prompts:\n{prompts[0]}\n", "-"*100)
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        return token_ids, logits
    
    def get_retrieval_query_outputs(self, questions: List[str], thoughts: List[str]) -> Tuple[Tensor, Tensor]:
        
        # prompts = [
        #     QUERY_FORMULATION_PROMPT.format(question=question, thought=thought)
        #     for question, thought in zip(questions, thoughts)
        # ]
        # prompts = self.model.prompt(inputs=prompts, use_chat=True)
        if self.use_demo:
            instructions = [QUERY_FORMULATION_INSTRUCTION + "\n\n" + QUERY_FORMULATION_DEMO] * len(questions)
        else:
            instructions = [QUERY_FORMULATION_INSTRUCTION] * len(questions)
        inputs = [
            QUERY_FORMULATION_INPUT.format(question=question, thought=thought)
            for question, thought in zip(questions, thoughts)
        ]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=self.use_chat) # use_chat=True)
        # print("-"*100, f"\nFormulate Retrieval Query Prompts:\n{prompts[0]}\n", "-"*100)
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        return token_ids, logits
    
    def get_relevant_triples_outputs(self, questions: List[str], reasoning_chains: List[str], queries: List[str], triples_texts: List[str]) -> Tuple[Tensor, Tensor]:
        
        # prompts = [
        #     RELEVANT_TRIPLES_PROMPT.format(question=question, reasoning_chain=reasoning_chain, candidate_triples=triples_text, K=topk)
        #     for question, reasoning_chain, triples_text in zip(questions, reasoning_chains, triples_texts)
        # ]
        # prompts = self.model.prompt(inputs=prompts, use_chat=True)
        if self.use_demo:
            instructions = [RELEVANT_TRIPLES_INSTRUCTION + "\n\n" + RELEVANT_TRIPLES_DEMO] * len(questions)
        else:
            instructions = [RELEVANT_TRIPLES_INSTRUCTION] * len(questions)
        inputs = [
            RELEVANT_TRIPLES_INPUT.format(question=question, reasoning_chain=reasoning_chain, query=query, candidate_triples=triples_text)
            for question, reasoning_chain, query, triples_text in zip(questions, reasoning_chains, queries, triples_texts)
        ]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=self.use_chat) # use_chat=True)
        # print("-"*100, f"\nIdentify Relevant Triples Prompts:\n{prompts[0]}\n", "-"*100)
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        return token_ids, logits
    
    def get_verbalize_triples_outputs(self, questions: List[str], thoughts: List[str], ctxs: List[str]) -> Tuple[Tensor, Tensor]:
        
        # prompts = [
        #     VERBALIZATION_PROMPT.format(question=question, thought=thought, ctx=ctx)
        #     for question, thought, ctx in zip(questions, thoughts, ctxs)
        # ]
        # prompts = self.model.prompt(inputs=prompts, use_chat=True)

        instructions = [VERBALIZATION_INSTRUCTION + "\n\n" + VERBALIZATION_DEMO] * len(questions)
        inputs = [
            VERBALIZATION_INPUT.format(question=question, thought=thought, ctx=ctx)
            for question, thought, ctx in zip(questions, thoughts, ctxs)
        ]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=True)
        # print("-"*100, f"\nVerbalization Prompt:\n{prompts[0]}\n", "-"*100)
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        return token_ids, logits
    
    def parse_retrieval_query_response(self, question: str, response: str) -> dict: 

        default_result = {"reasoning_chain": "", "query": ""}

        response = response.replace("**", "")
        # 匹配Step 1到下一个Step之间的所有内容
        reasoning_pattern = r"Step 1: Reasoning Chain:?\s*(.*?)(?=\n\s*Step|$)"
        reasoning_match = regex.search(reasoning_pattern, response, regex.DOTALL)
        if reasoning_match:
            reasoning_chain = reasoning_match.group(1).strip()
            default_result["reasoning_chain"] = reasoning_chain
        else:
            default_result["reasoning_chain"] = "Failed to parse reasoning chain: " + response
        
        # 匹配Step 3到结尾的内容 
        query_pattern = r"Step 3:.*?Query:?\s*(.*?)$"
        query_match = regex.search(query_pattern, response, regex.DOTALL)
        if query_match:
            query = query_match.group(1).strip()
            default_result["query"] = query
            if "none" in query.lower():
                default_result["query"] = None
        else:
            default_result["query"] = f"Question: {question}\n{response}"

        return default_result

    def parse_relevant_triples_response(self, response: str, triples: List[Dict[str, Union[str, List]]]) -> List[Dict[str, Union[str, List]]]:

        triples_texts_list = []
        # pattern = r'\d+\.\s*(.*?)(?:\n|$)'
        # matches = regex.finditer(pattern, response, regex.DOTALL)
        # for mat in matches:
        #     triple_text = mat.group(1).strip()
        #     if triple_text:
        #         # 尝试提取<>内的内容
        #         angle_bracket_pattern = r'<([^>]+)>'
        #         angle_bracket_match = regex.search(angle_bracket_pattern, triple_text)
        #         if angle_bracket_match:
        #             # 如果找到<>内的内容，使用它
        #             triple_text = angle_bracket_match.group(1)
        #         triples_texts_list.append(triple_text)

        angle_bracket_pattern = r'<([^>]+)>'
        matches = regex.finditer(angle_bracket_pattern, response)
        for mat in matches:
            triple_text = mat.group(1).strip()
            if triple_text:
                triples_texts_list.append("<" + triple_text + ">")
        
        if not triples_texts_list:
            # 如果无法提取出任何三元组，则将整个响应作为三元组
            triples_texts_list.append(response)
        
        selected_triples = []
        selected_triples_indices = set() 
        for triple_text in triples_texts_list:
            # similarities = [SequenceMatcher(None, triple_text, triple["text"]).ratio() for triple in triples] 
            similarities = [f1_score(triple_text, triple["text"])[0] for triple in triples]
            max_similarity_index = similarities.index(max(similarities))
            if max_similarity_index not in selected_triples_indices:
                selected_triples.append(triples[max_similarity_index])
                selected_triples_indices.add(max_similarity_index)

        return selected_triples
    
    def parse_verbalize_triples_response(self, response: str) -> str:
        pattern = r'(?i)Continuation Text[^:]*:\s*(.*?)$'
        match = regex.search(pattern, response, regex.DOTALL)
        if match:
            continuation_text = match.group(1).strip()
            return continuation_text
        return response.strip()
    
    def is_retrieval_required(self, question: str, thought: str, return_raw_response: bool = False) -> bool:

        token_ids, _ = self.get_is_retrieval_required_outputs(questions=[question], thoughts=[thought])
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        required = "yes" in response.lower()
        print("\n##>>>>>>>>>>>>>>>>>> Is Retrieval Required:", required)
        if return_raw_response:
            return required, response
        return (required, )

    def is_retrieval_required_with_context(self, question: str, thought: str, current_thought: str, context: List[str], return_raw_response: bool = False) -> bool:
        
        if self.use_demo:
            instructions = [DETERMINE_RETRIEVAL_INSTRUCTION_WITH_CONTEXT + "\n\n" + DETERMINE_RETRIEVAL_DEMO_WITH_CONTEXT]
        else:
            instructions = [DETERMINE_RETRIEVAL_INSTRUCTION_WITH_CONTEXT]
        # truncate long context 
        context = [text if len(text.split())<=250 else " ".join(text.split()[:250]) for text in context]
        inputs = [
            DETERMINE_RETRIEVAL_INPUTS_WITH_CONTEXT.format(
                retrieved_context = "\n\n".join(context), 
                question=question, 
                reasoning_history=thought, 
                current_reasoning_step=current_thought
            )
        ]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=self.use_chat) # use_chat=True)
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        print("\n##>>>>>>>>>>>>>>>>>> Determine Retrieval Response:", response)
        required = "yes" in response.lower()
        print("\n##>>>>>>>>>>>>>>>>>> Is Retrieval Required:", required)
        if return_raw_response:
            return required, response
        return (required, )
    
    def is_retrieval_required_with_thought(self, question: str, thought: str, return_raw_response: bool = False) -> bool:

        if self.use_demo:
            instructions = [DETERMINE_RETRIEVAL_INSTRUCTION_WITH_THOUGHT + "\n\n" + DETERMINE_RETRIEVAL_DEMO_WITH_THOUGHT]
        else:
            instructions = [DETERMINE_RETRIEVAL_INSTRUCTION_WITH_THOUGHT]
        inputs = [DETERMINE_RETRIEVAL_INPUTS_WITH_THOUGHT.format(question=question, thought=thought)]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=True)
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        pattern = r'(?:###\s*Answer:?\s*)(.*?)(?=\n\s*(?:###|$)|$)'
        match = regex.search(pattern, response, regex.DOTALL)
        answer_part = match.group(1).strip() if match else response 
        required = not any(key_word in answer_part.lower() for key_word in ["no", "not"])
        if return_raw_response:
            return required, response
        return (required, )
    
    def formulate_retrieval_query(self, question: str, thought: str, return_raw_response: bool = False) -> dict:
        
        token_ids, _ = self.get_retrieval_query_outputs(questions=[question], thoughts=[thought])
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        results = self.parse_retrieval_query_response(question=question, response=response)
        print("\n# >>>>>>>>>>>>>>>>> Formulate Retrieval Query Response <<<<<<<<<<<<<<<<<<<<<\n", response)
        print("\n# >>>>>>>>>>>>>>>>> Formulated Reasoning Chain <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", results["reasoning_chain"])
        print("\n# >>>>>>>>>>>>>>>>> Formulated Retrieval Query <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", results["query"])
        if return_raw_response:
            return results, response
        return (results, )
    
    def formulate_retrieval_query_with_thought(self, question: str, thought: str, return_raw_response: bool = False) -> dict:

        if self.use_demo:
            instructions = [QUERY_FORMULATION_INSTRUCTION_WITH_THOUGHT + "\n\n" + QUERY_FORMULATION_DEMO_WITH_THOUGHT]
        else:
            instructions = [QUERY_FORMULATION_INSTRUCTION_WITH_THOUGHT]
        inputs = [QUERY_FORMULATION_INPUT_WITH_THOUGHT.format(question=question, thought=thought)]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=True)
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        # parse the response
        pattern = r'(?:###\s*Output:?\s*)(.*?)(?=\n\s*(?:###|$)|$)' 
        match = regex.search(pattern, response, regex.DOTALL)
        output = match.group(1).strip() if match else response
        results = self.parse_retrieval_query_response(question=question, response=output)
        print("\n# >>>>>>>>>>>>>>>>> Formulate Retrieval Query Response <<<<<<<<<<<<<<<<<<<<<\n", response)
        print("\n# >>>>>>>>>>>>>>>>> Formulated Reasoning Chain <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", results["reasoning_chain"])
        print("\n# >>>>>>>>>>>>>>>>> Formulated Retrieval Query <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", results["query"])
        if return_raw_response:
            return results, response
        return (results, )

    def identify_relevant_triples(self, question: str, reasoning_chain: str, query: str, triples: List[Dict[str, Union[str, List]]], return_raw_response: bool = False) -> List[Dict[str, Union[str, List]]]:

        # triples_texts = [triple["text"] for triple in triples]
        triples_texts = [] 
        for triple in triples:
            triples_texts.append(f"Source Title: {triple['title']}\nText: {triple['text']}")
        triples_text = "\n".join([f"{i+1}. {triple_text}" for i, triple_text in enumerate(triples_texts)])
        token_ids, _ = self.get_relevant_triples_outputs(questions=[question], reasoning_chains=[reasoning_chain], queries=[query], triples_texts=[triples_text])
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        relevant_triples = self.parse_relevant_triples_response(response=response, triples=triples)
        print("\n# >>>>>>>>>>>>>>>>> Identify Relevant Triples Response <<<<<<<<<<<<<<<<<<<<<\n", response)
        print("\n# >>>>>>>>>>>>>>>>> Identified Relevant Triples <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", relevant_triples)
        if return_raw_response:
            return relevant_triples, response
        return (relevant_triples, )
    
    def identify_relevant_triples_with_thought(self, question: str, thought: str, reasoning_chain: str, query: str, triples: List[Dict[str, Union[str, List]]], return_raw_response: bool = False) -> List[Dict[str, Union[str, List]]]:
        
        triples_texts = [] 
        for triple in triples:
            triples_texts.append(f"Source Title: {triple['title']}\nText: {triple['text']}")
        triples_text = "\n".join([f"{i+1}. {triple_text}" for i, triple_text in enumerate(triples_texts)])

        if self.use_demo:
            instructions = [RELEVANT_TRIPLES_INSTRUCTION_WITH_THOUGHT + "\n\n" + RELEVANT_TRIPLES_DEMO_WITH_THOUGHT]
        else:
            instructions = [RELEVANT_TRIPLES_INSTRUCTION_WITH_THOUGHT] 
        inputs = [RELEVANT_TRIPLES_INPUT_WITH_THOUGHT.format(question=question, thought=thought, reasoning_chain=reasoning_chain, query=query, candidate_triples=triples_text)]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=True)
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        pattern = r'(?:###\s*Selected Triple:?\s*)(.*?)(?=\n\s*(?:###|$)|$)' 
        match = regex.search(pattern, response, regex.DOTALL)
        output = match.group(1).strip() if match else response
        relevant_triples = self.parse_relevant_triples_response(response=output, triples=triples)
        print("\n# >>>>>>>>>>>>>>>>> Identify Relevant Triples Response <<<<<<<<<<<<<<<<<<<<<\n", response)
        print("\n# >>>>>>>>>>>>>>>>> Identified Relevant Triples <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", relevant_triples)
        if return_raw_response:
            return relevant_triples, response
        return (relevant_triples, )
    
    def identify_relevant_triples_with_context(self, question: str, reasoning_chain: str, query: str, triples: List[Dict[str, Union[str, List]]], context: List[str], return_raw_response: bool = False) -> List[Dict[str, Union[str, List]]]:

        triples_texts = [] 
        for triple in triples:
            triples_texts.append(f"Source Title: {triple['title']}\nText: {triple['text']}")
        triples_text = "\n".join([f"{i+1}. {triple_text}" for i, triple_text in enumerate(triples_texts)])

        if self.use_demo:
            instructions = [RELEVANT_TRIPLES_INSTRUCTION_WITH_CONTEXT + "\n\n" + RELEVANT_TRIPLES_DEMO_WITH_CONTEXT]
        else:
            instructions = [RELEVANT_TRIPLES_INSTRUCTION_WITH_CONTEXT]
        
        inputs = [
            RELEVANT_TRIPLES_INPUT_WITH_CONTEXT.format(
                question=question, 
                reasoning_chain=reasoning_chain, 
                query=query, 
                candidate_triples=triples_text,
                context="\n".join(context)
            )
        ]

        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=True)
        # prompts = self.model.prompt(inputs=[inst + "\n" + inp for inst, inp in zip(instructions, inputs)], use_chat=True)
        # from pdb import set_trace; set_trace()
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        relevant_triples = self.parse_relevant_triples_response(response=response, triples=triples)
        print("\n# >>>>>>>>>>>>>>>>> Identify Relevant Triples Response <<<<<<<<<<<<<<<<<<<<<\n", response)
        print("\n# >>>>>>>>>>>>>>>>> Identified Relevant Triples <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", relevant_triples)
        if return_raw_response:
            return relevant_triples, response
        return (relevant_triples, ) 

    def verbalize_triples(self, question: str, thought: str, triples: List[Dict[str, Union[str, List]]], documents: List[Dict[str, Union[str, List]]]) -> str:

        def get_document_text(document: Dict[str, Union[str, List]]) -> str:
            text = document.get("text", None)
            if not text:
                text = " ".join([s.strip() for s in document.get("sentences", [])])
            return text 
        
        triples_texts = [triple["text"] for triple in triples]
        documents_texts = [get_document_text(document) for document in documents]
        context_list = [] 
        for triple_text, document_text in zip(triples_texts, documents_texts):
            context_list.append(f"Relevant Triple: {triple_text}\nSource Document: {document_text}\n")
        context_text = "\n".join(context_list)

        token_ids, _ = self.get_verbalize_triples_outputs(questions=[question], thoughts=[thought], ctxs=[context_text])
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        continuation_text = self.parse_verbalize_triples_response(response=response)
        print("\n# >>>>>>>>>>>>>>>>> Verbalize Triples Response <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", response)
        print("\n# >>>>>>>>>>>>>>>>> Verbalized Text <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", continuation_text)
        return continuation_text

    def get_ablation1_outputs(self, questions: List[str], thoughts: List[str]) -> Tuple[Tensor, Tensor]:

        instructions = [ABLATION1_INSTRUCTION] * len(questions)
        inputs = [ABLATION1_INPUT.format(question=question, thought=thought) for question, thought in zip(questions, thoughts)]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=True)
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        return token_ids, logits

    def ablation1(self, question: str, thought: str, return_raw_response: bool = False) -> bool:

        """
        直接用一个prompt来判断是否需要检索以及检索的query是什么
        """
        token_ids, _ = self.get_ablation1_outputs(questions=[question], thoughts=[thought])
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0] 
        print("\n# >>>>>>>>>>>>>>>>> Ablation1 Response <<<<<<<<<<<<<<<<<<<<<<<\n", response)
        # parse the response
        try:
            data = json.loads(parse_json_from_text(response)[0])
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return True, question

        if data["retrieval_required"].lower() == "yes" and data["retrieval_query"] is not None:
            return True, data["retrieval_query"]
        
        return False, None

    def joint_determine_retrieval_and_query_formulation(self, question: str, thought: str, current_thought: str, context: List[str], return_raw_response: bool = False) -> bool:
        
        instructions = [JOINT_DETERMINE_RETRIEVAL_QUERY_INSTRUCTION_WITH_CONTEXT]
        # truncate long context 
        context = [text if len(text.split())<=250 else " ".join(text.split()[:250]) for text in context]
        inputs = [
            DETERMINE_RETRIEVAL_INPUTS_WITH_CONTEXT.format(
                retrieved_context = "\n\n".join(context), 
                question=question, 
                reasoning_history=thought, 
                current_reasoning_step=current_thought
            )
        ]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=self.use_chat) 
        token_ids, logits = self.get_model_outputs(prompts=prompts)
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        print("\n### Joint Determine Retrieval and Query Formulation: ", response)
        try:
            # data = json.loads(parse_json_from_text(response)[0])
            required_match = regex.search(r'"retrieval_required"\s*:\s*"([^"]+)"', response)
            retrieval_required = required_match.group(1) if required_match else None
            query_match = regex.search(r'"retrieval_query"\s*:\s*"([^"]+)"', response)
            retrieval_query = query_match.group(1) if query_match else None
            data = {"retrieval_required": retrieval_required, "retrieval_query": retrieval_query}
        except Exception as e:
            logger.error(f"Error parsing response: {e}.")
            return True, question

        if data["retrieval_required"].lower() == "yes" and data["retrieval_query"] is not None:
            return True, data["retrieval_query"]

        return False, None 
    
    def direct_formulate_retrieval_query(self, question: str, thought: str) -> dict:

        instructions = [DIRECE_QUERY_FORMULATION_INSTRUCTION]
        inputs = [DIRECT_QUERY_FORMULATION_INPUT.format(question=question, thought=thought)]
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=self.use_chat)
        token_ids, _ = self.get_model_outputs(prompts=prompts) 
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0] 
        print("\n### Direct Query Formulation Response: ", response) 
        return response 
    
    def direct_document_selection(self, question: str, thought: str, documents: List[dict]) -> List[dict]:

        texts = [doc["text"] if "text" in doc else " ".join(doc["sentences"]) for doc in documents]
        context = "\n".join([f"[{i+1}] {text}" for i, text in enumerate(texts)])

        instructions = [DIRECT_KNOWLEDGE_INTEGRATION_INSTRUCTION]
        inputs = [DIRECT_KNOWLEDGE_INTEGRATION_INPUT.format(question=question, thought=thought, context=context)] 
        prompts = self.model.prompt(instructions=instructions, inputs=inputs, use_chat=self.use_chat)
        token_ids, _ = self.get_model_outputs(prompts=prompts) 
        response = self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0] 

        print("\n### Direct Knowledge Integration Response: ", response)
        response = response.replace("[", "").replace("]", "").strip()
        try:
            idx = int(response)
            return documents[idx] 
        except:
            print("Failed to parse selected document. Return the top-rank document instead.")
            idx = 0 
            return documents[idx]

class OpenAIKGAdaptivePolicyModel(KGAdaptivePolicyModel):

    def __init__(self, config: HParams):
        super(KGAdaptivePolicyModel, self).__init__()
        self.config = config

        #    self.model: OpenAIGPTGenerator = OpenAIGPTGenerator(
        #        model_name=config.model_name_or_path, 
        #        max_new_tokens=config.max_new_tokens, 
        #        num_outputs=1, 
        #    )
        from evoagentx.models import OpenRouterConfig, OpenRouterLLM 
        from setup.setup import OPENROUTER_API_KEY
        config = OpenRouterConfig(
            model = config.model_name_or_path, 
            openrouter_key = OPENROUTER_API_KEY,
            max_tokens = config.max_new_tokens, 
            temperature = config.temperature
        )
        self.model = OpenRouterLLM(config)

    def is_retrieval_required(self, question: str, thought: str, return_raw_response: bool = False) -> bool:

        instruction = DETERMINE_RETRIEVAL_INSTRUCTION + "\n\n" + DETERMINE_RETRIEVAL_DEMO
        user_input = DETERMINE_RETRIEVAL_INPUTS.format(question=question, thought=thought)

        response = self.model.generate(
            # instruction=instruction, 
            # user_inputs=user_input, 
            system_message=instruction, 
            prompt=user_input
        ).content 
        required = "yes" in response.lower()
        print("\n##>>>>>>>>>>>>>>>>>> Is Retrieval Required:", required)
        if return_raw_response:
            return required, response
        return required
    

    def formulate_retrieval_query(self, question: str, thought: str, return_raw_response: bool = False) -> dict:

        instruction = QUERY_FORMULATION_INSTRUCTION + "\n\n" + QUERY_FORMULATION_DEMO
        user_input = QUERY_FORMULATION_INPUT.format(question=question, thought=thought)

        # from pdb import set_trace; set_trace()
        response = self.model.generate(
            # instruction=instruction, 
            # user_inputs=user_input, 
            # model="openai/gpt-4o", 
            system_message=instruction, 
            prompt=user_input
        ).content 
        results = self.parse_retrieval_query_response(question=question, response=response)
        print("\n# >>>>>>>>>>>>>>>>> Formulate Retrieval Query Response <<<<<<<<<<<<<<<<<<<<<\n", response)
        print("\n# >>>>>>>>>>>>>>>>> Formulated Reasoning Chain <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", results["reasoning_chain"])
        print("\n# >>>>>>>>>>>>>>>>> Formulated Retrieval Query <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", results["query"])
        if return_raw_response:
            return results, response
        return results
    

    def identify_relevant_triples(self, question: str, reasoning_chain: str, query: str, triples: List[Dict[str, Union[str, List]]], return_raw_response: bool = False) -> List[Dict[str, Union[str, List]]]:

        triples_texts = [] 
        for triple in triples:
            triples_texts.append(f"Source Title: {triple['title']}\nText: {triple['text']}")
        triples_text = "\n".join([f"{i+1}. {triple_text}" for i, triple_text in enumerate(triples_texts)])

        instruction = RELEVANT_TRIPLES_INSTRUCTION + "\n\n" + RELEVANT_TRIPLES_DEMO
        user_input = RELEVANT_TRIPLES_INPUT.format(question=question, reasoning_chain=reasoning_chain, query=query, candidate_triples=triples_text)

        response = self.model.generate(
            # instruction=instruction, 
            # user_inputs=user_input, 
            system_message=instruction, 
            prompt=user_input
        ).content 
        relevant_triples = self.parse_relevant_triples_response(response=response, triples=triples)
        print("\n# >>>>>>>>>>>>>>>>> Identify Relevant Triples Response <<<<<<<<<<<<<<<<<<<<<\n", response)
        print("\n# >>>>>>>>>>>>>>>>> Identified Relevant Triples <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", relevant_triples)
        if return_raw_response:
            return relevant_triples, response
        return relevant_triples

    def is_retrieval_required_with_context(self, question: str, thought: str, current_thought: str, context: List[str], return_raw_response: bool = False) -> bool:
        
        instruction = DETERMINE_RETRIEVAL_INSTRUCTION_WITH_CONTEXT # + "\n\n" + DETERMINE_RETRIEVAL_DEMO_WITH_CONTEXT
        # truncate long context 
        
        context = [text if len(text.split())<=250 else " ".join(text.split()[:250]) for text in context]
        user_input = DETERMINE_RETRIEVAL_INPUTS_WITH_CONTEXT.format(
            retrieved_context = "\n\n".join(context), 
            question=question, 
            reasoning_history=thought, 
            current_reasoning_step=current_thought
        )
        # from pdb import set_trace; set_trace()
        response = self.model.generate(
            # instruction=instruction, 
            # user_inputs=user_input, 
            system_message=instruction, 
            prompt=user_input
        ).content
        print("\n##>>>>>>>>>>>>>>>>>> Determine Retrieval Response:", response)
        required = "yes" in response.lower()
        print("\n##>>>>>>>>>>>>>>>>>> Is Retrieval Required:", required)
        if return_raw_response:
            return required, response
        return required

    def identify_relevant_triples_with_context(self, question: str, reasoning_chain: str, query: str, triples: List[Dict[str, Union[str, List]]], context: List[str], return_raw_response: bool = False) -> List[Dict[str, Union[str, List]]]:

        triples_texts = [] 
        for triple in triples:
            triples_texts.append(f"Source Title: {triple['title']}\nText: {triple['text']}")
        triples_text = "\n".join([f"{i+1}. {triple_text}" for i, triple_text in enumerate(triples_texts)])

        instruction = RELEVANT_TRIPLES_INSTRUCTION_WITH_CONTEXT + "\n\n" + RELEVANT_TRIPLES_DEMO_WITH_CONTEXT
        user_input = RELEVANT_TRIPLES_INPUT_WITH_CONTEXT.format(
            question=question, 
            reasoning_chain=reasoning_chain, 
            query=query, 
            candidate_triples=triples_text,
            context="\n".join(context)
        )

        # from pdb import set_trace; set_trace()
        response = self.model.generate(
            # instruction=instruction, 
            # user_inputs=user_input, 
            system_message=instruction, 
            prompt=user_input
        ).content 
        relevant_triples = self.parse_relevant_triples_response(response=response, triples=triples)
        print("\n# >>>>>>>>>>>>>>>>> Identify Relevant Triples Response <<<<<<<<<<<<<<<<<<<<<\n", response)
        print("\n# >>>>>>>>>>>>>>>>> Identified Relevant Triples <<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n", relevant_triples)
        if return_raw_response:
            return relevant_triples, response
        return relevant_triples
    