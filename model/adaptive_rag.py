import nltk 
import torch 
import re 
import logging
import torch.nn as nn
from copy import deepcopy 
from typing import List, Dict, Optional, Union, Any
from difflib import SequenceMatcher

from transformers import AutoTokenizer
from retriever.retrievers import BaseRetriever, DenseRetriever
from data.kg_adaptive_collators import E5Collator, BGECollator, ContrieverCollator

from knowledge_graph.kg_generator import KGGenerator
from generator.generator import Generator 
from prompts.adaptive_rag import (
    ANSWER_PROMPT, ANSWER_PROMPT_FOR_NON_REASONING_MODEL,
    ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA,
    ANSWER_PROMPT_LONGFORMQA,
    ANSWER_PROMPT_FOR_NON_REASONING_MODEL_OPENROUTER,
    ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA_OPENROUTER,
    ANSWER_PROMPT_OPENROUTER,
    ANSWER_PROMPT_FOR_NON_REASONING_MODEL_WO_RETRIEVAL_DECISION_AGENT,
    ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA_WO_RETRIEVAL_DECISION_AGENT, 
)
from model.policy_models import KGAdaptivePolicyModel, OpenAIKGAdaptivePolicyModel 

from utils.utils import HParams, to_device, escape_braces
from utils.pipeline_utils import (
    get_llm_params, 
    load_llm_tokenizer_and_model, 
    get_llm_generation_params
)

logger = logging.getLogger(__file__)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading 'punkt_tab' tokenizer...")
    nltk.download('punkt_tab')


OPENROUTER_MODEL_MAP = {
    "qwen2.5_72b_instruct": "qwen/qwen-2.5-72b-instruct", 
    "gpt_4o": "openai/gpt-4o", 
    "qwq_32b": "qwen/qwq-32b", 
    "deepseek-r1": "deepseek/deepseek-r1-0528",
    "gpt_4.1_mini": "openai/gpt-4.1-mini",
    "gpt_4o_mini": "openai/gpt-4o-mini"
}

class OpenRouterLLMWrapper:

    def __init__(self, config: dict, **kwargs): 

        from evoagentx.models import OpenRouterConfig, OpenRouterLLM 
        from setup.setup import OPENROUTER_API_KEY

        openrouter_params = {
            "model": OPENROUTER_MODEL_MAP[config["model_name_or_path"]], 
            "openrouter_key": OPENROUTER_API_KEY, 
            "max_tokens": config["max_new_tokens"], 
            "stream": False, 
            "output_response": True 
        }
        for param in ["temperature", "top_p", "top_k"]:
            if param in config:
                openrouter_params[param] = config[param]
        
        self.model = OpenRouterLLM(OpenRouterConfig(**openrouter_params))
        self.model_name = config["model_name_or_path"]
        self.device = torch.device("cuda")
        self.kwargs = kwargs 
    
    def prompt(self, inputs: List[str], **kwargs) -> List[str]:
        return inputs 
    
    def generate(self, prompts: List[str], stop_words: List[str], **kwargs) -> List[str]:

        # if self.model_name in ["qwq_32b", "deepseek_r1"]:
        #     generated_texts = [] 
        #     for prompt in prompts:
        #         messages = [{"role": "user", "content": prompt}]
        #         completion = self.model._client.chat.completions.create(
        #             messages=messages, 
        #             **self.model.get_completion_params(stop=stop_words)
        #         )
        #         reasoning_text = completion.choices[0].message.reasoning 
        #         text = completion.choices[0].message.content
        #         if reasoning_text:
        #             generated_texts.append(reasoning_text)
        #         elif text:
        #             generated_texts.append(text)
        #     return generated_texts

        responses = self.model.generate(prompt=prompts, stop=stop_words)
        return [res.content for res in responses]


def load_reasoning_model(reasoning_model_config: HParams):
    config: dict = reasoning_model_config.get_hparams()
    model_name_or_path = config.get("model_name_or_path")
    if any(api_name in model_name_or_path.lower() for api_name in ["qwen2.5_72b_instruct", "gpt_4o", "deepseek-r1", "gpt_4.1_mini", "gpt_4o_mini"]):
        model = OpenRouterLLMWrapper(config=config) 
        return model 
    # tok, llm = load_llm_tokenizer_and_model(**get_llm_params(config=config))
    llm_params = get_llm_params(config=config)
    if llm_params["model_name"] == "qwen2.5_14b_instruct":
        llm_params["load_in_4bit"] = False 
        llm_params["load_in_8bit"] = True 
    elif llm_params["model_name"] in ["qwen2.5_32b_instruct", "qwq_reasoning_32b"]:
        llm_params["load_in_4bit"] = True 
        llm_params["load_in_8bit"] = False
    tok, llm = load_llm_tokenizer_and_model(**llm_params)
    model = Generator(tokenizer=tok, generator=llm, **config)
    return model 

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

class KGAdaptiveRAG(nn.Module):

    def __init__(
        self, 
        retriever: DenseRetriever, 
        kg_generator: KGGenerator, 
        policy_model_config: HParams, 
        reasoning_model_config: HParams, 
        triple_filter_retriever: str = "e5", 
        triple_filter_retriever_path: str = None, 
        num_candidate_triples: int = 25, 
        max_num_context: int = 10, 
        max_reasoning_steps: int = 20,
        max_retrieval_count: int = 5, 
        max_request_times: int = None, 
        topk: int = 10, 
        batch_size: int = 1, 
        **kwargs 
    ):
        super().__init__()

        self.retriever = retriever
        self.kg_generator = kg_generator
        # load policy model 
        self.policy_model_config = policy_model_config
        if policy_model_config.model_name_or_path in ["gpt-4o-mini", "gpt-4o"]:
            self.policy_model: OpenAIKGAdaptivePolicyModel = OpenAIKGAdaptivePolicyModel(policy_model_config)
        else:
            self.policy_model: KGAdaptivePolicyModel = KGAdaptivePolicyModel(policy_model_config)
        # load reasoning model 
        self.reasoning_model_config = reasoning_model_config
        # 判断reasoning model的类型
        if self.reasoning_model_config.model_name_or_path in ["deepseek_r1_distill_qwen_14b", "qwq_reasoning_32b"]: # , "deepseek_r1"]:
            self.use_long_reasoning_model = True 
        elif self.reasoning_model_config.model_name_or_path in ["qwen2.5_3b_instruct", "qwen2.5_7b_instruct", "qwen2.5_14b_instruct", "qwen2.5_32b_instruct", "qwen2.5_7b", "qwen2.5_72b_instruct", "gpt_4o", "gpt_4.1_mini", "gpt_4o_mini"]:
            self.use_long_reasoning_model = False
        else:
            raise ValueError(f"Unsupported reasoning model: {self.reasoning_model_config.model_name_or_path} for KGAdaptiveRAG!")
        
        self.reasoning_model: Generator = load_reasoning_model(reasoning_model_config)
        self.device = self.reasoning_model.device 
        self.triple_filter_collator, self.triple_filter = self.load_triple_filter(triple_filter_retriever, triple_filter_retriever_path)
        self.topk = topk
        self.batch_size = batch_size
        self.num_candidate_triples = num_candidate_triples 
        self.max_num_context = max_num_context  
        self.max_reasoning_steps = max_reasoning_steps
        self.max_retrieval_count = max_retrieval_count 
        self.max_request_times = max_request_times or max_reasoning_steps

    def load_triple_filter(self, retriever_name: str, retriever_path: Optional[str]=None):

        if "e5" in retriever_name.lower():
            default_retriever_path = "intfloat/e5-large-v2"
        elif "bge" in retriever_name.lower():
            default_retriever_path = "BAAI/bge-large-en-v1.5"
        elif "contriever" in retriever_name.lower():
            default_retriever_path = "facebook/contriever"
        else:
            raise NotImplementedError(f"triple filter retriever \"{retriever_name}\" is not implemented!")

        if retriever_path is None:
            retriever_path = default_retriever_path
        tokenizer = AutoTokenizer.from_pretrained(default_retriever_path)
        # if retriever_name == "finetuned_e5":
        if "e5" in retriever_name.lower():
            collator = E5Collator(tokenizer=tokenizer, query_maxlength=256, doc_maxlength=128)
            logger.info(f"loading finetuned E5 retriever from {retriever_path}...")
            retriever = BaseRetriever(retriever_name="E5Retriever", model_name_or_path=retriever_path)
        # elif retriever_name == "finetuned_bge":
        elif "bge" in retriever_name.lower():
            collator = BGECollator(tokenizer=tokenizer, query_maxlength=256, doc_maxlength=128)
            logger.info(f"loading finetuned BGE retriever from {retriever_path}...")
            retriever = BaseRetriever(retriever_name="BGERetriever", model_name_or_path=retriever_path)
        # elif retriever_name == "finetuned_contriever":
        elif "contriever" in retriever_name.lower():
            collator = ContrieverCollator(tokenizer=tokenizer, query_maxlength=256, doc_maxlength=128)
            logger.info(f"loading finetuned Contriever retriever from {retriever_path}...")
            retriever = BaseRetriever(retriever_name="ContrieverRetriever", model_name_or_path=retriever_path)
        else:
            raise NotImplementedError(f"triple filter retriever \"{retriever_name}\" is not implemented!")

        retriever.to(self.device)
        retriever.eval()
        return collator, retriever

    def generate(self, prompts: List[str], config: dict, stop_words: List[str] = None) -> List[str]:
        """
        Generate text from prompts using the reasoning model.

        Args:
            prompts (List[str]): List of prompts to generate text from. Assumed to be user prompts, 
                will use apply_chat_template to convert to chat format.
            config (dict): Dictionary of generation configuration.
            stop_words (List[str]): List of words to stop generation at.

        Returns:
            List of generated text.
        """
        if isinstance(self.reasoning_model, OpenRouterLLMWrapper):
            texts = self.reasoning_model.generate(prompts, stop_words=stop_words)
            return texts  
        
        generation_params = get_llm_generation_params(config)
        inputs = self.reasoning_model.tokenizer_encode(prompts=prompts)
        outputs = self.reasoning_model.generate(inputs, **generation_params, stop_words=stop_words)[0]
        texts = self.reasoning_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts 

    def get_candidate_triples_from_documents(self, documents: List[Dict[str, Union[str, List]]]):
        """
        Input: 
            documents: [{"id": str, "title": str, "text": str / "sentence": str, "triples": ["text": str, "sentence": int]}]
        Output:
            triples: [{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...]
        """
        triples = [] 
        for doc in documents:
            doc_id = doc["id"]
            title = doc["title"]
            for one_triple_in_doc in doc["triples"]:
                triple = {
                    "title": title,
                    "text": one_triple_in_doc["text"],
                    "reference": [doc_id, one_triple_in_doc["sentence"]]
                }
                triples.append(triple)
        
        return triples

    def calculate_triple_filter_triple_embeddings(self, triples_texts: List[str], max_length: int=None):

        original_collator_doc_maxlength = self.triple_filter_collator.doc_maxlength
        if max_length is not None:
            self.triple_filter_collator.doc_maxlength = max_length

        triple_embeddings_list = [] 
        for i in range((len(triples_texts)-1) // self.batch_size + 1):
            batch_triple_texts = triples_texts[i*self.batch_size: (i+1)*self.batch_size]
            batch_triple_args = self.triple_filter_collator.encode_doc(batch_triple_texts)
            batch_triple_args = to_device(batch_triple_args, device=self.device)
            batch_triple_embeddings = self.triple_filter.doc(batch_triple_args)
            triple_embeddings_list.append(batch_triple_embeddings.detach().cpu())
        triple_embeddings = torch.cat(triple_embeddings_list, dim=0)

        if max_length is not None:
            self.triple_filter_collator.doc_maxlength = original_collator_doc_maxlength
        
        return triple_embeddings  

    def calculate_triple_filter_query_embeddings(self, query_list: List[str], max_length: int=None):

        original_collator_query_maxlength = self.triple_filter_collator.query_maxlength
        if max_length is not None:
            self.triple_filter_collator.query_maxlength = max_length
        
        query_embeddings_list = [] 
        for i in range((len(query_list)-1)//self.batch_size+1):
            batch_query_list = query_list[i*self.batch_size: (i+1)*self.batch_size]
            batch_query_args = self.triple_filter_collator.encode_query(batch_query_list)
            batch_query_args = to_device(batch_query_args, device=self.device)
            batch_query_embeddings = self.triple_filter.query(batch_query_args)
            query_embeddings_list.append(batch_query_embeddings.detach().cpu())
        query_embeddings = torch.cat(query_embeddings_list, dim=0)

        if max_length is not None:
            self.triple_filter_collator.query_maxlength = original_collator_query_maxlength
        
        return query_embeddings 

    def filter_candidate_triples(self, question: str, query: str, triples: List[Dict[str, Union[str, List]]]):

        num_triples = len(triples)
        triples_texts = [triple["text"] for triple in triples]
        triples_embeddings = self.calculate_triple_filter_triple_embeddings(triples_texts=triples_texts)
        query_embeddings = self.calculate_triple_filter_query_embeddings(query_list=[question + " " + query])
        query_triple_similarities = torch.matmul(query_embeddings, triples_embeddings.T)
        topk_relevant_triples_scores, topk_relevant_triples_indices = torch.topk(
            query_triple_similarities, 
            k = min(self.num_candidate_triples, num_triples),
            dim=1
        )
        # [0]是因为只有一个query
        topk_relevant_triples_indices = topk_relevant_triples_indices.tolist()[0]
        topk_relevant_triples_scores = topk_relevant_triples_scores.tolist()[0]
        
        # return topk_relevant_triples_indices, topk_relevant_triples_scores
        filtered_triples = [triples[i] for i in topk_relevant_triples_indices]
        return filtered_triples, topk_relevant_triples_scores
    
    def get_documents_from_triples(self, triples: List[Dict[str, Union[str, List]]], documents: List[Dict[str, Union[str, List]]]) -> List[Dict[str, Union[str, List]]]:
        docids = [triple["reference"][0] for triple in triples]
        docid2document = {document["id"]: document for document in documents}
        documents = [docid2document[docid] for docid in docids]
        # additionally add top-1 retrieved document 
        if documents[0]["id"] not in docids:
            documents.append(documents[0])
        return documents 
    
    def get_context(self, document: Dict[str, Union[str, List]]) -> str:
        text = document.get("text", None)
        if not text:
            text = " ".join([s.strip() for s in document.get("sentences", [])])
            # truncate text (added after 2025.06.24)
            if len(text.split()) > 250:
                text = " ".join(text.split()[:250]) + " ..." 
        return "Title: {}\nText: {}".format(document["title"], text)
    
    def get_context_from_docids(self, docids_to_scores: Dict[str, float]) -> List[str]:

        sorted_docids = [docid for docid, _ in sorted(docids_to_scores.items(), key=lambda x: x[1], reverse=True)]
        sorted_docids = sorted_docids[:self.max_num_context]
        sorted_documents = self.retriever.get_documents(docid_list=sorted_docids)
        context = [self.get_context(document=document) for document in sorted_documents]
        return context
    
    def calculate_query_similarity(self, query1: str, query2: str) -> float:
        """
        计算两个查询的相似度 (Levenshtein + Jaccard)
        
        Args:
            query1: 第一个查询字符串
            query2: 第二个查询字符串
            
        Returns:
            相似度分数 (0-1之间，1表示完全相同)
        """
        # 计算Levenshtein相似度
        levenshtein_similarity = SequenceMatcher(None, query1.lower(), query2.lower()).ratio()
        
        # 计算Jaccard相似度
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if len(words1) == 0 and len(words2) == 0:
            jaccard_similarity = 1.0
        elif len(words1) == 0 or len(words2) == 0:
            jaccard_similarity = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_similarity = intersection / union
        
        return (levenshtein_similarity + jaccard_similarity) / 2.0 
    
    def convert_non_reasoning_model_thought_to_thought(self, thought: str) -> str:

        segments = thought.split("</think>")
        segments = [seg.replace("<think>", "").strip() for seg in segments if seg.strip()]
        # return "\n\n".join(segments) + "\n\n"
        output = "\n\n".join(segments)
        if output:
            output += "\n\n"
        return output 
    
    def get_context_corresponding_to_triples(self, triples: List[Dict[str, Any]]) -> List[str]:
        
        docids = [] 
        for triple in triples:
            docid = triple["reference"][0]
            if docid not in docids:
                docids.append(docid)
        documents = self.retriever.get_documents(docid_list=docids)
        context = [self.get_context(document=document) for document in documents]
        return context
    
    @torch.no_grad()
    def forward_v2_for_reasoning_model(self, question: str, is_multihop_qa: bool = False, save_intermediate_results: bool = False, use_thought: bool = False, is_long_form_qa: bool=False) -> dict:

        if self.use_long_reasoning_model:
            if is_long_form_qa:
                prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT_LONGFORMQA.format(question=question)], use_chat=True)[0]
            else:
                if isinstance(self.reasoning_model, OpenRouterLLMWrapper):
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT_OPENROUTER.format(question=question)], use_chat=True)[0]
                else:
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT.format(question=question)], use_chat=True)[0]
        else:
            if is_long_form_qa:
                if isinstance(self.reasoning_model, OpenRouterLLMWrapper):
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA_OPENROUTER.format(question=question)], use_chat=True)[0]
                else:
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA.format(question=question)], use_chat=True)[0]
            else:
                if isinstance(self.reasoning_model, OpenRouterLLMWrapper):
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT_FOR_NON_REASONING_MODEL_OPENROUTER.format(question=question)], use_chat=True)[0]
                else:
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT_FOR_NON_REASONING_MODEL.format(question=question)], use_chat=True)[0]
        
        results = {
            "question": question, 
            "prompt": prompt,
            "output": "",
            "retrieval_count": 0,
            "latent_reasoning_chains": [],
            "retrieval_queries": [], 
            "retrieval_results": {},
            "selected_triples": [], 
            "context": [],
            "finished": False,
            # save the raw LLM input-output pairs 
            "determine_retrieval_pairs": [], 
            "retrieval_query_pairs": [], 
            "identify_relevant_triples_pairs": [], 
        }

        docids_to_scores = {} 
        step = 0  # step of reasoning  
        request_times = 0  # request times of retrieval 

        retrieved_documents = self.retriever.forward(queries=question, topk=self.topk)
        retrieved_documents_with_kgs = self.kg_generator(retrieved_documents)
        retrieved_triples = self.get_candidate_triples_from_documents(retrieved_documents_with_kgs)
        retrieved_triples, retrieved_triples_scores = self.filter_candidate_triples(question=question, query="", triples=retrieved_triples)
        for triple, score in zip(retrieved_triples, retrieved_triples_scores):
            docid = triple["reference"][0] 
            docids_to_scores[docid] = max(docids_to_scores.get(docid, 0), score)
        context = self.get_context_from_docids(docids_to_scores=docids_to_scores)
        results["context"] = context

        # from pdb import set_trace; set_trace()
        while True:

            finished = results["finished"]
            if finished or step >= self.max_reasoning_steps:
                break 
            
            try:
                generated_text = self.generate(
                    prompts=[results["prompt"].format(context="\n\n".join(results["context"]))],
                    config=self.reasoning_model_config.get_hparams(), 
                    stop_words=["\n\n", self.reasoning_model.tokenizer.eos_token] if self.use_long_reasoning_model \
                        else ["</think>", self.reasoning_model.tokenizer.eos_token]
                )[0]
            except:
                # For API Model
                generated_text = self.generate(
                    prompts=[results["prompt"].format(context="\n\n".join(results["context"]))],
                    config=self.reasoning_model_config.get_hparams(), 
                    stop_words=["\n\n"] if self.use_long_reasoning_model else ["</think>"]
                )[0]
                generated_text_rstrip = generated_text.rstrip()
                if self.use_long_reasoning_model:
                    if generated_text_rstrip:
                        generated_text = generated_text.rstrip() + "\n\n"
                else:
                    if not generated_text_rstrip.endswith("</answer>") and not generated_text_rstrip.endswith("</think>"): 
                        generated_text = generated_text.rstrip() + " </think>\n"
            generated_text = escape_braces(generated_text)
            # from pdb import set_trace; set_trace()

            # if "</think>" in generated_text:
            if (
                self.use_long_reasoning_model and "</think>" in generated_text
            ) or (
                not self.use_long_reasoning_model and \
                    ("</answer>" in generated_text or "<answer>" in generated_text)
            ):
                
                if isinstance(self.reasoning_model, OpenRouterLLMWrapper):
                    if self.use_long_reasoning_model:
                        raise NotImplementedError
                    else:
                        if "</answer>" in generated_text:
                            results["prompt"] += generated_text 
                            results["output"] += generated_text 
                            break 
                
                # reasoning is finished, generate the answer
                results["prompt"] += generated_text 
                results["output"] += generated_text 
                answer_text = self.generate(
                    prompts=[results["prompt"].format(context="\n\n".join(results["context"]))], 
                    config=self.reasoning_model_config.get_hparams()
                )[0]
                results["prompt"] += answer_text 
                results["output"] += answer_text 
                results["finished"] = True 
                break 

            # if self.use_long_reasoning_model and isinstance(self.reasoning_model, OpenRouterLLMWrapper) and r"\boxed{" in generated_text:
            #     results["prompt"] += generated_text 
            #     results["output"] += generated_text 
            #     break
            
            existing_prompt = deepcopy(results["prompt"])
            existing_thought = deepcopy(results["output"])
            current_thought = deepcopy(generated_text)
            results["prompt"] += generated_text
            results["output"] += generated_text
            step += 1 

            print("\n##>>>>>>>>>>>>>>>>>> Current Generated Output: <<<<<<<<<<<<<<<<<<<<<\n", results["output"])
            if results["retrieval_count"] >= self.max_retrieval_count or request_times >= self.max_request_times:
                continue

            # determine whether retrieval is required
            is_retrieval_required, *others = self.policy_model.is_retrieval_required_with_context(
                question=question,
                thought=existing_thought if self.use_long_reasoning_model else self.convert_non_reasoning_model_thought_to_thought(existing_thought), 
                current_thought=current_thought if self.use_long_reasoning_model else self.convert_non_reasoning_model_thought_to_thought(current_thought), 
                context=results["context"], 
                return_raw_response=save_intermediate_results
            )
            request_times += 1 
            if save_intermediate_results:
                thought_for_save = existing_thought if self.use_long_reasoning_model else self.convert_non_reasoning_model_thought_to_thought(existing_thought)
                current_thought_for_save = current_thought if self.use_long_reasoning_model else self.convert_non_reasoning_model_thought_to_thought(current_thought)
                results["determine_retrieval_pairs"].append({"question": question, "thought": thought_for_save, "current_thought": current_thought_for_save, "context": results["context"], "raw_response": others[0], "is_retrieval_required": is_retrieval_required})
            
            if is_retrieval_required:
                # obtain retrieval query
                if use_thought:
                    query_info, *others = self.policy_model.formulate_retrieval_query_with_thought(question=question, thought=results["output"], return_raw_response=save_intermediate_results)
                else:
                    # query_info, *others = self.policy_model.formulate_retrieval_query(question=question, thought=results["output"], return_raw_response=save_intermediate_results)
                    query_info, *others = self.policy_model.formulate_retrieval_query(
                        question=question, 
                        thought=results["output"] if self.use_long_reasoning_model else self.convert_non_reasoning_model_thought_to_thought(results["output"]), 
                        return_raw_response=save_intermediate_results
                    ) 
                reasoning_chain: str = query_info["reasoning_chain"]
                retrieval_query: str = query_info["query"]
                if save_intermediate_results:
                    thought_for_save = results["output"] if self.use_long_reasoning_model else self.convert_non_reasoning_model_thought_to_thought(results["output"])
                    results["retrieval_query_pairs"].append({"question": question, "thought": thought_for_save, "raw_response": others[0], "reasoning_chain": reasoning_chain, "retrieval_query": retrieval_query})
                if retrieval_query is None:
                    continue

                # 检查是否已有相似的查询
                should_skip_retrieval = False
                if retrieval_query in results["retrieval_results"]:
                    # if the query already eixsts, skip current retrieval 
                    should_skip_retrieval = True
                else:
                    # 检查与已有查询的相似度
                    for existing_query in results["retrieval_results"].keys():
                        similarity = self.calculate_query_similarity(retrieval_query, existing_query)
                        if similarity > 0.8:
                            should_skip_retrieval = True
                            break
                
                if should_skip_retrieval:
                    continue
                
                retrieved_documents = self.retriever.forward(queries=retrieval_query, topk=self.topk)
                # extract & filter triples 
                retrieved_documents_with_kgs = self.kg_generator(retrieved_documents)
                retrieved_triples = self.get_candidate_triples_from_documents(retrieved_documents_with_kgs)
                if not retrieved_triples:
                    continue
                retrieved_triples, retrieved_triples_scores = self.filter_candidate_triples(question=question, query=retrieval_query, triples=retrieved_triples)
                if use_thought:
                    relevant_triples, *others = self.policy_model.identify_relevant_triples_with_thought(question=question, thought=results["output"], reasoning_chain=reasoning_chain, query=retrieval_query, triples=retrieved_triples, return_raw_response=save_intermediate_results)
                else:
                    relevant_triples, *others = self.policy_model.identify_relevant_triples(question=question, reasoning_chain=reasoning_chain, query=retrieval_query, triples=retrieved_triples, return_raw_response=save_intermediate_results)
                if save_intermediate_results:
                    results["identify_relevant_triples_pairs"].append(
                        {
                            "question": question, 
                            "reasoning_chain": reasoning_chain, 
                            "query": retrieval_query, 
                            "triples": retrieved_triples, 
                            # "context": retrieved_triples_corresponding_context, # 如果用版本1的话这里需要注释掉
                            "raw_response": others[0], 
                            "relevant_triples": relevant_triples
                        }
                    )
                if not relevant_triples:
                    continue  

                triple_text_to_index = {triple["text"]: index for index, triple in enumerate(retrieved_triples)}
                for relevant_triple in relevant_triples:
                    idx = triple_text_to_index[relevant_triple["text"]]
                    retrieved_triples_scores[idx] += 0.5 
                for triple, score in zip(retrieved_triples, retrieved_triples_scores):
                    docid = triple["reference"][0]
                    docids_to_scores[docid] = max(docids_to_scores.get(docid, 0), score)
                
                """
                # only add the relevant documents 
                current_step_docids_to_scores = {doc["id"]: doc["score"] for doc in retrieved_documents}
                for relevant_triple in relevant_triples:
                    docid = relevant_triple["reference"][0]
                    docids_to_scores[docid] = max(docids_to_scores.get(docid, 0), current_step_docids_to_scores[docid])
                """
                
                results["latent_reasoning_chains"].append(reasoning_chain)
                results["retrieval_queries"].append(retrieval_query)
                results["retrieval_results"][retrieval_query] = retrieved_documents
                results["selected_triples"].append(relevant_triples)
                results["retrieval_count"] += 1 
                results["context"] = self.get_context_from_docids(docids_to_scores=docids_to_scores)

        if is_long_form_qa:
            if (
                self.use_long_reasoning_model and (
                    '</think>' not in results["output"] # or r"\boxed{" not in results["output"]
                )
            ) or (
                not self.use_long_reasoning_model and (
                    '</answer>' not in results["output"] or '<answer>' not in results["output"]
                )
            ): 
                if self.use_long_reasoning_model:
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT_LONGFORMQA.format(question=question)], use_chat=True)[0]
                else:
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA.format(question=question)], use_chat=True)[0]
                
                generated_text = self.generate(
                    prompts = [prompt.format(context="\n\n".join(results["context"]))], 
                    config=self.reasoning_model_config.get_hparams()
                )[0]
                results["prompt"] = prompt
                results["output"] = generated_text
                results["finished"] = True
        
        else:
            if (
                self.use_long_reasoning_model and (
                    '</think>' not in results["output"] or r"\boxed{" not in results["output"]
                )
            ) or (
                not self.use_long_reasoning_model and (
                    '</answer>' not in results["output"] or '<answer>' not in results["output"]
                )
            ):
                if self.use_long_reasoning_model:
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT.format(question=question)], use_chat=True)[0]
                else:
                    prompt = self.reasoning_model.prompt(inputs=[ANSWER_PROMPT_FOR_NON_REASONING_MODEL.format(question=question)], use_chat=True)[0]
                
                generated_text = self.generate(
                    prompts = [prompt.format(context="\n\n".join(results["context"]))], 
                    config=self.reasoning_model_config.get_hparams()
                )[0]
                results["prompt"] = prompt
                results["output"] = generated_text
                results["finished"] = True
        
        print(results["output"])
        return results
