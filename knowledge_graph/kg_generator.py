import os
import re 
import time 
import pickle
import argparse
import numpy as np
from copy import deepcopy 
from tqdm import trange, tqdm
from nltk.tokenize import sent_tokenize
from typing import Union, Optional, Tuple, List, Dict

import torch 
import torch.nn as nn
from torch import Tensor

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel 
from transformers import LlamaForCausalLM, Qwen2ForCausalLM
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from retriever.e5 import get_e5_embeddings_for_document, get_e5_embeddings_for_query

from utils.utils import * 
from setup.setup import HF_TOKEN


class KGGenerator(nn.Module):

    def __init__(self, tokenizer, generator, max_length=4096, max_new_tokens=512, examplar_type="hotpotqa", num_examplars=5, adaptive_examplars=True, adaptive_examplars_retriever="e5", batch_size=4, verbose=False, **kwargs):
        
        super().__init__()
        assert examplar_type in ["hotpotqa", "2wikimultihopqa", "musique", "nq", "tqa", "webqa", "bamboogle", "wikipedia", "asqa"] 

        self.tokenizer = tokenizer
        self.generator = generator
        assert isinstance(generator, (LlamaForCausalLM, Qwen2ForCausalLM)) # currently only support using LLaMA3 or Gemma2 as the generator
        self.device = self.generator.device 
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.num_examplars = num_examplars
        self.adaptive_examplars = adaptive_examplars
        self.adaptive_examplars_retriever = adaptive_examplars_retriever
        self.examplars = self.load_examplars(examplar_type)
        if self.adaptive_examplars:
            self.examplars_embeddings = self.calculate_examplars_embeddings()
        self.batch_size = batch_size
        self.verbose = verbose
        self.cached_kg_triples = None 
        self.task_instruction = "You are a knowledge graph constructor tasked with extracting knowledge triples in the form of <head entity; relation; tail entity> from a document. "\
            "Each triple denotes a specific relationship between entities or an event. The head entity and tail entity can be the provided title or phrases in the text. "\
                "If multiple tail entities share the same relation with a head entity, aggregate these tail entities using commas. "\
                    "Format your output in the form of <head entity; relation; tail entity>."
        self.kwargs = kwargs
    
    def load_examplars(self, examplar_type):

        print(f"loading {examplar_type} examplars for KGGenerator ... ")
        if examplar_type in ["hotpotqa", "hotpotqa_hipporag"]:
        # if examplar_type in ["hotpotqa", "bamboogle"]:
            from prompts.kg_construction.hotpotqa_demonstrations import generate_knowledge_triples_hotpotqa_examplars
            examplars = generate_knowledge_triples_hotpotqa_examplars
        elif examplar_type in ["2wikimultihopqa", "2wikimultihopqa_hipporag"]:
            from prompts.kg_construction.wikimultihopqa_demonstrations import generate_knowledge_triples_2wikimultihopqa_examplars
            examplars = generate_knowledge_triples_2wikimultihopqa_examplars
        elif examplar_type in ["musique", "musique_hipporag"]:
            from prompts.kg_construction.musique_demonstrations import generate_knowledge_triples_musique_examplars
            examplars =  generate_knowledge_triples_musique_examplars
        elif examplar_type in ["wikipedia", "nq", "tqa", "webqa", "bamboogle", "asqa"]:
        # elif examplar_type in ["wikipedia"]:
            from prompts.kg_construction.wikipedia_demonstrations import generate_knowledge_triples_wikipedia_examplars
            examplars = generate_knowledge_triples_wikipedia_examplars
        else:
            raise KeyError(f"{examplar_type} is not a supported examplar type!")
        
        return examplars
    
    def get_texts_from_documents(self, documents: Union[Dict[str, str], List[Dict[str, str]]]) -> List[str]:

        is_list = isinstance(documents, list)
        if not is_list:
            documents = [documents]

        texts = [] 
        for doc in documents:
            title = doc["title"]
            text = doc.get("text", None)
            if text is None:
                text = " ".join([sent.strip() for sent in doc["sentences"]])
            texts.append("Title: {}\nText: {}".format(title, text))
        
        if not is_list:
            texts = texts[0]

        return texts
    
    def calculate_examplars_embeddings(self):

        texts = self.get_texts_from_documents(self.examplars)
        if self.adaptive_examplars_retriever == "e5":
            print("calculating embeddings for demonstrations ...")
            with torch.no_grad():
                examplars_embeddings = get_e5_embeddings_for_document(texts, max_length=256)
        else:
            raise KeyError(f"{self.adaptive_examplars_retriever} is not a supported retriever!")
        return examplars_embeddings
    
    def rank_examplars(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:

        """
        Input: [{"title": str, "text": str / "sentences": str}]
        Output: [{"title": str, "text": str / "sentences": str, "ranked_examplars_indices": [str]}]
        """
        texts = self.get_texts_from_documents(documents)
        if self.adaptive_examplars_retriever == "e5":
            with torch.no_grad():
                texts_embeddings = get_e5_embeddings_for_document(texts, max_length=256)
        else:
            raise KeyError(f"{self.adaptive_examplars_retriever} is not a supported retriever!")
        
        similarities = torch.matmul(texts_embeddings, self.examplars_embeddings.T)
        indices = torch.argsort(similarities, dim=1, descending=True).tolist()
        for doc, ranked_examplars_indices in zip(documents, indices):
            doc["ranked_examplars_indices"] = ranked_examplars_indices
        return documents
    
    def load_cached_kg_triples(self, paths: Union[str, List[str]]):
        
        if isinstance(paths, str):
            paths = [paths]
        
        if self.cached_kg_triples is None:
            print("Initializing a new KG triples cache ...")
            self.cached_kg_triples = {}
        
        for path in paths:
            if os.path.exists(path):
                print(f"loading cached KG triples from {path} ...")
                self.cached_kg_triples.update(pickle.load(open(path, "rb")))
    
    def save_cached_kg_triples(self, path):
        
        if self.cached_kg_triples is not None:
            print(f"saving cached KG triples to {path} ...")
            pickle.dump(self.cached_kg_triples, open(path, "wb"))

    def get_documents_inputs(self, documents: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """
        Input: [{"title": str, "text": str / "sentences": str, (optional: "ranked_examplars_indices": [int])}]
        Output: instructions: [str], inputs: [str]
        """
        def vary_num_examplars_based_on_context_window(examplars, doc):
            final_examplars = None
            while len(examplars) > 0:
                for num in range(len(examplars), 0, -1):
                    candidate_examplars = examplars[:num]
                    candidate_prompt = self.task_instruction + "\n\n" + "\n\n".join(candidate_examplars) + "\n\n" + self.get_texts_from_documents(doc)
                    candidate_prompt_tokens = self.tokenizer.encode(candidate_prompt)
                    if len(candidate_prompt_tokens) <= self.max_length:
                        final_examplars = examplars[:num]
                        break
                if final_examplars is None:
                    examplars = examplars[1:]
                else:
                    break
            if final_examplars is None:
                final_examplars = [] 
            return final_examplars

        instructions, inputs = [], []
        for doc in documents:
            ranked_examplars_indices = doc.get("ranked_examplars_indices", None)
            if ranked_examplars_indices is None:
                ranked_examplars_indices = list(range(len(self.examplars)))
            doc_specific_examplars = [self.examplars[idx] for idx in ranked_examplars_indices[:self.num_examplars]]
            doc_specific_examplars = [
                "{}\nKnowledge Triples: {}".format(
                    self.get_texts_from_documents(example), 
                    example["triples"]
                )
                for example in doc_specific_examplars
            ]
            doc_specific_examplars = vary_num_examplars_based_on_context_window(doc_specific_examplars, doc)

            instructions.append(self.task_instruction+"\n\n"+"\n\n".join(doc_specific_examplars))
            # inputs.append(
            #     "Extract knowledge triples from the following document according to the task instruction.\n\n" + \
            #     self.get_texts_from_documents(doc)
            # )
            inputs.append(self.get_texts_from_documents(doc))
        return instructions, inputs

    def get_documents_prompts_chat_format(self, instructions: List[str], inputs: List[str]):
        """
        Input: instruction: [str], inputs: [str]
        """
        prompts = [] 
        for instruction, input in zip(instructions, inputs):
            if isinstance(self.generator, (LlamaForCausalLM, Qwen2ForCausalLM)):
                prompts.append(
                    [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": input}
                    ]
                )
            # elif isinstance(self.generator, Gemma2ForCausalLM):
            #     # content = instruction + "\n\nPlease help me extract knowledge triples for the following document.\n\n" + input
            #     content = instruction + "\n\n" + input
            #     prompts.append(
            #         [
            #             {"role": "user", "content": content}
            #         ]
            #     )
            else:
                raise NotImplemented(f"chat format for {type(self.generator)} is not implemented yet!")
        return prompts
    
    def tokenizer_encode_chat_format(self, prompts: List[List[Dict[str, str]]]):
        texts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        batch_dict = self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs
    
    def generator_generate(self, inputs):
        inputs = to_device(inputs, self.device)
        generated_token_ids = self.generator.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature=1.0, do_sample=False)
        return generated_token_ids

    def parse_triples_text(self, triples_text: str):
        results = [] 
        for one_triple_text in re.findall(r'<([^>]*)>', triples_text):
            if "head entity" in one_triple_text or "tail entity" in one_triple_text:
                continue
            results.append("<{}>".format(one_triple_text.strip()))
        return results
    
    def find_sentence_for_one_triple(self, doc: Dict[str, str], triple: str):

        def get_common_word_count(substring, text): 
            return np.sum([word in text for word in substring.split()])
        
        sentences = doc.get("sentences", None)
        if sentences is None:
            sentences = sent_tokenize(doc["text"])
        common_word_counts = [get_common_word_count(triple, sentence) for sentence in sentences]
        index = int(np.argmax(common_word_counts))
        return index

    def parse_generator_outputs(self, documents: List[Dict[str, str]], generator_outputs: List[str]):
        """
        Input: documents: [{"title": str, "text": str / "sentences": str}], generator_outputs: ["xxx<xxx>\n<xx>", ...]
        Output: [{"title": str, "text": str / "sentences": str, "triples": [{"text": str, "sentence": int}]}]
        """
        for doc, one_doc_generator_output in zip(documents, generator_outputs):
            triples = [] 
            triples_texts = self.parse_triples_text(one_doc_generator_output)
            for one_triple in triples_texts:
                sentence = self.find_sentence_for_one_triple(doc, one_triple)
                triples.append({"text": one_triple, "sentence": sentence})
            doc["triples"] = triples 
        
        return documents

    def generate_kg_triples_wo_cache(self, documents: Union[Dict[str, str], List[Dict[str, str]]], cache_path: str = None, cache_freq: int = 2000):
        
        """
        Input: {"title": str, "text": str / "sentences": str} or [{"title": str, "text": str / "sentences": str}]
        Output: {"title": str, "text": str / "sentences": str, "triples": ["text": str, "sentence": int]}, or its list version 
        """
        is_list = isinstance(documents, list)
        if not is_list:
            documents = [documents]
        if self.adaptive_examplars:
            documents = self.rank_examplars(documents)

        if self.verbose:
            progress_bar = trange((len(documents)-1) // self.batch_size + 1, desc="Generating Knowledge Triples")

        generated_contents = []
        num_processed_docs = 0
        for i in range((len(documents)-1) // self.batch_size + 1):
            batch_document = documents[i*self.batch_size: (i+1)*self.batch_size]
            batch_instructions, batch_inputs = self.get_documents_inputs(batch_document)
            batch_prompts = self.get_documents_prompts_chat_format(batch_instructions, batch_inputs)
            batch_generator_inputs = self.tokenizer_encode_chat_format(batch_prompts)
            batch_generated_token_ids = self.generator_generate(batch_generator_inputs)
            batch_input_ids = batch_generator_inputs["input_ids"]
            batch_generated_token_ids = batch_generated_token_ids[:, batch_input_ids.shape[1]:]
            batch_generated_texts = self.tokenizer.batch_decode(batch_generated_token_ids, skip_special_tokens=True)
            generated_contents.extend(batch_generated_texts)
            num_processed_docs += len(batch_document)
            if self.cached_kg_triples is not None and cache_path is not None and num_processed_docs % cache_freq == 0:
                print(f"saving intermediate {num_processed_docs} cached KG triples to {cache_path} ...")
                processed_documents = documents[:num_processed_docs]
                processed_documents_generated_contents = generated_contents[:num_processed_docs]
                processed_documents_with_kgs = self.parse_generator_outputs(processed_documents, processed_documents_generated_contents)
                tmp_cache_kg_triples = {doc["id"]: doc for doc in processed_documents_with_kgs}
                pickle.dump(tmp_cache_kg_triples, open(cache_path, "wb"))
            if self.verbose:
                progress_bar.update(1)
        
        # parser model outputs 
        documents_with_triples = self.parse_generator_outputs(documents, generated_contents)
        if not is_list:
            documents_with_triples = documents_with_triples[0]

        return documents_with_triples
    
    def generate_kg_triples_with_cache(self, documents: Union[Dict[str, str], List[Dict[str, str]]], cache_path: str = None, cache_freq: int = 2000):

        assert self.cached_kg_triples is not None # muse use "load_cached_kg_triples(path)" function to load or initialize KG cache!

        is_list = isinstance(documents, list)
        if not is_list:
            documents = [documents]

        all_docids = [doc["id"] for doc in documents]
        docs_wo_cached_kg_triples = [doc for docid, doc in zip(all_docids, documents) if docid not in self.cached_kg_triples]
        docs_wo_cached_kg_triples = deepcopy(docs_wo_cached_kg_triples)
        if len(docs_wo_cached_kg_triples) > 0:
            docs_with_kgs = self.generate_kg_triples_wo_cache(docs_wo_cached_kg_triples, cache_path=cache_path, cache_freq=cache_freq)
        else:
            docs_with_kgs = []
        # update cache 
        self.cached_kg_triples.update({doc["id"]: doc for doc in docs_with_kgs})
        # get kg triples for all input documents 
        all_docs_with_kgs = [self.cached_kg_triples[docid] for docid in all_docids]
        if not is_list:
            all_docs_with_kgs = all_docs_with_kgs[0]
        
        return all_docs_with_kgs
    
    def forward(self, documents: Union[Dict[str, str], List[Dict[str, str]]], cache_path: str = None, cache_freq: int = 2000):

        if self.cached_kg_triples is None:
            return self.generate_kg_triples_wo_cache(documents=documents)
        else:
            return self.generate_kg_triples_with_cache(documents=documents, cache_path=cache_path, cache_freq=cache_freq)


class APIKGGenerator(KGGenerator):

    class _PromptLengthTokenizer:

        def encode(self, text: Optional[str]):
            if text is None:
                return []
            return text.split()

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 4096,
        max_new_tokens: int = 512,
        examplar_type: str = "hotpotqa",
        num_examplars: int = 5,
        adaptive_examplars: bool = True,
        adaptive_examplars_retriever: str = "e5",
        batch_size: int = 4,
        temperature: float = 0.0,
        verbose: bool = False,
        **kwargs,
    ):
        nn.Module.__init__(self)
        assert examplar_type in [
            "hotpotqa",
            "2wikimultihopqa",
            "musique",
            "nq",
            "tqa",
            "webqa",
            "bamboogle",
            "wikipedia",
            "asqa",
            "hotpotqa_hipporag", 
            "2wikimultihopqa_hipporag",
            "musique_hipporag"
        ]

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.num_examplars = num_examplars
        self.adaptive_examplars = adaptive_examplars
        self.adaptive_examplars_retriever = adaptive_examplars_retriever
        self.examplars = self.load_examplars(examplar_type)
        if self.adaptive_examplars:
            self.examplars_embeddings = self.calculate_examplars_embeddings()
        self.batch_size = batch_size
        self.verbose = verbose
        self.cached_kg_triples = None
        self.task_instruction = (
            "You are a knowledge graph constructor tasked with extracting knowledge triples in the form of <head entity; relation; tail entity> from a document. "
            "Each triple denotes a specific relationship between entities or an event. The head entity and tail entity can be the provided title or phrases in the text. "
            "If multiple tail entities share the same relation with a head entity, aggregate these tail entities using commas. "
            "Format your output in the form of <head entity; relation; tail entity>."
        )
        self.kwargs = kwargs
        self.temperature = temperature
        self.model = self._build_openrouter_model()
        self.tokenizer = self._PromptLengthTokenizer() 

    def _build_openrouter_model(self):
        from evoagentx.models import OpenRouterConfig, OpenRouterLLM
        from setup.setup import OPENROUTER_API_KEY

        api_key = OPENROUTER_API_KEY
        if api_key is None:
            raise ValueError("OpenRouter API key is required to initialize APIKGGenerator.")

        config = OpenRouterConfig(
            model=self.model_name_or_path,
            openrouter_key=api_key,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return OpenRouterLLM(config)

    def _generate_with_openrouter(self, instruction: str, user_input: str) -> str:
        response = self.model.generate(system_message=instruction, prompt=user_input)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def generate_kg_triples_wo_cache(
        self,
        documents: Union[Dict[str, str], List[Dict[str, str]]],
        cache_path: str = None,
        cache_freq: int = 2000,
    ):
        is_list = isinstance(documents, list)
        if not is_list:
            documents = [documents]
        if self.adaptive_examplars:
            documents = self.rank_examplars(documents)

        if self.verbose:
            progress_bar = trange((len(documents) - 1) // self.batch_size + 1, desc="Generating Knowledge Triples Using APIKGGenerator")

        generated_contents = []
        num_processed_docs = 0
        num_batches = (len(documents) - 1) // self.batch_size + 1
        for i in range(num_batches):
            batch_document = documents[i * self.batch_size : (i + 1) * self.batch_size]
            batch_instructions, batch_inputs = self.get_documents_inputs(batch_document)

            batch_generated_texts = []
            for instruction, input_text in zip(batch_instructions, batch_inputs):
                batch_generated_texts.append(self._generate_with_openrouter(instruction, input_text))

            generated_contents.extend(batch_generated_texts)
            num_processed_docs += len(batch_document)

            if (
                self.cached_kg_triples is not None
                and cache_path is not None
                and cache_freq > 0
                and num_processed_docs % cache_freq == 0
            ):
                print(f"saving intermediate {num_processed_docs} cached KG triples to {cache_path} ...")
                processed_documents = documents[:num_processed_docs]
                processed_documents_generated_contents = generated_contents[:num_processed_docs]
                processed_documents_with_kgs = self.parse_generator_outputs(
                    processed_documents, processed_documents_generated_contents
                )
                tmp_cache_kg_triples = {doc["id"]: doc for doc in processed_documents_with_kgs}
                pickle.dump(tmp_cache_kg_triples, open(cache_path, "wb"))

            if self.verbose:
                progress_bar.update(1)

        documents_with_triples = self.parse_generator_outputs(documents, generated_contents)
        if not is_list:
            documents_with_triples = documents_with_triples[0]

        return documents_with_triples


tokenizer = None 
model = None 
tokenizer_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# tokenizer_name_or_path = "google/gemma-2-9b-it"
# model_name_or_path = "google/gemma-2-9b-it"
# tokenizer_name_or_path = "Qwen/Qwen2-7B-Instruct"
# model_name_or_path = "Qwen/Qwen2-7B-Instruct"
device = torch.device("cuda")


def get_tokenizer():

    padding_side = "left"
    print(f"loading tokenizer for \"{tokenizer_name_or_path}\" with padding_side: \"{padding_side}\"")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side, token=HF_TOKEN)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Missing padding token, setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_model():
    print(f"loading {model_name_or_path} model in bfloat16 ...")
    model_torch_dtype = torch.bfloat16 
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=model_torch_dtype, token=HF_TOKEN)
    model.to(device)
    model.eval()
    return model 


if __name__ == "__main__":

    tokenizer = get_tokenizer()
    model = get_model()

    generator = KGGenerator(tokenizer, model, examplar_type="wikipedia", num_examplars=5, verbose=True)

    # outputs = generator(
    #     [
    #         {"title": "Wonky (album)", "text": "Wonky is the eight studio album by Orbital, released on their own ACP label (via Warner Music Group/Alternative Distribution Alliance) in the UK on 2 April 2012, and exclusively through iTunes in the USA and Canada on 17 April 2012. The album is their first since the \"Blue Album\" in 2004 and the first since they reformed in 2008. It features vocals from Zola Jesus and Lady Leshurr. The album was taken off of Spotify and iTunes in the United States for unknown reasons. There are some songs you cannot find at all in their original versions, like Beelzedub or Distractions."},
    #     ]
    # )

    import sys
    from setup.setup import COMMON_FOLDER
    sys.path.append(COMMON_FOLDER)
    from my_utils import load_json 

    print("loading corpus .... ")
    # corpus = load_json("/nfs/common/data/2wikimultihopqa/open_domain_data/corpus.json")
    from data.corpus import load_psg_data
    corpus = load_psg_data("/nfs/common/data/wikipedia/psgs_w100.tsv")

    corpus_samples = corpus[:20]
    with torch.no_grad():
        corpus_samples_with_triples = generator(corpus_samples)

    def print_triples(index):
        item = corpus_samples_with_triples[index]
        print(
            "{}\n\nKnowledge Triples:\n{}".format(
                generator.get_texts_from_documents(item),
                "\n".join(["{}\t{}".format(triple["text"], triple["sentence"]) for triple in item["triples"]])
            )
        )
    from pdb import set_trace; set_trace()
    print_triples(0)
