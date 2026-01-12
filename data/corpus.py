import os 
import random
import pickle
import string 
import logging
import numpy as np 
from tqdm import tqdm 
from copy import deepcopy
from collections import defaultdict
from typing import Dict

import torch
from torch.utils.data import Dataset

import sys
from setup.setup import * 
sys.path.append(COMMON_FOLDER)
from my_utils import load_json, save_json, load_id2json, convert_to_unicode, contain_answers, remove_bracket
from my_evaluation import ems 

logger = logging.getLogger(__name__)


def load_psg_data(path):

    punctuation = set(string.punctuation)
    def remove_punctuation(text):
        if text[0] in punctuation:
            text = text[1:]
        if text[-1] in punctuation:
            text = text[:-1]
        text = text.replace("\"\"", "\"")
        return text

    data = []
    print("loading wikipedia passage data ... ")
    with open(path, encoding="utf-8", mode="r") as fin:
        for line in fin:
            pieces = line.strip().split("\t")
            data.append(
                {
                    "id": str(pieces[0]), 
                    "title": remove_punctuation(pieces[2]), 
                    "text": remove_punctuation(pieces[1])
                }
            )
    data = data[1:] # 第一行是"id \t text \t title 

    return data 

class Corpus(Dataset):

    def __init__(self, title_prefix='title:', passage_prefix='context:', **kwargs):

        self.data = self.load_corpus_data()
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.kwargs = kwargs
        self.passage_format = "{title_prefix} {title}, {passage_prefix} {passage}"

        passage_id_name = self.get_passage_id_name()
        self.index_to_passage_id = {i: example[passage_id_name] for i, example in enumerate(self.data)}
        self.passage_id_to_index = {example[passage_id_name]: i for i, example in enumerate(self.data)}
    
    def load_corpus_data(self):
        raise NotImplementedError("load_corpus_data is not Implemented!")
    
    def get_passage_id_name(self):
        for key in self.data[0].keys():
            if "id" in key:
                return key

    def __len__(self):
        return len(self.data)
    
    def get_document(self, passage_id):
        return self.data[self.passage_id_to_index[passage_id]]
    
    def doc_to_str(self, doc):
        raise NotImplementedError(f"doc_to_text for {type(self).__name__} is not implemented!")

    def get_document_text(self, passage_id):
        doc = self.get_document(passage_id=passage_id)
        return self.doc_to_str(doc=doc)
    
    def get_document_item(self, passage_id):
        return self.__getitem__(self.passage_id_to_index[passage_id])


class Wikipedia(Corpus):

    def __init__(self, title_prefix='title:', passage_prefix='context:', **kwargs):
        
        self.corpus_path = "/nfs/common/data/wikipedia/psgs_w100.tsv"
        # self.corpus_path = "/nfs/common/data/wikipedia/psgs_w100_head100.tsv"
        super().__init__(title_prefix, passage_prefix, **kwargs)
        print(f"Successfully load {len(self.data)} examples from {self.corpus_path}!")
    
    def load_corpus_data(self):
        return load_psg_data(self.corpus_path)
    
    def doc_to_str(self, doc):

        text = self.passage_format.format(
            title_prefix=self.title_prefix, 
            title = doc["title"], 
            passage_prefix = self.passage_prefix, 
            passage = doc["text"]
        ).strip()

        return text

    def __getitem__(self, index):

        """
        self.data的格式:
        {
            "id": str, 
            "title": str, 
            "text": str, 
        }
        """
        example = self.data[index]
        passage = self.passage_format.format(\
            title_prefix=self.title_prefix, title=example["title"], \
                passage_prefix=self.passage_prefix, passage=example["text"]
            ).strip()
        
        results = {
            "index": index, 
            "passage_id": example["id"], 
            "passage": passage
        }
        return results


class HotPotQA(Corpus):

    def __init__(self, title_prefix='title:', passage_prefix='context:', **kwargs):

        self.corpus_path = "/nfs/common/data/hotpotqa/open_domain_data/corpus.json"
        # self.corpus_path = "/nfs/common/data/hipporag/hotpotqa/corpus.json"
        super().__init__(title_prefix, passage_prefix, **kwargs)
        print(f"Successfully load {len(self.data)} examples from {self.corpus_path}!")
    
    def load_corpus_data(self):
        return load_json(self.corpus_path)
    
    def doc_to_str(self, doc):
        
        text = self.passage_format.format(
            title_prefix=self.title_prefix, 
            title=doc["title"],
            passage_prefix=self.passage_prefix, 
            passage=" ".join([sent.strip() for sent in doc["sentences"]])
        ).strip() 

        return text
    
    def __getitem__(self, index):

        """
        self.data的格式:
        {
            "id": str, 
            "title": str, 
            "sentences": str, 
        }
        """
        example = self.data[index]
        passage = self.passage_format.format(
            title_prefix=self.title_prefix, 
            title=example["title"], 
            passage_prefix=self.passage_prefix, 
            passage=" ".join([sent.strip() for sent in example["sentences"]])
        ).strip()
        
        results = {
            "index": index, 
            "passage_id": example["id"], 
            "passage": passage
        }
        return results


class HotPotQAHippoRAG(HotPotQA):

    def __init__(self, title_prefix='title:', passage_prefix='context:', **kwargs):

        # self.corpus_path = "/nfs/common/data/hotpotqa/open_domain_data/corpus.json"
        self.corpus_path = "/nfs/common/data/hipporag/hotpotqa/corpus.json"
        Corpus.__init__(self, title_prefix, passage_prefix, **kwargs)
        print(f"Successfully load {len(self.data)} examples from {self.corpus_path}!")


class WikiMultiHopQA(Corpus):

    def __init__(self, title_prefix='title:', passage_prefix='context:', **kwargs):

        self.corpus_path = "/nfs/common/data/2wikimultihopqa/open_domain_data/corpus.json"
        # self.corpus_path = "/nfs/common/data/hipporag/2wikimultihopqa/corpus.json"
        super().__init__(title_prefix, passage_prefix, **kwargs)
        print(f"Successfully load {len(self.data)} examples from {self.corpus_path}!")
    
    def load_corpus_data(self):
        return load_json(self.corpus_path)
    
    def doc_to_str(self, doc):

        text = self.passage_format.format(
            title_prefix=self.title_prefix, 
            title=doc["title"], 
            passage_prefix=self.passage_prefix, 
            passage=" ".join([sent.strip() for sent in doc["sentences"]])
        ).strip()

        return text
    
    def __getitem__(self, index):

        """
        self.data的格式:
        {
            "id": str, 
            "title": str, 
            "sentences": str, 
        }
        """
        example = self.data[index]
        passage = self.passage_format.format(
            title_prefix=self.title_prefix, 
            title=example["title"], 
            passage_prefix=self.passage_prefix, 
            passage=" ".join([sent.strip() for sent in example["sentences"]])
        ).strip()
        
        results = {
            "index": index, 
            "passage_id": example["id"], 
            "passage": passage
        }
        return results


class WikiMultiHopQAHippoRAG(WikiMultiHopQA):

    def __init__(self, title_prefix='title:', passage_prefix='context:', **kwargs):

        # self.corpus_path = "/nfs/common/data/2wikimultihopqa/open_domain_data/corpus.json"
        self.corpus_path = "/nfs/common/data/hipporag/2wikimultihopqa/corpus.json"
        Corpus.__init__(self, title_prefix, passage_prefix, **kwargs)
        print(f"Successfully load {len(self.data)} examples from {self.corpus_path}!")
    

class MuSiQue(Corpus):

    def __init__(self, title_prefix='title:', passage_prefix='context:', **kwargs):

        self.corpus_path = "/nfs/common/data/musique/open_domain_data/corpus.json"
        # self.corpus_path = "/nfs/common/data/hipporag/musique/corpus.json" 
        super().__init__(title_prefix, passage_prefix, **kwargs)
        print(f"Successfully load {len(self.data)} examples from {self.corpus_path}!")
    
    def load_corpus_data(self):
        return load_json(self.corpus_path)
    
    def doc_to_str(self, doc):
        
        text = self.passage_format.format(
            title_prefix=self.title_prefix, 
            title=doc["title"], 
            passage_prefix=self.passage_prefix, 
            passage=" ".join([sent.strip() for sent in doc["sentences"]])
        ).strip()

        return text
    
    def __getitem__(self, index):

        """
        self.data的格式:
        {
            "id": str, 
            "title": str, 
            "sentences": str, 
        }
        """
        example = self.data[index]
        passage = self.passage_format.format(
            title_prefix=self.title_prefix, 
            title=example["title"], 
            passage_prefix=self.passage_prefix, 
            passage=" ".join([sent.strip() for sent in example["sentences"]])
        ).strip()
        
        results = {
            "index": index, 
            "passage_id": example["id"], 
            "passage": passage
        }
        return results


class MuSiQueHippoRAG(MuSiQue):

    def __init__(self, title_prefix='title:', passage_prefix='context:', **kwargs):

        # self.corpus_path = "/nfs/common/data/musique/open_domain_data/corpus.json"
        self.corpus_path = "/nfs/common/data/hipporag/musique/corpus.json" 
        Corpus.__init__(self, title_prefix, passage_prefix, **kwargs)
        print(f"Successfully load {len(self.data)} examples from {self.corpus_path}!")


class KGTripleCorpus(Corpus):

    def __init__(self, corpus_path, title_prefix="", passage_prefix="", **kwargs):
        self.corpus_path = corpus_path
        super().__init__(title_prefix=title_prefix, passage_prefix=passage_prefix, **kwargs)
        print(f"Successfully load {len(self.data)} KG Triples from {self.corpus_path}!")

    def load_corpus_data(self):
        """
        raw_data format: 
        {
            "passage id": {
                "id": str, 
                "title": str, 
                "sentences": List[str], 
                "triples": [{"text": str, "sentence": int}]
            }
        }
        """
        raw_data: Dict[str, dict] = pickle.load(open(self.corpus_path, "rb"))
        data = []
        triple_index = 0 
        self.document_corpus = {}

        for passage_id, item in raw_data.items():

            title = item["title"]
            for triple in item["triples"]:
                data.append(
                    {
                        "triple_id": str(triple_index), 
                        "title": title, 
                        "text": triple["text"], 
                        "reference": [passage_id, triple["sentence"]]
                    }
                )
                triple_index += 1 
            
            self.document_corpus[passage_id] = {
                "id": passage_id, 
                "title": title, 
                "sentences": item["sentences"]
            }
        
        return data
    
    def get_raw_document(self, passage_id):
        return self.document_corpus[passage_id]

    def get_passage_id_name(self):
        return "triple_id"
    
    def doc_to_str(self, doc: dict):
        return doc["text"]
    
    def __getitem__(self, index):
        example = self.data[index]
        passage = self.doc_to_str(example)
        results = {
            "index": index, 
            "passage_id": example["triple_id"], 
            "passage": passage
        }
        return results
    

class HotPotQAKGTriples(KGTripleCorpus):
    def __init__(self, **kwargs):
        corpus_path = "/nfs/common/data/hotpotqa/open_domain_data/cached_kg_triples.pkl"
        super().__init__(corpus_path=corpus_path, **kwargs)


class WikiMultiHopQAKGTriples(KGTripleCorpus):

    def __init__(self, **kwargs):
        corpus_path = "/nfs/common/data/2wikimultihopqa/open_domain_data/cached_kg_triples.pkl"
        super().__init__(corpus_path=corpus_path, **kwargs)


class MuSiQueKGTriples(KGTripleCorpus):

    def __init__(self, **kwargs):
        corpus_path = "/nfs/common/data/musique/open_domain_data/cached_kg_triples.pkl"
        super().__init__(corpus_path=corpus_path, **kwargs)