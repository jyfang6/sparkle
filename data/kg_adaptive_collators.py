import torch
import random
from copy import deepcopy
from typing import List, Dict

from data.collators import (
    RetrieverWithPosNegsCollator, 
    EncoderDecoderCollator, 
    encode_question_passages
)

class E5Collator(RetrieverWithPosNegsCollator):

    def __init__(self, tokenizer, query_maxlength, doc_maxlength=None, query_padding="max_sequence", doc_padding ="max_sequence", **kwargs):
        if doc_maxlength is None:
            doc_maxlength = query_maxlength
        super().__init__(tokenizer, query_maxlength, doc_maxlength, query_padding=query_padding, doc_padding=doc_padding, **kwargs)
    
    def encode_query(self, query_list, **kwargs):
        # return self.encode(["query: " + query for query in query_list], self.query_maxlength, self.query_padding, **kwargs)
        query_list = ["query: " + query for query in query_list]
        return super().encode_query(query_list=query_list, **kwargs)
    
    def encode_doc(self, doc_list, **kwargs):
        # return self.encode(["passage: "+ doc for doc in doc_list], self.doc_maxlength, self.doc_padding, **kwargs)
        doc_list = ["passage: "+ doc for doc in doc_list]
        return super().encode_doc(doc_list=doc_list, **kwargs)


class E5CollatorWithReaderScores(E5Collator):

    def __call__(self, batch):
        """
        Input Data Format:
        [
            {
                "index": int, 
                "question": str, the query 
                "positive_passage": str, 
                "positive_passage_score": float, 
                "negative_passages": [str],
                "negative_passages_scores": [float]
            }
        ]
        """
        if isinstance(batch[0], list):
            batch = sum(batch, [])
        
        query_list = [example["question"] for example in batch]
        doc_list, doc_score_list, positive_doc_indices = [], [], [] 
        for example in batch:
            positive_doc_indices.append(len(doc_list))
            doc_list.append(example["positive_passage"])
            doc_score_list.append(example["positive_passage_score"])
            doc_list.extend(example["negative_passages"])
            doc_score_list.extend(example["negative_passages_scores"])
        
        query_args = self.encode_query(query_list)
        doc_args = self.encode_doc(doc_list)
        positive_doc_indices = torch.tensor(positive_doc_indices, dtype=torch.long)
        index = torch.tensor([example["index"] for example in batch], dtype=torch.long)
        doc_scores = torch.tensor(doc_score_list, dtype=torch.float)

        return query_args, doc_args, positive_doc_indices, index, doc_scores


class ContrieverCollator(RetrieverWithPosNegsCollator):

    def __init__(self, tokenizer, query_maxlength, doc_maxlength=None, query_padding="max_sequence", doc_padding="max_sequence", **kwargs):
        super().__init__(tokenizer, query_maxlength, doc_maxlength, query_padding, doc_padding, **kwargs)
    

class BeamDRCollator(RetrieverWithPosNegsCollator):

    def __init__(self, tokenizer, query_maxlength, doc_maxlength=None, query_padding="max_sequence", doc_padding="max_sequence", **kwargs):
        super().__init__(tokenizer, query_maxlength, doc_maxlength, query_padding, doc_padding, **kwargs)
    

class BGECollator(RetrieverWithPosNegsCollator):

    def __init__(self, tokenizer, query_maxlength, doc_maxlength=None, query_padding="max_sequence", doc_padding="max_sequence", **kwargs):
        super().__init__(tokenizer, query_maxlength, doc_maxlength, query_padding, doc_padding, **kwargs)
    
    def encode_query(self, query_list, **kwargs):
        instruction = "Represent this sentence for searching relevant passages:"
        # return self.encode([instruction + " " + query for query in query_list], self.query_maxlength, self.query_padding, **kwargs)
        query_list = [instruction + " " + query for query in query_list]
        return super().encode_query(query_list=query_list, **kwargs)


class APIRetrieverCollator:

    def __init__(self, query_maxlength, doc_maxlength=None, **kwargs):
        # APIRetriever使用的是外部API进行embedding的，因此不需要tokenizer
        self.query_maxlength = query_maxlength 
        self.doc_max_length = doc_maxlength or query_maxlength 
    
    def truncate_text(self, text: str, max_length: int) -> str:
        tokens = text.split()
        if len(tokens) <= max_length:
            return text 
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return ' '.join(tokens)
    
    def encode_query(self, query_list, **kwargs):
        query_maxlength = kwargs.get("query_maxlength", self.query_maxlength)
        return [self.truncate_text(query, query_maxlength) for query in query_list]

    def encode_doc(self, doc_list, **kwargs):
        doc_max_length = kwargs.get("doc_maxlength", self.doc_max_length)
        return [self.truncate_text(doc, doc_max_length) for doc in doc_list]
    

class FiDTripleSelectorCollator(EncoderDecoderCollator):

    def __call__(self, batch):
        """
        batch = [
            {
                "index": int, 
                "question": str, 
                "answers": [str],
                "positive_passage": str, 
                "negative_passages": [str]
            }
        ]
        """
        if isinstance(batch[0], list):
            batch = sum(batch, [])
        
        index = torch.tensor([example['index'] for example in batch])

        question_list, answer_index_list, context_list = [], [], [] 
        for example in batch:
            negative_passages = example["negative_passages"]
            num_negatives = len(negative_passages)
            answer_index = random.choice(range(num_negatives))
            context = negative_passages[:answer_index] + [example["positive_passage"]] + negative_passages[answer_index:]
            question_list.append(example["question"])
            answer_index_list.append(answer_index)
            context_list.append(context)

        #! 之所以从1开始是因为t5的tokenizer对0进行tokenize之后是两个token
        target = [str(answer_index+1) + " " + self.tokenizer.eos_token for answer_index in answer_index_list]
        target = self.tokenizer.batch_encode_plus(
            target, 
            max_length=self.answer_maxlength, 
            padding=True if self.padding=="max_sequence" else "max_length", 
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False # 手动添加eos token
        )
        target_input_ids = target["input_ids"]
        target_attention_mask = target["attention_mask"]
        target_input_ids[~target_attention_mask.bool()] = -100 # 把非答案tokens的id设置为-100 

        # 处理triples
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> version 1: FiD version >>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # num_triples = len(context_list[0])
        # def append_question(question, context):
        #     context_with_question = [] 
        #     for i, triple in enumerate(context):
        #         context_with_question.append(question + f"\nCandidate Triple: {str(i+1)}. " + triple)
        #     return context_with_question

        # encoder_input_texts = [append_question(question, context) for question, context in zip(question_list, context_list)]
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> version 1: FiD version >>>>>>>>>>>>>>>>>>>>>>>>>>>>


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> version 2: T5 version >>>>>>>>>>>>>>>>>>>>>>>>>>>>
        encoder_input_texts = [] 
        for question, context in zip(question_list, context_list):
            encoder_input_texts.append([self.get_encoder_input_text(question, context)])# 加一个[]是为复用FiD的代码
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> version 2: T5 version >>>>>>>>>>>>>>>>>>>>>>>>>>>>

        encoder_input_ids, encoder_attention_mask = encode_question_passages(
            encoder_input_texts, self.tokenizer, self.text_maxlength
        )

        # 得到answers 
        answers = [str(answer_index+1) for answer_index in answer_index_list]
        return index, encoder_input_ids, encoder_attention_mask, target_input_ids, target_attention_mask, answers
    
    def get_encoder_input_text(self, question: str, context: List[str]):

        def prompt(q, cs):
            choices = ["{}. {}".format(i+1, c) for i, c in enumerate(cs)]
            return q + f"\nCandidate Triples:\n" + "\n".join(choices) + "\nSelected Triple Option: "
        
        num_context = len(context)
        for i in range(num_context):
            candidate_context = context[:num_context-i]
            candidate_input = prompt(question, candidate_context)
            candidate_tokens = self.tokenizer.tokenize(candidate_input)
            if len(candidate_tokens) < self.text_maxlength:
                break
        result = candidate_input
        
        return result
    