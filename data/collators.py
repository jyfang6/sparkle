import torch
import random
from copy import deepcopy
from typing import List, Dict

def get_question_prompt(prompt_type, has_context, raw_question=None):

    if prompt_type not in ["remove_question"]:
        assert raw_question is not None # must provide question 

    if prompt_type == "standard":
        question_prompt = ""
        if has_context:
            question_prompt += "Based on the context, given "
        else:
            question_prompt += "Given "
        question_prompt += "the {} \nThe answer is:" # 不能有空格，不然有些tokenizer会根据空格多得到一个token
        question = question_prompt.format(raw_question)

    elif prompt_type == "remove_question":
        question_prompt = "The answer is:"
        question = question_prompt

    else:
        raise ValueError(f"{prompt_type} is not a supported question prompt type!")
    
    return question

def get_context_prompt(prompt_type, context_list, question=None):

    if prompt_type in ["prepend_question"]:
        assert question is not None # must provide question 

    if prompt_type == "concat":
        context_prompt = "The context is:\n"
        for i, text in enumerate(context_list):
            context_prompt += "{}. {}\n".format(i+1, text)
        # context_prompt = context_prompt.strip()
    elif prompt_type == "prepend_question":
        context_prompt = ""
        for i, text in enumerate(context_list):
            context_prompt += "{}. {} {}\n".format(i+1, question, text)
    else:
        raise ValueError(f"{prompt_type} is not a supported context prompt type!")
    return context_prompt

def add_token_padding(tokens, attention_mask, max_length, pad_token, padding_type, truncation_type):

    # 这个函数当tokens的数量大于max_length时,会删掉一些tokens
    assert len(tokens) == len(attention_mask)
    padding_length = max_length - len(tokens)
    if padding_type == "right":
        new_tokens = tokens + [pad_token] * padding_length
        new_attention_mask = attention_mask + [0] * padding_length
    elif padding_type == "left":
        new_tokens = [pad_token] * padding_length + tokens
        new_attention_mask = [0] * padding_length + attention_mask
    else:
        raise ValueError(f"{padding_type} is an invalid padding type!")
    
    if truncation_type == "right":
        new_tokens = new_tokens[:max_length]
        new_attention_mask = new_attention_mask[:max_length]
    elif truncation_type == "left":
        new_tokens = new_tokens[-max_length:]
        new_attention_mask = new_attention_mask[-max_length:]
    else:
        raise ValueError(f"{truncation_type} is not a valid truncation type!")
    
    assert len(new_tokens) == max_length
    assert len(new_attention_mask) == max_length
    
    return new_tokens, new_attention_mask

def truncate_to_max_sequence(input_ids, attention_mask, labels=None):

    # 首先检测是left padding还是right padding 
    assert len(input_ids.shape) == 2 # input_ids  must be 2-dimensional 
    assert len(attention_mask.shape) == 2 # attention_mask must be 2-dimensional 

    num_tokens = attention_mask.sum(0)
    padding_type = "right" if num_tokens[0].item() > num_tokens[-1].item() else "left"
    # 然后根据attention_mask进行truncate 
    if padding_type == "right":
        max_num_tokens = (attention_mask!=0).max(0)[0].nonzero(as_tuple=False)[-1].item()+1
        input_ids = input_ids[..., :max_num_tokens]
        attention_mask = attention_mask[..., :max_num_tokens]
        if labels is not None:
            labels = labels[..., :max_num_tokens]
    else:
        min_padding_length = (attention_mask!=0).max(0)[0].nonzero(as_tuple=False)[0].item()
        input_ids = input_ids[..., min_padding_length:]
        attention_mask = attention_mask[..., min_padding_length:]
        if labels is not None:
            labels = labels[..., min_padding_length:]
    
    if labels is not None:
        return input_ids, attention_mask, labels
    else:
        return input_ids, attention_mask
    
def get_position_ids(attention_mask):
    position_ids = ((torch.cumsum(attention_mask, dim=1)).type_as(attention_mask) * attention_mask).long() - 1 
    position_ids = torch.where(position_ids>=0, position_ids, 0)
    return position_ids


class DecoderCollator:

    """
    DecoderOnly模型的Collator, 它的输入格式为: passages (根据prompt构建) + question + answer (训练的时候)
    """

    def __init__(self, tokenizer, text_maxlength, answer_maxlength=25, context_prompt_type="concat", \
                 question_prompt_type="standard", padding="max_sequence", padding_type=None, **kwargs):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.context_prompt_type = context_prompt_type
        self.question_prompt_type = question_prompt_type
        self.padding = padding
        assert self.padding in ["max_length", "max_sequence"] 
        self.padding_type = padding_type
        self.kwargs = kwargs
    
    def __call__(self, batch):

        batch_size = len(batch)
        index = torch.tensor([example['index'] for example in batch])

        batch_input_ids_with_target, batch_attention_mask_with_target, batch_labels = [], [], []
        batch_input_ids_wo_target, batch_attention_mask_wo_target = [], []

        for example in batch:

            has_context = (example["passages"] is not None and len(example["passages"]) > 0)
            question = get_question_prompt(self.question_prompt_type, has_context=has_context, raw_question=example["question"])
            if "target" in example and example["target"] is not None:
                # question: "xxx, the answer is: "
                question_with_target = question + " " + example["target"].strip() + " " + self.tokenizer.eos_token 
            else:
                question_with_target = question 
            question_tokens = self.tokenizer.tokenize(question)
            question_with_target_tokens = self.tokenizer.tokenize(question_with_target)
            answer_length = len(question_with_target_tokens) - len(question_tokens)

            # 处理passage
            if has_context:
                context = get_context_prompt(self.context_prompt_type, example["passages"], question=example["question"])
                context_tokens = self.tokenizer.tokenize(context)
                # truncate context
                max_context_length = self.text_maxlength - self.answer_maxlength - len(question_tokens)
                context_tokens = context_tokens[:max_context_length]
            else:
                context_tokens = [] 
            
            # 获取带有答案的input_ids, attention_mask, position_ids和labels
            input_with_target_tokens = context_tokens + question_with_target_tokens
            input_with_target_attention_mask = [1] * len(input_with_target_tokens)
            # padding & truncation 
            # input_with_target_padding_type = example["padding"] if "padding" in example else "right"
            input_with_target_padding_type = self.padding_type if self.padding_type is not None else "right"
            input_with_target_tokens, input_with_target_attention_mask = \
                add_token_padding(
                    tokens=input_with_target_tokens, 
                    attention_mask=input_with_target_attention_mask, 
                    max_length=self.text_maxlength, 
                    pad_token=self.tokenizer.pad_token,
                    padding_type=input_with_target_padding_type,
                    truncation_type="left" # 如果答案过长的话把context的开头删掉一些，保留全部的答案
                )
            input_with_target_token_ids = self.tokenizer.convert_tokens_to_ids(input_with_target_tokens)

            if answer_length > 0:
                # 当带有答案的时候
                label = torch.tensor(input_with_target_token_ids, dtype=torch.long)
                padding_length = (label == self.tokenizer.pad_token_id).sum().item()
                if input_with_target_padding_type=="right":
                    context_question_length = self.text_maxlength-padding_length-answer_length
                    label[:context_question_length] = -100 # 去掉context和question
                    label[context_question_length+answer_length:] = -100 # 去掉padding
                elif input_with_target_padding_type == "left":
                    label[:(self.text_maxlength-answer_length)] = -100 
                else:
                    raise ValueError(f"{input_with_target_padding_type} is an invalid padding type!")
                label = label.tolist() # 转换为列表
            else:
                label = None 
            
            # 获取不带答案的input_ids, attention_mask, position_ids 
            input_wo_target_tokens = context_tokens + question_tokens
            input_wo_target_attention_mask = [1] * len(input_wo_target_tokens)
            # padding & truncation 
            # input_wo_target_padding_type = example["padding"] if "padding" in example else "left" 
            input_wo_target_padding_type = self.padding_type if self.padding_type is not None else "left" # generate的时候默认用
            input_wo_target_tokens, input_wo_target_attention_mask = \
                add_token_padding(
                    tokens=input_wo_target_tokens,
                    attention_mask=input_wo_target_attention_mask,
                    max_length=self.text_maxlength,
                    pad_token=self.tokenizer.pad_token, 
                    padding_type=input_wo_target_padding_type, 
                    truncation_type="left" # 去掉context保留question
                )
            input_wo_target_token_ids = self.tokenizer.convert_tokens_to_ids(input_wo_target_tokens)

            batch_input_ids_with_target.append(input_with_target_token_ids)
            batch_attention_mask_with_target.append(input_with_target_attention_mask)
            if label is not None:
                batch_labels.append(label)
            batch_input_ids_wo_target.append(input_wo_target_token_ids)
            batch_attention_mask_wo_target.append(input_wo_target_attention_mask)
        
        # 转换为tensor 
        input_ids_with_target = torch.tensor(batch_input_ids_with_target, dtype=torch.long)
        attention_mask_with_target = torch.tensor(batch_attention_mask_with_target, dtype=torch.long)
        labels = torch.tensor(batch_labels) if len(batch_labels) == batch_size else None
        input_ids_wo_target = torch.tensor(batch_input_ids_wo_target, dtype=torch.long)
        attention_mask_wo_target = torch.tensor(batch_attention_mask_wo_target, dtype=torch.long)

        # truncate to max sequence length 
        if self.padding == "max_sequence":
            input_ids_with_target, attention_mask_with_target, labels = truncate_to_max_sequence(
                input_ids=input_ids_with_target, 
                attention_mask=attention_mask_with_target,
                labels=labels
            )
            input_ids_wo_target, attention_mask_wo_target = truncate_to_max_sequence(
                input_ids=input_ids_wo_target,
                attention_mask=attention_mask_wo_target
            )
        
        # obtain position ids 
        position_ids_with_target = get_position_ids(attention_mask_with_target)
        position_ids_wo_target = get_position_ids(attention_mask_wo_target)
        
        return index, input_ids_with_target, attention_mask_with_target, position_ids_with_target, labels, \
            input_ids_wo_target, attention_mask_wo_target, position_ids_wo_target


class DecoderCollatorWithPathsChatFormat:

    def __init__(self, tokenizer, text_maxlength, answer_maxlength=25, context_type="triples", \
        num_demonstration=-1, demonstration_dataset=None, **kwargs):

        self.tokenizer = tokenizer 
        self.tokenizer.padding_side = "left" #设置为左padding
        self.text_maxlength = text_maxlength 
        self.answer_maxlength = answer_maxlength
        assert context_type in ["triples", "sentences", "all_documents", "documents", "chains"]
        self.context_type = context_type
        self.num_demonstration = num_demonstration
        if num_demonstration > 0:
            assert demonstration_dataset is not None # must provide demonstration dataset when num_demonstration>0
        self.kwargs = kwargs 
    

    def get_contexts(self, example):

        # 这个函数需要有以下的数据
        paths = example["paths"]
        contexts = example["contexts"]

        if self.context_type == "triples":
            # 单纯只用triples
            # paths_list = [
            #     ", ".join([triple_item["triple"] for triple_item in path])
            #         for i, path in enumerate(paths)
            # ]
            # v2: 把不同路径上的triples拼接在一起
            # paths_list = []
            # for i, path in enumerate(paths):
            #     for triple_item in path:
            #         triple = triple_item["triple"]
            #         if triple not in paths_list:
            #             paths_list.append(triple)

            # v3: 把不同路径上的triples转换为自然语言的格式
            paths_list = [] 
            for i, path in enumerate(paths):
                for triple_item in path["triples"]:
                    triple = triple_item['triple']
                    triple_sentence = triple.replace("<", "").replace(">", "").replace(";", "", 2)
                    if triple_sentence not in paths_list:
                        paths_list.append(triple_sentence)
            
            # v4: 把不同路径上的triples按照majority voting的方法进行排序
            # triples_scores = {}
            # for i, path in enumerate(paths):
            #     path_score = path["score"]
            #     for triple_item in path["triples"]:
            #         triple = triple_item['triple']
            #         triple_sentence = triple.replace("<", "").replace(">", "").replace(";", "", 2)
            #         # triples_scores[triple_sentence] = triples_scores.get(triple_sentence, 0.0) + path_score
            #         triples_scores[triple_sentence] = triples_scores.get(triple_sentence, 0.0) + 1 
            # ranked_triples = sorted(triples_scores.items(), key=lambda x: x[1], reverse=True)
            # paths_list = [triple for triple, score in ranked_triples]

        # TODO: 因为我修改了 paths 的格式，所以下面的代码可能需要重新修改一下
        
        if self.context_type == "sentences":
            # v1: 使用reasoning paths + sentences 
            # sentences_list = [] 
            # for i, path in enumerate(paths):
            #     one_path_text = "reasoning paths: {}".format(", ".join([triple_item["triple"] for triple_item in path]))
            #     one_path_sentences_indices = [] 
            #     for triple_item in path:
            #         doc_idx, sent_idx = triple_item["triple_position"]
            #         if doc_idx>=0 and sent_idx >=0 and (doc_idx, sent_idx) not in one_path_sentences_indices:
            #             one_path_sentences_indices.append((doc_idx, sent_idx))
            #     one_path_sentences = [
            #         "title: {}, context: {}".format(contexts[doc_idx]["title"], contexts[doc_idx]["sentences"][sent_idx]) 
            #             for doc_idx, sent_idx in one_path_sentences_indices
            #     ]
            #     sentences_list.append(one_path_text + " texts: {}".format(" ".join(one_path_sentences)))

            # v2: 使用triples来源的sentences以及它的前后一个句子
            paths_sentences_indices_count_dict = {}
            for i, path in enumerate(paths):
                for triple_item in path:
                    doc_idx, sent_idx = triple_item["triple_position"]
                    if doc_idx >= 0 and sent_idx >= 0:
                        paths_sentences_indices_count_dict[(doc_idx, sent_idx)] = \
                            paths_sentences_indices_count_dict.get((doc_idx, sent_idx), 0) + 1 
            sentences_list = [] 
            ranked_paths_sentences_indices = sorted(paths_sentences_indices_count_dict.items(), key=lambda x: x[1], reverse=True)
            for (doc_idx, sent_idx), count in ranked_paths_sentences_indices:
                title = contexts[doc_idx]["title"]
                sentences = contexts[doc_idx]["sentences"]
                one_sentence_texts_list = []
                if sent_idx - 1 >= 0:
                    one_sentence_texts_list.append(sentences[sent_idx-1])
                one_sentence_texts_list.append(sentences[sent_idx])
                if sent_idx + 1 < len(sentences):
                    one_sentence_texts_list.append(sentences[sent_idx+1])
                one_candidate_sentence = "title: {}, text: {}".format(title, " ".join(one_sentence_texts_list))
                # if one_candidate_sentence not in sentences_list:
                sentences_list.append(one_candidate_sentence)
        
        
        if self.context_type == "documents":
            # v1: 这里的documents是在路径的后面拼接上triples来源的文档
            # paths_with_documents_list = [] 
            # for i, path in enumerate(paths):
            #     one_path_text = "reasoning paths: {}".format(", ".join([triple_item["triple"] for triple_item in path]))
            #     one_path_documents_indices = [] 
            #     for triple_item in path:
            #         doc_idx, sent_idx = triple_item["triple_position"]
            #         if doc_idx >=0 and doc_idx not in one_path_documents_indices:
            #             one_path_documents_indices.append(doc_idx)
            #     one_path_documents = [
            #         "title: {}, context: {}".format(
            #             contexts[idx]["title"], " ".join(contexts[idx]["sentences"])
            #         ) 
            #         for idx in one_path_documents_indices
            #     ]
            #     one_path_text += " texts: {}".format(" ".join(one_path_documents))
            #     paths_with_documents_list.append(one_path_text)

            # v2: 按照triples的出现顺序来拼接document
            # documents_indices = [] 
            # for i, path in enumerate(paths):
            #     path_score = path["score"]
            #     for triple_item in path["triples"]:
            #         doc_idx, sent_idx = triple_item["triple_position"]
            #         if doc_idx > 0 and doc_idx not in documents_indices:
            #             documents_indices.append(doc_idx)
            # paths_with_documents_list = [] 
            # for idx in documents_indices:
            #     paths_with_documents_list.append("title: {}, text: {}".format(contexts[idx]["title"], " ".join(contexts[idx]["sentences"])))
            
            # v3: 按照document出现的次数进行排序
            paths_texts_list = [] 
            paths_documents_indices_count_dict = {}
            for i, path in enumerate(paths):
                # one_path_text = "reasoning paths: {}".format(", ".join([triple_item["triple"] for triple_item in path]))
                path_score = path["score"]
                for triple_item in path["triples"]:
                    triple = triple_item["triple"]
                    triple_sentence = triple.replace("<", "").replace(">", "").replace(";", "", 2)
                    if triple_sentence not in paths_texts_list:
                        paths_texts_list.append(triple_sentence)
                for triple_item in path["triples"]:
                    doc_idx, sent_idx = triple_item["triple_position"]
                    if doc_idx >=0:
                        paths_documents_indices_count_dict[doc_idx] = paths_documents_indices_count_dict.get(doc_idx, 0) + 1 
            
            # 对document按照triples的出现顺序进行排序
            paths_with_documents_list = []
            ranked_paths_documents_indices = sorted(paths_documents_indices_count_dict.items(), key=lambda x: x[1], reverse=True)
            for idx, count in ranked_paths_documents_indices:
                paths_with_documents_list.append("title: {}, text: {}".format(contexts[idx]["title"], " ".join(contexts[idx]["sentences"])))
            
            # paths_with_documents_list = paths_texts_list + paths_with_documents_list # 这里不拼接triples, 拼接triples之后性能会有一点点的下降
            
            # 额外添加document
            # existing_document_indices = [idx for idx, count in ranked_paths_documents_indices]
            # for i in range(len(contexts)):
            #     if i not in existing_document_indices:
            #         paths_with_documents_list.append("title: {}, text: {}".format(contexts[i]["title"], " ".join(contexts[i]["sentences"])))      


        if self.context_type == "all_documents":
            # documents_list = [" ".join(context_item["sentences"]) for context_item in contexts]
            num_contexts = self.kwargs.get("n_context", len(contexts))
            documents_list = ["title: {}, text: {}".format(context_item["title"], " ".join(context_item["sentences"])) for context_item in contexts[:num_contexts]]
        
        if self.context_type == "chains":
            chains_list = [] 
            for path in paths:
                triples_texts = [triple["triple"] for triple in path["triples"]]
                chains_list.append(", ".join(triples_texts))

        if self.context_type == "triples":
            context_text_list = paths_list 
        elif self.context_type == "sentences":
            context_text_list = sentences_list
        elif self.context_type == "all_documents":
            context_text_list = documents_list
        elif self.context_type == "documents":
            context_text_list = paths_with_documents_list
        elif self.context_type == "chains":
            context_text_list = chains_list

        context_text = "\n".join(["{}. {}".format(i+1, text) for i, text in enumerate(context_text_list)])

        return context_text


    def get_prompts_chat_format(self, batch):

        def convert_several_examplars_to_text(examplars):
            return "\n\n".join(examplars)

        prompts = [] 
        has_contexts = batch[0]["paths"] is not None 

        for example in batch:

            if has_contexts:
                # instruction = "Given a question and some contexts, please only output the answer to the question."
                if self.context_type == "chains":
                    instruction = "You will be given some reasoning chains and a question. "\
                        "A reasoning chain is a series of logically connected knowledge triples in the form of <head entity; relation; tail entity> that can help answer a question. "\
                            "Since the specific reasoning chain is unknown, there are several chains representing different possibilities to obtain an answer, with some overlapping between chains. "\
                                "Your task is to correctly answer the question based on these reasoning chains. Please ONLY output the answer to the question."
                else:
                    instruction = "Given some contexts and a question, please only output the answer to the question."
            else:
                instruction = "Given a question, please only output the answer to the question."

            if self.num_demonstration > 0:
                instruction = instruction + "\n\nHere are some examples:\n\n"
                examplars = [self.demonstration_dataset[i] for i in range(self.num_demonstration)]
                if has_contexts:
                    examplars_template = "context:\n{}\nquestion: {}\nthe correct answer is:{}"
                    examplars = [examplars_template.format(self.get_contexts(item), item["question"], item["answers"][0]) for item in examplars]
                else:
                    examplars_template = "question: {}\nthe correct answer is:{}"
                    examplars = [examplars_template.format(item["question"], item["answers"][0]) for item in examplars]
                instruction = instruction + convert_several_examplars_to_text(examplars)
                
            user_input_text = example["question"] 
            if has_contexts:
                # user_input_text = user_input_text + "\ncontext:\n" + self.get_contexts(example) + "\n" + "the correct answer is:"
                if self.context_type == "chains":
                    user_input_text = "reasoning chains:\n" + self.get_contexts(example) + "\n" + user_input_text + "\n" + "the correct answer is:"
                else:
                    user_input_text = "context:\n" + self.get_contexts(example) + "\n" + user_input_text + "\n" + "the correct answer is:"
            else:
                user_input_text = user_input_text + "\n" + "the correct answer is:"
            
            prompts.append(
                [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_input_text}
                ]
            )
        # from pdb import set_trace; set_trace()
        return prompts 
    
    
    def tokenizer_encode(self, prompts):
        texts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        batch_dict = self.tokenizer(texts, max_length=self.text_maxlength, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs 

    
    def __call__(self, batch): 

        """
        batch: [
            {
                "index": int, 
                "question": str, 
                "target": str, 
                "answers": [str], 
                "contexts": None / [
                    {
                        "title": str, 
                        "sentences": [str],
                    }
                ],
                # "paths": None / [
                #     [{"triple": "<xx; xx; xx>", "triple_position": [int, int]}]
                # ]
                "paths": None / [
                    {
                        "triples": [{"triple": "<xx; xx; xx>", "triple_position": [int, int]}], 
                        "score": float, 
                    }
                ]
            }
        ]
        """

        batch_size = len(batch)
        index = torch.tensor([example['index'] for example in batch])
        prompts = self.get_prompts_chat_format(batch)
        inputs = self.tokenizer_encode(prompts)
        return index, inputs 


class FlanT5CollatorWithPaths(DecoderCollatorWithPathsChatFormat):

    def __init__(self, tokenizer, text_maxlength, answer_maxlength=25, context_type="triples", \
        num_demonstration=-1, demonstration_dataset=None, **kwargs):

        self.tokenizer = tokenizer 
        self.tokenizer.padding_side = "right" #设置为右padding
        self.text_maxlength = text_maxlength 
        self.answer_maxlength = answer_maxlength
        assert context_type in ["triples", "sentences", "all_documents", "documents"]
        self.context_type = context_type
        self.num_demonstration = num_demonstration
        if num_demonstration > 0:
            assert demonstration_dataset is not None # must provide demonstration dataset when num_demonstration>0
        self.kwargs = kwargs 

    def get_prompts(self, batch):

        has_contexts = batch[0]["paths"] is not None

        if has_contexts:
            instruction = "Given some contexts and a question, please only output the answer to the question"
        else:
            instruction =  "Given a question, please only output the answer to the question"

        instruction_token_ids = self.tokenizer.encode(instruction, add_special_tokens=False)

        prompts_list = [] 
        for example in batch:
            question = example["question"]
            if has_contexts:
                candidate_context = "context:\n{}".format(self.get_contexts(example))
                candidate_context_token_ids = self.tokenizer.encode(candidate_context, add_special_tokens=False)
                question_token_ids = self.tokenizer.encode(question, add_special_tokens=False)
                max_possible_context_token_ids = self.text_maxlength - len(instruction_token_ids) - len(question_token_ids) - 10 
                context = self.tokenizer.decode(candidate_context_token_ids[:max_possible_context_token_ids])
                prompt = context + "\n" + question
                prompts_list.append(prompt)
            else:
                prompts_list.append(question)
        
        prompts_list = ["{}: {}\nthe correct answer is:".format(instruction, prompt) for prompt in prompts_list]

        # from pdb import set_trace; set_trace()
        return prompts_list
    

    def tokenizer_encode(self, prompts):
        batch_dict = self.tokenizer(prompts, max_length=self.text_maxlength, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs 
    
        
    def __call__(self, batch):

        batch_size = len(batch)
        index = torch.tensor([example['index'] for example in batch])
        prompts = self.get_prompts(batch)
        inputs = self.tokenizer_encode(prompts)
        return index, inputs 


class MistralCollatorWithPaths(DecoderCollatorWithPathsChatFormat):

    def __init__(self, tokenizer, text_maxlength, answer_maxlength=25, context_type="triples", \
        num_demonstration=-1, demonstration_dataset=None, **kwargs):

        self.tokenizer = tokenizer 
        self.tokenizer.padding_side = "left" 
        self.text_maxlength = text_maxlength 
        self.answer_maxlength = answer_maxlength
        assert context_type in ["triples", "sentences", "all_documents", "documents"]
        self.context_type = context_type
        self.num_demonstration = num_demonstration
        if num_demonstration > 0:
            assert demonstration_dataset is not None # must provide demonstration dataset when num_demonstration>0
        self.kwargs = kwargs 

    def get_prompts(self, batch):

        has_contexts = batch[0]["paths"] is not None

        if has_contexts:
            instruction = "Given some contexts and a question, please only output the answer to the question.\n"
        else:
            instruction =  "Given a question, please only output the answer to the question.\n"
        
        prompts_list = [] 
        for example in batch:
            question = example["question"]
            if has_contexts:
                context = "context:\n{}".format(self.get_contexts(example))
                prompt = context + "\n" + question
                prompts_list.append(prompt)
            else:
                prompts_list.append(question)
        
        prompts_list = ["{}{}\nthe correct answer is:".format(instruction, prompt) for prompt in prompts_list]

        # from pdb import set_trace; set_trace()
        return prompts_list
    

    def tokenizer_encode(self, prompts):
        batch_dict = self.tokenizer(prompts, max_length=self.text_maxlength, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs 
    
        
    def __call__(self, batch):

        batch_size = len(batch)
        index = torch.tensor([example['index'] for example in batch])
        prompts = self.get_prompts(batch)
        inputs = self.tokenizer_encode(prompts)
        return index, inputs 
 

def encode_question_passages(batch_text_passages, tokenizer, max_length):

    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding="max_length",
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)

    return passage_ids, passage_masks


class EncoderDecoderCollator:

    def __init__(self, tokenizer, text_maxlength, answer_maxlength=25, padding="max_sequence", **kwargs):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.padding = padding
    
    def __call__(self, batch):

        index = torch.tensor([example['index'] for example in batch])
        target = [example["target"] + " " + self.tokenizer.eos_token for example in batch]
        target = self.tokenizer.batch_encode_plus(
            target, # T5的tokenizer会自己添加
            max_length=self.answer_maxlength, 
            padding=True if self.padding=="max_sequence" else "max_length", 
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False # 手动添加eos token
        )
        target_input_ids = target["input_ids"]
        target_attention_mask = target["attention_mask"]
        target_input_ids[~target_attention_mask.bool()] = -100 # 把非答案tokens的id设置为-100 

        # 处理passages
        num_passages = max([len(example["passages"]) for example in batch]) if batch[0]["passages"] is not None else 0 
        def append_question(example):
            if example['passages'] is None or num_passages==0:
                # 不使用passages的时候
                return [example['question']]
            passages = example["passages"]
            while len(passages) < num_passages:
                # 可能有些问题的context达不到指定的数量
                passages = passages + passages[:(num_passages - len(passages))]
            return [example['question'] + " " + t for t in passages[:num_passages]]
        
        encoder_input_texts = [append_question(example) for example in batch]
        encoder_input_ids, encoder_attention_mask = encode_question_passages(
            encoder_input_texts, self.tokenizer, self.text_maxlength
        )

        # 得到每一个passage是否包含正确答案
        if "has_answers" in batch[0] and batch[0]["has_answers"] is not None:
            positive_passage_label = torch.tensor([example["has_answers"] for example in batch], dtype=torch.float32)
        else:
            positive_passage_label = None
        
        other_model_kwargs = {}
        other_model_kwargs["positive_passage_label"] = positive_passage_label
        
        return index, encoder_input_ids, encoder_attention_mask, target_input_ids, target_attention_mask, other_model_kwargs
    

class EncoderDecoderWOFiDCollator:

    def __init__(self, tokenizer, text_maxlength, answer_maxlength=25, context_prompt_type="concat", \
                 question_prompt_type="standard", padding="max_sequence", **kwargs):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.context_prompt_type = context_prompt_type
        self.question_prompt_type = question_prompt_type
        self.padding = padding
    
    def __call__(self, batch): 

        index = torch.tensor([example['index'] for example in batch])
        target = [example["target"] + " " + self.tokenizer.eos_token for example in batch]
        target = self.tokenizer.batch_encode_plus(
            target, # T5的tokenizer会自己添加eos token 
            max_length=self.answer_maxlength, 
            padding=True if self.padding=="max_sequence" else "max_length", 
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False # 手动添加eos token
        )
        target_input_ids = target["input_ids"]
        target_attention_mask = target["attention_mask"]
        target_input_ids[~target_attention_mask.bool()] = -100 # 把非答案tokens的id设置为-100 

        num_passages = max([len(example["passages"]) for example in batch]) if batch[0]["passages"] is not None else 0 
        def prepend_context(example):
            if example['passages'] is None or num_passages==0:
                question = get_question_prompt(self.question_prompt_type, False, example["question"])
                return question
            else:
                question = get_question_prompt(self.question_prompt_type, True, example["question"])
                context_text = get_context_prompt(self.context_prompt_type, example["passages"], question=example["question"])
                return context_text + question
        
        encoder_input_texts = [prepend_context(example) for example in batch]
        encoder_inputs = self.tokenizer.batch_encode_plus(
            encoder_input_texts, 
            max_length=self.text_maxlength, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
        )
        encoder_input_ids, encoder_attention_mask = encoder_inputs["input_ids"], encoder_inputs["attention_mask"]
        
        return index, encoder_input_ids, encoder_attention_mask, target_input_ids, target_attention_mask 


class RetrieverCollator:

    def __init__(self, tokenizer, query_maxlength, doc_maxlength, query_padding="max_sequence", doc_padding ="max_sequence", **kwargs):
        self.tokenizer = tokenizer
        self.query_maxlength = query_maxlength
        self.doc_maxlength = doc_maxlength
        self.query_padding = query_padding # must be chosen from ["max_length", "max_sequence"]
        self.doc_padding = doc_padding
        self.kwargs = kwargs
    
    def encode(self, text_list, maxlength, padding, **kwargs):

        # text_list是一个str的列表或者一个列表的列表
        assert padding in ["max_length", "max_sequence"] # padding must be chosen from ["max_length", "max_sequence"]

        if text_list is None or (isinstance(text_list, (tuple, list)) and len(text_list) == 0):
            raise ValueError("text_list is None or an empty tuple/list!")
        
        if isinstance(text_list, str) or isinstance(text_list[0], str):
            # 如果text_list是单个字符串(tokenizer的输出也是一个二维的矩阵，矩阵第一维为1)或者字符串的列表
            padding_scheme = padding if padding == "max_length" else True
            outputs = self.tokenizer(text_list, max_length=maxlength, padding=padding_scheme, truncation=True, return_tensors='pt')
            input_ids, attention_mask = outputs["input_ids"], outputs["attention_mask"]
        elif isinstance(text_list[0], (list, tuple)):
            input_ids, attention_mask = encode_question_passages(text_list, self.tokenizer, maxlength) # 默认会padding到maxlength
            if padding == "max_sequence":
                *input_ids_shape, _ = input_ids.shape
                *attention_mask_shape, _ = attention_mask.shape
                input_ids, attention_mask = truncate_to_max_sequence(input_ids.reshape(-1, maxlength), attention_mask.reshape(-1, maxlength))
                input_ids = input_ids.reshape(*input_ids_shape, input_ids.shape[-1])
                attention_mask = attention_mask.reshape(*attention_mask_shape, attention_mask.shape[-1])
        else:
            raise ValueError(f"Unrecognised type for {text_list}!")
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_query(self, query_list, **kwargs):
        query_maxlength = kwargs.get("max_length", None) or self.query_maxlength
        return self.encode(query_list, query_maxlength, self.query_padding, **kwargs)

    def encode_doc(self, doc_list, **kwargs):
        doc_maxlength = kwargs.get("max_length", None) or self.doc_maxlength
        return self.encode(doc_list, doc_maxlength, self.doc_padding, **kwargs)

    def __call__(self):
        raise NotImplementedError("__call__ is not implemented for the base RetrieverCollator!")


class RetrieverWithPosNegsCollator(RetrieverCollator):

    def __init__(self, tokenizer, query_maxlength, doc_maxlength=None, query_padding="max_sequence", doc_padding ="max_sequence", **kwargs):
        if doc_maxlength is None:
            doc_maxlength = query_maxlength
        super().__init__(tokenizer, query_maxlength, doc_maxlength, query_padding=query_padding, doc_padding=doc_padding, **kwargs)

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
            # 在KGChainRetrieverDataset中因为一个question有多个正样本, 所以返回了一个列表
            batch = sum(batch, []) 
        query_list = [example["question"] for example in batch]
        doc_list, positive_doc_indices = [], [] 
        for example in batch:
            positive_doc_indices.append(len(doc_list))
            doc_list.append(example["positive_passage"])
            doc_list.extend(example["negative_passages"])
        
        query_args = self.encode_query(query_list)
        doc_args = self.encode_doc(doc_list)
        positive_doc_indices = torch.tensor(positive_doc_indices, dtype=torch.long)
        index = torch.tensor([example["index"] for example in batch], dtype=torch.long)

        return query_args, doc_args, positive_doc_indices, index 
    

class RetrieverWithReaderScoresCollator(RetrieverCollator):

    def __init__(self, tokenizer, query_maxlength, doc_maxlength=None, query_padding="max_sequence", doc_padding ="max_sequence", **kwargs):
        if doc_maxlength is None:
            doc_maxlength = query_maxlength
        super().__init__(tokenizer, query_maxlength, doc_maxlength, query_padding=query_padding, doc_padding=doc_padding, **kwargs)
    
    def __call__(self, batch): 

        """
        batch = [
            {
                "index": int, 
                "question": str, 
                "answers": [str],
                "passages: [str], 
                "gold_scores": [float] / None, 
            }
        ]
        """
        query_list = [example["question"] for example in batch]
        doc_list = [example["passages"] for example in batch]
        if "gold_scores" in batch[0] and batch[0]["gold_scores"] is not None:
            gold_scores_list = [example["gold_scores"] for example in batch]
            gold_scores = torch.tensor(gold_scores_list).float()
        else:
            gold_scores = None

        query_args = self.encode_query(query_list)
        doc_args = self.encode_doc(doc_list)
        index = torch.tensor([example["index"] for example in batch], dtype=torch.long)

        return query_args, doc_args, gold_scores, index

