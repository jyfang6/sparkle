import warnings
from tqdm import trange
from copy import deepcopy 
from typing import List, Dict, Union, Optional, Any, Tuple

import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    LlamaForCausalLM, 
    Qwen2ForCausalLM,
    MistralForCausalLM,
    Gemma2ForCausalLM, 
    T5ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig,
)

from utils.utils import * 
from generator.utils import (
    pad_token_ids, 
    pad_token_logits,
    append_texts_to_decoder_only_generator_inputs,
    append_texts_to_encoder_decoder_generator_inputs
)

SUPPORTED_DECODER_ONLY_GENERATORS = [LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, Gemma2ForCausalLM]
SUPPORTED_ENCODER_DECODER_GENERATORS = [T5ForConditionalGeneration]


class Generator(nn.Module):

    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        generator: AutoModelForCausalLM, 
        max_length: int=4096, 
        max_new_tokens: int=128,
        batch_size: int=4, 
        **kwargs
    ):
        super().__init__()

        supported_generator_types = tuple(SUPPORTED_DECODER_ONLY_GENERATORS+SUPPORTED_ENCODER_DECODER_GENERATORS)
        assert isinstance(generator, supported_generator_types) # currently only support using LLaMA3 or Qwen2 as the generator

        self.tokenizer = tokenizer
        self.generator = generator
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.check_tokenizer_padding()

        self.is_chat = kwargs.get("is_chat", None) or self.init_is_chat()
        self.is_encoder_decoder = kwargs.get("is_encoder_decoder", None) or self.init_is_encoder_decoder()
        self.config = self.generator.config 
        self.config.update(kwargs)
    
    @property
    def device(self):
        return self.generator.device 
    
    @property
    def dtype(self):
        return self.generator.dtype

    def init_is_chat(self):
        model_name_or_path = self.generator.config._name_or_path.lower()
        if "instruct" in model_name_or_path or "chat" in model_name_or_path or "-it" in model_name_or_path:
            is_chat = True
        elif any(name in model_name_or_path for name in ["qwq-32b", "searchr1"]):
            # NOTE: searchr1使用chat template而r1-searcher则使用普通的format
            is_chat = True
        else:
            is_chat = False
        return is_chat
    
    def init_is_encoder_decoder(self):
        if isinstance(self.generator, tuple(SUPPORTED_ENCODER_DECODER_GENERATORS)):
            is_encoder_decoder = True 
        elif isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
            is_encoder_decoder = False
        else:
            raise ValueError(f"{type(self.generator)} is an unknow generator!")
        return is_encoder_decoder
    
    def check_tokenizer_padding(self):
        if isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
                raise ValueError("pad_token or pad_token_id is None in the tokenizer of generator. suggest to set pad_token and pad_token_id to eos_token and eos_token_id respectively.")
            if self.tokenizer.padding_side == "right":
                raise ValueError("Dected right padding using decoder-only transformers as the generator, which may cause some errors. It is suggested to use \"left\" padding!")
    
    def can_use_chat_format(self): 
        """Check if tokenizer supports chat template"""
        try:
            # Try with minimal input to test if chat template works
            self.tokenizer.apply_chat_template([{"role": "user", "content": "test"}], tokenize=False)
            return True
        except Exception:
            return False
    
    def get_generator_prompts_chat_format(
        self, 
        instructions: List[str], 
        messages: Union[List[List[dict]], List[str]],
        **kwargs
    ) -> List[List[Dict[str, str]]]:
        """
        Input: 
            instruction: [str]
            messages: [str] or [[{"user": "user_content"}, {"assistant": "assistant_content"}],...]
        Output:
            prompts: [[{"role": xxx, "content": xxx}, {"role": xxx, "content": xxx}]]
        """
        prompts = [] 
        if instructions is None:
            instructions = [""] * len(messages)
        assert len(instructions) == len(messages) # number of instructions shoule be the same as messages 
        for instruction, message_list in zip(instructions, messages):
            if isinstance(self.generator, (LlamaForCausalLM, Qwen2ForCausalLM)):
                if instruction.strip():
                    # add system instruction if provided
                    one_prompt = [{"role": "system", "content": instruction}]
                else:
                    one_prompt = []
                if isinstance(message_list, str):
                    one_prompt.append({"role": "user", "content": message_list})
                elif isinstance(message_list, list):
                    assert "user" in message_list[0] # # the first message must comes from user in the form of: {"user": "user_message"}
                    for message in message_list:
                        if "user" in message:
                            one_prompt.append({"role": "user", "content": message["user"]})
                        # if "system" in message:
                        #     one_prompt.append({"role": "system", "content": message["system"]})
                        if "assistant" in message:
                            one_prompt.append({"role": "assistant", "content": message["assistant"]})
                else:
                    raise ValueError(f"Invalid message type: {type(message_list)}. Only support str or List[dict] messages")
                prompts.append(one_prompt)
            elif isinstance(self.generator, (MistralForCausalLM, Gemma2ForCausalLM)):
                # Mistral Don't have System Role 
                if isinstance(message_list, str):
                    one_prompt = [{"role": "user", "content": instruction + "\n\n" + message_list}]
                elif isinstance(message_list, list):
                    assert "user" in message_list[0] # the first message must comes from user in the form of: {"user": "user_message"}
                    one_prompt = [{"role": "user", "content": instruction + "\n\n" + message_list[0]["user"]}]
                    for message in message_list[1:]:
                        if "user" in message:
                            one_prompt.append({"role": "user", "content": message["user"]})
                        if "assistant" in message:
                            one_prompt.append({"role": "assistant", "content": message["assistant"]})
                else:
                    raise ValueError(f"Invalid message type: {type(message_list)}. Only support str or List[dict] messages")
                prompts.append(one_prompt)
            else:
                raise NotImplemented(f"chat format for {type(self.generator)} is not implemented yet!")
        return prompts
    
    def tokenizer_encode_chat_format(self, prompts: List[List[Dict[str, str]]], max_length: int=None, add_generation_prompt: bool=True, **kwargs) -> Dict[str, Tensor]:
        """
        Params:
            add_generation_prompt: 为True的时候会在token的最后添加<|start_header_id|>assistant<|end_header_id|>, 为False的时候以<|eot_id|>结尾
        Input:
            prompts: [[{"role": xxx, "content": xxx}, {"role": xxx, "content": xxx}]]
        Output:
            {"input_ids": Tensor, "attention_mask": Tensor}
        """
        max_length = self.max_length if max_length is None else max_length
        # apply_chat_template 会根据是user还是assistant的输入来添加特殊的token, 比如在LLaMA3的格式是:
        # <begin_of_text><|start_header_id|>system<|end_header_id|>\n\n 指令输入 <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n 用户输入 <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        texts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=add_generation_prompt) 
        # 下面这里默认添加了特殊的tokens, 比如LLaMA3会在开头添加<begin_of_text> token, 而T5会在最后添加</s>
        batch_dict = self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs
    
    def tokenizer_encode(self, prompts: List[str], max_length: int=None, **kwargs) -> Dict[str, Tensor]:
        max_length = self.max_length if max_length is None else max_length
        # 下面这里默认添加了特殊的tokens, 比如LLaMA3会在开头添加<begin_of_text> token, 而T5会在最后添加</s>
        batch_dict = self.tokenizer(prompts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs
    
    def get_generated_token_ids(self, input_ids: Tensor, token_ids: Tensor) -> Tensor:
        if isinstance(self.generator, T5ForConditionalGeneration): # 不清楚BART是否也是第一个是其他的token，所以只写T5
            generated_token_ids = token_ids[:, 1:] # T5模型第一个token是<bos> token
        elif isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
            generated_token_ids = token_ids[:, input_ids.shape[1]:]
        else:
            raise NotImplementedError(f"get_generated_token_ids is not implemented for {type(self.generator)}!")
        return generated_token_ids
    
    def get_stop_symbols_stopping_criteria(self, prompt_size: int, stop_words: Union[str, List[str]]) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        from generator.stop_word_criteria import StopWordCriteria
        criteria.append(
            StopWordCriteria(tokenizer=self.tokenizer, prompt_size=prompt_size, stop_words=stop_words)
        )
        return criteria 
    
    def greedy_generate(
        self, 
        inputs: Dict[str, Tensor],
        pad_to_max_new_tokens: bool=False,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Inputs: 
        {"input_ids": Tensor, "attention_mask": Tensor}

        Outputs:
        Tensor, Tensor
        """
        # batch_size, max_new_tokens, device = self.batch_size, self.max_new_tokens, self.device
        device = self.device
        batch_size = kwargs.get("batch_size", None) or self.batch_size
        max_new_tokens = kwargs.get("max_new_tokens", None) or self.max_new_tokens
        stopping_criteria = kwargs.get("stopping_criteria", None)
        verbose = kwargs.get("verbose", False)
        
        if verbose:
            progress_bar = trange((len(inputs["input_ids"])-1)//batch_size+1, desc="Generation Progress")

        generated_token_ids_list, generated_token_logits_list = [], [] 
        for i in range((len(inputs["input_ids"])-1)//batch_size+1):
            batch_inputs = {k: v[i*batch_size: (i+1)*batch_size] for k, v in inputs.items()}
            batch_inputs = to_device(batch_inputs, device)
            batch_outputs = self.generator.generate(
                **batch_inputs, 
                max_new_tokens=max_new_tokens, 
                output_scores=True, 
                return_dict_in_generate=True, 
                do_sample=False, 
                temperature=1.0,
                stopping_criteria=stopping_criteria,
            ) # temperature=1.5, do_sample=True)
            # batch_generated_token_ids = batch_outputs.sequences[:, batch_inputs["input_ids"].shape[1]:].detach().cpu()
            batch_generated_token_ids = self.get_generated_token_ids(batch_inputs["input_ids"], batch_outputs.sequences).detach().cpu()
            batch_generated_token_logits = torch.cat([token_scores.unsqueeze(1) for token_scores in batch_outputs.scores], dim=1).detach().cpu()
            
            generated_token_ids_list.append(batch_generated_token_ids)
            generated_token_logits_list.append(batch_generated_token_logits)
            if verbose:
                progress_bar.update(1)
        
        max_generation_length = max_new_tokens if pad_to_max_new_tokens else \
            max([x.shape[-1] for x in generated_token_ids_list])
        generated_token_ids_list = [
            pad_token_ids(
                token_ids, 
                max_length=max_generation_length, 
                pad_token_id=self.tokenizer.pad_token_id
            ) 
            for token_ids in generated_token_ids_list
        ]
        generated_token_logits_list = [
            pad_token_logits(
                token_logits, 
                max_length=max_generation_length
            ) 
            for token_logits in generated_token_logits_list
        ]

        generated_token_ids = torch.cat(generated_token_ids_list, dim=0)
        generated_token_logits = torch.cat(generated_token_logits_list, dim=0)

        return generated_token_ids, generated_token_logits
    
    def _prepare_generation_config(self, generation_config: Optional[GenerationConfig] = None, **kwargs) -> GenerationConfig:
        """
        根据generation_config和kwargs来准备generation config, 默认使用greedy decoding的方法, 目前只设置下面的几个参数, 如果需要设置额外的参数的话需要更新代码
        """
        if generation_config is None:
            # by default, use greedy decoding 
            generation_config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1)
        else:
            generation_config = deepcopy(generation_config) 
        
        # handle max new tokens
        max_new_tokens = kwargs.get("max_tokens", None) or kwargs.get("max_new_tokens", None)
        if max_new_tokens:
            generation_config.max_new_tokens = max_new_tokens
        if not generation_config.max_new_tokens:
            generation_config.max_new_tokens = self.max_new_tokens

        generation_config.do_sample = kwargs.get("do_sample", generation_config.do_sample)
        generation_config.num_beams = kwargs.get("num_beams", generation_config.num_beams)
        generation_config.temperature = kwargs.get("temperature", generation_config.temperature)
        generation_config.top_k = kwargs.get("top_k", generation_config.top_k)
        generation_config.top_p = kwargs.get("top_p", generation_config.top_p)
        generation_config.num_return_sequences = kwargs.get("num_return_sequences", generation_config.num_return_sequences)

        return generation_config
    
    def generation_config_based_generate(
        self, 
        inputs: Dict[str, Tensor],
        generation_config: GenerationConfig = None,
        pad_to_max_new_tokens: bool=False,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        device = self.device
        batch_size = kwargs.get("batch_size", self.batch_size)
        # max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        max_new_tokens = generation_config.max_new_tokens 
        stopping_criteria = kwargs.get("stopping_criteria", None)
        verbose = kwargs.get("verbose", False)

        if verbose:
            progress_bar = trange((len(inputs["input_ids"])-1)//batch_size+1, desc="Generation Progress")
        
        # print("do_sample: ", generation_config.do_sample)
        # print("num_beams: ", generation_config.num_beams)
        # print("temperature: ", generation_config.temperature)
        # print("top_k: ", generation_config.top_k)
        # print("top_p: ", generation_config.top_p) 
        # print("max_new_tokens: ", max_new_tokens)

        generated_token_ids_list, generated_token_logits_list = [], [] 
        for i in range((len(inputs["input_ids"])-1)//batch_size+1):
            batch_inputs = {k: v[i*batch_size: (i+1)*batch_size] for k, v in inputs.items()}
            batch_inputs = to_device(batch_inputs, device)
            # 如果generation_config.num_return_sequences > 1, 返回的token_ids的shape为(batch_size * num_return_sequences, num_new_tokens)
            batch_outputs = self.generator.generate(
                **batch_inputs, 
                # max_new_tokens=max_new_tokens, 
                output_scores=True, 
                return_dict_in_generate=True,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )
            batch_generated_token_ids = self.get_generated_token_ids(batch_inputs["input_ids"], batch_outputs.sequences).detach().cpu()
            batch_generated_token_logits = torch.cat([token_scores.unsqueeze(1) for token_scores in batch_outputs.scores], dim=1).detach().cpu()
            
            generated_token_ids_list.append(batch_generated_token_ids)
            generated_token_logits_list.append(batch_generated_token_logits)
            if verbose:
                progress_bar.update(1)
        
        max_generation_length = max_new_tokens if pad_to_max_new_tokens else \
            max([x.shape[-1] for x in generated_token_ids_list])
        generated_token_ids_list = [
            pad_token_ids(
                token_ids, 
                max_length=max_generation_length, 
                pad_token_id=self.tokenizer.pad_token_id
            ) 
            for token_ids in generated_token_ids_list
        ]
        generated_token_logits_list = [
            pad_token_logits(
                token_logits, 
                max_length=max_generation_length
            ) 
            for token_logits in generated_token_logits_list
        ]

        generated_token_ids = torch.cat(generated_token_ids_list, dim=0)
        generated_token_logits = torch.cat(generated_token_logits_list, dim=0)

        return generated_token_ids, generated_token_logits
    
    def generate(self, inputs, generation_config: Optional[GenerationConfig] = None, **kwargs) -> Tuple[Tensor, Tensor]:

        """
        目前支持的kwargs中的变量有:
        max_tokens/max_new_tokens: int, 
        batch_size: int, 
        stop_words: str / [str] 当模型在生成的过程中生成stop_words中的token的时候就停止, 只传一个str的时候会把每一个character当做stop words
        generation_config: GenerationConfig, 生成配置
        verbose: bool, 是否要打印进度条, 默认为False   

        TODO: 视情况考虑要不要加其他的采样策略吧
        (1) beam search: 会生成概率最大的num_beams个sequence, 设置参数: num_beams=2-5, temperature(可选), top_k=0以及top_p=1.0 (后面两个是默认设置, 可以不用去设置)
        (2) sample generation: 
            top_k: 在生成下一个token的时候, 只考虑概率最大的top_k个token, 默认为0, 允许所有的token参与采样; 如果为1的话, 只选择概率最大的token 
            top_p: 在生成下一个token的时候, 只考虑前概率之和大于top_p的token
        """

        batch_size = kwargs.get("batch_size", None)
        kwargs["batch_size"] = self.batch_size if batch_size is None else batch_size

        # obtain generation config
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        stopping_criteria = None
        if kwargs.get("stop_words", None) is not None:
            if isinstance(self.generator, T5ForConditionalGeneration):
                prompt_size = 1 # <bos> token
            elif isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
                prompt_size = inputs["input_ids"].shape[-1] # (left-padding) prompt length NOTE: might cause error when using right padding 
            else:
                raise ValueError(f"{type(self.generator)} is not a supported generator!")
            stopping_criteria = self.get_stop_symbols_stopping_criteria(prompt_size, kwargs["stop_words"])
        kwargs["stopping_criteria"] = stopping_criteria

        # return self.greedy_generate(inputs, **kwargs)
        return self.generation_config_based_generate(inputs=inputs, generation_config=generation_config, **kwargs)

    def _check_inputs(self, instructions: List[str] = None, inputs: Union[List[List[dict]], List[str]] = None):

        if inputs is None:
            raise ValueError("inputs is required!")
        if instructions is None:
            instructions = [""] * len(inputs)
        assert len(instructions) == len(inputs), \
            "instructions and inputs must have the same length! But got {} and {}".format(len(instructions), len(inputs))
        return instructions, inputs

    def _check_use_chat(self, use_chat: Optional[bool] = None) -> bool:
        # Encoder-decoder models don't use chat format
        if self.is_encoder_decoder:
            return False
        
        # For chat models (self.is_chat is True)
        if self.is_chat:
            # Default to True if use_chat is not specified
            return True if use_chat is None else use_chat
        
        # For non-chat models (self.is_chat is False)
        if use_chat:
            # Only check compatibility if use_chat is explicitly True
            return self.can_use_chat_format()
        
        # Default to False for non-chat models or when use_chat is False
        return False
    
    def prompt(
        self, 
        instructions: List[str] = None, 
        inputs: Union[List[List[dict]], List[str]] = None, 
        use_chat: bool = None,
        **kwargs
    ) -> List[str]:

        instructions, inputs = self._check_inputs(instructions, inputs)
        # if not self.is_encoder_decoder and self.is_chat and (use_chat is None or use_chat):
        if self._check_use_chat(use_chat):
            prompts_chat_format = self.get_generator_prompts_chat_format(
                instructions=instructions, messages=inputs, **kwargs
            )
            prompts = self.tokenizer.apply_chat_template(prompts_chat_format, tokenize=False, add_generation_prompt=True)
        else:
            assert all([isinstance(user_input, str) for user_input in inputs]) # "inputs" must be List[str] when not using instructed decoder-only models 
            prompts = [inst + "\n\n" + user_input if inst else user_input for inst, user_input in zip(instructions, inputs)]
        return prompts
        
    def generator_generate(
        self, 
        instructions: List[str] = None, 
        inputs: List[str] = None, 
        current_generated_texts: List[str]=None, 
        generation_config: GenerationConfig = None,
        use_chat: bool = None,
        **kwargs
    ):
        
        instructions, inputs = self._check_inputs(instructions, inputs)
        if current_generated_texts is not None:
            assert len(instructions) == len(current_generated_texts), \
                "instructions and current_generated_texts must have the same length! But got {} and {}".format(len(instructions), len(current_generated_texts))
        
        if self.is_encoder_decoder:
            prompts = [inst + "\n\n" + user_input if inst else user_input for inst, user_input in zip(instructions, inputs)]
            generator_inputs = self.tokenizer_encode(prompts)
            if current_generated_texts is not None:
                generator_inputs = append_texts_to_encoder_decoder_generator_inputs(
                    tokenizer=self.tokenizer, inputs=generator_inputs, texts=current_generated_texts,
                        decoder_start_token_id=self.config.decoder_start_token_id
                )
        else:
            # if self.is_chat and (use_chat is None or use_chat):
            if self._check_use_chat(use_chat):
                prompts_chat_format = self.get_generator_prompts_chat_format(
                    instructions=instructions, messages=inputs, **kwargs
                )
                generator_inputs = self.tokenizer_encode_chat_format(prompts_chat_format, **kwargs)
                if current_generated_texts is not None:
                    generator_inputs = append_texts_to_decoder_only_generator_inputs(
                        tokenizer=self.tokenizer, inputs=generator_inputs, texts=current_generated_texts
                    )
            else:
                prompts = [inst + "\n\n" + user_input if inst else user_input for inst, user_input in zip(instructions, inputs)]
                if current_generated_texts is not None:
                    prompts = [prompt + " " + text if text else prompt for prompt, text in zip(prompts, current_generated_texts)]
                generator_inputs = self.tokenizer_encode(prompts)

        # from pdb import set_trace; set_trace()
        generated_token_ids, generated_token_logits = self.generate(generator_inputs, generation_config=generation_config, **kwargs)
        return generated_token_ids, generated_token_logits
    

class AnswerGenerator(Generator):

    def __init__(self, tokenizer: AutoTokenizer, generator: AutoModelForCausalLM, max_length: int = 4096, max_new_tokens: int = 128, batch_size: int = 4, use_cot: bool= False, **kwargs):

        super().__init__(tokenizer, generator, max_length, max_new_tokens, batch_size, **kwargs)
        self.task_instruction = kwargs.get("task_instruction", None) 
        self.task_instruction_wo_context = "Given a question, please only output the answer to the question."
        self.task_instruction_with_context = "Given some context and a question, please only output the answer to the question."
        self.task_instruction_cot = "Answer the following question by reasoning step-by-step. After reasoning, you MUST use \"So the answer is:\" to output the answer."
        self.use_cot = use_cot
        self.answer_prefix = "The answer is:" if not self.use_cot else "Thought:"
        if self.use_cot:
            self.cot_examplars = self.load_cot_examplars()
    
    def load_cot_examplars(self):

        examplar_type: str = self.kwargs.get("examplar_type", "hotpotqa")
        num_examplars: int = self.kwargs.get("num_examplars", 3)
        use_ctxs: bool = self.kwargs.get("use_ctxs_in_examplars", False)

        if examplar_type.lower() == "hotpotqa":
            from prompts.ircot.hotpotqa_demonstrations import ircot_hotpotqa_examplars
            examplars = ircot_hotpotqa_examplars
        elif examplar_type.lower() == "2wikimultihopqa":
            from prompts.ircot.wikimultihopqa_demonstrations import ircot_2wikimultihopqa_examplars
            examplars = ircot_2wikimultihopqa_examplars
        elif examplar_type.lower() == "musique":
            from prompts.ircot.musique_demonstrations import ircot_musique_examplars
            examplars = ircot_musique_examplars
        else:
            raise ValueError("{} is an unknown examplar_type! Current available examplar_type: {}".format(examplar_type, ["hotpotqa", "2wikimultihopqa", "musique"]))

        results = [] 
        for example in examplars[:num_examplars]:
            one_result = ""
            if use_ctxs:
                ctxs_list = ["title: {}, text: {}".format(ctx["title"], ctx["text"]) for ctx in example["ctxs"]]
                ctxs_str = "\n\n".join(["{}. {}".format(i+1, ctx) for i, ctx in enumerate(ctxs_list)])
                one_result += f"context:\n\n{ctxs_str}\n\n"
            one_result += "question: {}\n{} {}".format(example["question"], self.answer_prefix, example["answer"])
            results.append(one_result)
        
        return results

    def get_generator_inputs(
        self, 
        questions: List[str], 
        contexts: Optional[List[List[str]]]=None, 
        task_instructions: Optional[Union[str, List[str]]]=None, 
        **kwargs, 
    ) -> Tuple[List[str], List[str]]:
        
        # 得到instructions的顺序: task_instructions -> self.task_instruction -> self.task_instruction_wo/with_context
        if task_instructions is None:
            if self.task_instruction is not None:
                task_instructions = [self.task_instruction] * len(questions)
            else:
                if self.use_cot:
                    instruction = self.task_instruction_cot
                    instruction += "\n\nExamples:\n{}".format("\n\n".join(self.cot_examplars))
                else:
                    instruction = self.task_instruction_wo_context if contexts is None else self.task_instruction_with_context
                task_instructions = [instruction] * len(questions)
        """Prompt Format: 
        context:

        1. {context1}
        2. {context2}
        ...

        question: {question}
        {self.answer_prefix}
        """
        user_inputs = [] 
        for i, question in enumerate(questions):
            user_input = "" 
            if contexts is not None:
                context = contexts[i]
                context_text = "\n\n".join(["{}. {}".format(j+1, text) for j, text in enumerate(context)])
                user_input += f"context:\n\n{context_text}\n\n"
            user_input += f"question: {question}\n{self.answer_prefix}"
            user_inputs.append(user_input)
        
        return task_instructions, user_inputs

    def parse_generated_answers(self, texts: List[str]) -> List[str]:

        def parse_answer(answer: str) -> str:
            candidate_answers = answer.split("\n")
            answer = ""
            i = 0 
            while len(answer) < 1 and i<len(candidate_answers):
                answer = candidate_answers[i].strip()
                i += 1 
            if "answer is" in answer:
                idx = answer.find("answer is")
                answer = answer[idx+len("answer is"): ].strip()
                if answer.startswith(":"):
                    answer = answer[1:].strip()
            return answer
        
        return [parse_answer(text) for text in texts]

    def batch_generate_answers(
        self, 
        questions: List[str], 
        contexts: Optional[List[List[str]]]=None,
        task_instructions: Optional[Union[str, List[str]]]=None, 
        **kwargs, 
    ):
        if task_instructions is not None and isinstance(task_instructions, str):
            task_instructions = [task_instructions]*len(questions)
        if contexts is not None:
            assert len(questions) == len(contexts) # number of question and number of context list must be equal 
        if task_instructions is not None:
            assert len(questions) == len(task_instructions) # number of questions and number of task instructions must be equal 
        
        instructions, user_inputs = self.get_generator_inputs(
            questions=questions, 
            contexts=contexts, 
            task_instructions=task_instructions,
            **kwargs
        )
        generated_token_ids, _ = self.generator_generate(
            instructions=instructions, 
            inputs=user_inputs, 
            **kwargs
        )
        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        # print("Generated Text: {}".format(generated_texts))
        # print(generated_token_ids)
        answers = self.parse_generated_answers(generated_texts)

        return answers
    
    def generate_answer(
        self, 
        question: Union[List[str], str], 
        context: Optional[Union[List[List[str]], List[str]]]=None,
        task_instruction: Optional[Union[str, List[str]]]=None, 
        **kwargs, 
    ):
        single_question = False
        if isinstance(question, str):
            single_question = True
            question=[question]
            context = [context] if context is not None else None
        answers = self.batch_generate_answers(
            questions=question, 
            contexts=context, 
            task_instructions=task_instruction,
            **kwargs
        )
        results = answers[0] if single_question else answers

        return results


class AnswerGeneratorChatFormat(Generator):

    def __init__(self, tokenizer: AutoTokenizer, generator: AutoModelForCausalLM, max_length: int = 4096, max_new_tokens: int = 128, batch_size: int = 4, **kwargs):
        
        super().__init__(tokenizer, generator, max_length, max_new_tokens, batch_size, **kwargs)
        self.task_instruction_wo_contexts = "Given a question, please only output the answer to the question."
        self.task_instruction_with_contexts = "Given some contexts and a question, please only output the answer to the question."

    def get_generator_inputs(
        self, 
        questions: List[str], 
        contexts: Optional[List[List[Any]]]=None, 
        task_instruction: Optional[str]=None, 
        **kwargs, 
    ) -> Tuple[List[str], List[List[dict]]]:
        
        if contexts is not None:
            assert len(questions) == len(contexts) # number of questions and contexts must be equal 
        
        if task_instruction is None:
            task_instruction = self.task_instruction_wo_contexts if contexts is None else self.task_instruction_with_contexts
        instructions = [task_instruction] * len(questions)

        messages = [] 
        for i, question in enumerate(questions):
            input_text = ""
            if contexts is not None:
                context = contexts[i]
                context_text = "\n".join(["{}. {}".format(j+1, text) for j, text in enumerate(context)])
                input_text += f"context:\n{context_text}\n"
            input_text += f"question: {question}\nthe correct answer is:"
            # messages.append([{"user": input_text}])
            messages.append(input_text)
        
        return instructions, messages
        
    def generate(
        self, 
        questions: Union[str, List[str]], 
        contexts: Union[List[Any], List[List[Any]]]=None, 
        **kwargs
    ) -> Union[str, List[str]]:
        
        single_question = False
        if isinstance(questions, str):
            single_question = True
            questions = [questions]
            contexts = [contexts] if contexts is not None else None
        
        instructions, messages = self.get_generator_inputs(questions, contexts)
        prompts_chat_format = self.get_generator_prompts_chat_format(instructions, messages)
        generator_inputs = self.tokenizer_encode_chat_format(prompts_chat_format)
        # generated_token_ids, _ = self.greedy_generate(inputs=generator_inputs, **kwargs)
        generated_token_ids = super().generate(generator_inputs, **kwargs)[0]
        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

        if single_question:
            return generated_texts[0]
        else:
            return generated_texts
    
    def calculate_answers_prob(
        self, 
        questions: Union[str, List[str]], 
        answers: Union[str, List[str]], 
        contexts: Union[List[Any], List[List[Any]]]=None,
        **kwargs
    ) -> Tensor:
        
        single_question = False
        if isinstance(questions, str):
            single_question = True
            questions = [questions]
            answers = [answers]
            contexts = [contexts] if contexts is not None else None
        
        if contexts is None:
            task_instruction = "Given a question, please only output the answer (MUST use lowercase) to the question."
        else:
            task_instruction = "Given some contexts and a question, please only output the answer (MUST use lowercase) to the question."
        
        instructions, messages = self.get_generator_inputs(
            questions=questions, contexts=contexts, task_instruction=task_instruction
        )
        prompts_chat_format = self.get_generator_prompts_chat_format(instructions, messages)
        # NOTE: 下面的代码要求tokenizer是left padding的, 不然就会出错
        inputs_wo_answers = self.tokenizer_encode_chat_format(prompts_chat_format)
        # prompts_wo_answers = self.tokenizer.apply_chat_template(prompts_chat_format, tokenize=False, add_generation_prompt=True)
        # inputs_wo_answers = self.tokenizer(prompts_wo_answers, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        prompts_wo_answers_lengths = inputs_wo_answers["attention_mask"].sum(1).tolist()

        inputs_with_answers = {k: torch.zeros((len(questions), self.max_length), dtype=torch.long) for k, _ in inputs_wo_answers.items()}
        answers_lengths = torch.zeros((len(questions), ), dtype=torch.long)

        for i in range(len(questions)):
            answer_token_ids = self.tokenizer(answers[i].lower(), add_special_tokens=False)["input_ids"]
            num_answer_tokens = len(answer_token_ids)
            answers_lengths[i] = num_answer_tokens

            inputs_with_answers["input_ids"][i, -num_answer_tokens: ] = torch.tensor(answer_token_ids, dtype=torch.long)
            inputs_with_answers["attention_mask"][i, -num_answer_tokens: ] = 1 
            inputs_with_answers["input_ids"][i, -num_answer_tokens-prompts_wo_answers_lengths[i]: -num_answer_tokens] = \
                inputs_wo_answers["input_ids"][i, -prompts_wo_answers_lengths[i]:]
            inputs_with_answers["attention_mask"][i, -num_answer_tokens-prompts_wo_answers_lengths[i]: -num_answer_tokens] = 1 
        
        # truncate tensors 
        start_idx = int(inputs_with_answers["attention_mask"].sum(0).nonzero(as_tuple=True)[0][0])
        inputs_with_answers = {k: v[:, start_idx:] for k, v in inputs_with_answers.items()}
        
        """
        prompts_wo_answers = [] 
        for question, context in zip(questions, contexts):
            context_text = "\n".join(["{}. {}".format(i+1, text) for i, text in enumerate(context)])
            prompt = task_instruction + "\n\n" + "context:\n{}\n".format(context_text) + "question: {}\nthe correct answer is:".format(question)
            prompts_wo_answers.append(prompt)
        # from pdb import set_trace; set_trace()
        inputs_wo_answers = self.tokenizer(prompts_wo_answers, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        prompts_wo_answers_lengths = inputs_wo_answers["attention_mask"].sum(1)
        prompts_with_answers = [prompt+" " + answer.lower() for prompt, answer in zip(prompts_wo_answers, answers)]
        inputs_with_answers = self.tokenizer(prompts_with_answers, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        prompts_with_answers_lengths = inputs_with_answers["attention_mask"].sum(1)
        answers_lengths = prompts_with_answers_lengths - prompts_wo_answers_lengths
        """

        logits_list = [] 
        for i in range((len(questions)-1)//self.batch_size+1):
            batch_inputs = {k: v[i*self.batch_size: (i+1)*self.batch_size] for k, v in inputs_with_answers.items()}
            batch_attention_mask = batch_inputs["attention_mask"]
            batch_inputs["position_ids"] = batch_attention_mask.long().cumsum(-1) - 1 
            batch_inputs["position_ids"].masked_fill_(batch_attention_mask==0, 1)
            batch_inputs = to_device(batch_inputs, self.device)
            logits_list.append(self.generator(**batch_inputs).logits.detach().cpu())
        logits = torch.cat(logits_list, dim=0)

        answers_logprob = torch.zeros((len(questions), ), dtype=torch.float32)
        for i in range(len(questions)):
            token_logprobs = F.log_softmax(logits[i, -1-answers_lengths[i]:-1], dim=-1)
            answer_token_ids = inputs_with_answers["input_ids"][i, -answers_lengths[i]:]
            answer_token_logprob = token_logprobs.gather(1, answer_token_ids.view(-1, 1))
            answers_logprob[i] = torch.sum(answer_token_logprob) / answers_lengths[i]
        answers_prob = torch.exp(answers_logprob)

        if single_question:
            return answers_prob[0]
        else:
            return answers_prob


if __name__ == "__main__":

    from utils.pipeline_utils import load_llm_tokenizer_and_model

    tokenizer, model = load_llm_tokenizer_and_model("llama3")

    # generator = Generator(tokenizer, model)
    # prompts = generator.get_generator_prompts_chat_format(["output answer to the question"]*2, [[{"user": "what is the capital city of China?"}], [{"user": "What is the capital city of UK?"}]])

    # generator.get_generator_prompts_chat_format(["output answer to the question"]*2, ["what is the capital city of China?", "What is the capital city of UK?"])

    # inputs = generator.tokenizer_encode_chat_format(prompts) 
    # token_ids, token_logits = generator.generate(inputs, batch_size=4, stop_words=list('!@#$%^&*()\n\n)(*&^%$#@!'))

    # tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
    # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    # generator = Generator(tokenizer, model)

    questions = ["what is the capital city of China?", "Where is London located?"]
    contexts=[
        ["Beijing is the center of China.", "London is the capital of United Kingdom (UK)."], 
        ["Beijing is the center of China.", "London is the capital of United Kingdom (UK)."]
    ]

    generator = AnswerGenerator(tokenizer=tokenizer, generator=model) # use_cot=True, examplar_type="musique", num_examplars=5, use_ctxs_in_examplars=True)
    answers = generator.generate_answer(question=questions, context=contexts, max_new_tokens=10)
    from pdb import set_trace; set_trace()
    a = 0

    # inputs = generator.tokenizer_encode(prompts)
    # token_ids, token_logits = generator.generate(inputs, stop_words=["capital"])

    # generator = AnswerGeneratorChatFormat(tokenizer, model)
    # instructions, messages = generator.get_generator_inputs(questions=["what is the capital city of China?"])
    # instructions, messages = generator.get_generator_inputs(
    #     questions=["what is the capital of China?"],
    #     contexts=[["Beijing is the center of China.", "London is the capital of United Kingdom (UK)."]]
    # )
    # instructions, messages = generator.get_generator_inputs(
    #     questions=["what is the capital of China?"],
    #     contexts=[["Beijing is the center of China.", "London is the capital of United Kingdom (UK)."]],
    #     answers=["Beijing"]
    # )
    # instructions, messages = generator.get_generator_inputs(
    #     questions=["what is the capital of China?"],
    #     contexts=[["Beijing is the center of China.", "London is the capital of United Kingdom (UK)."]],
    #     answers=["Beijing"], task_instruction="answer the question."
    # )
    # generator.generate(questions="what is the capital of China?", contexts=["Beijing is the center of China.", "London is the capital of United Kingdom (UK)."])
    # generator.calculate_answers_prob(questions="what is the capital of China?", answers="Beijing") #, contexts=["Beijing is the center of China.", "London is the capital of United Kingdom (UK)."])
