from dataclasses import dataclass
from typing import List, Union

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from setup.setup import OPENAI_API_KEY
from litellm import token_counter, cost_per_token


@dataclass
class Cost:
    input_tokens: int 
    output_tokens: int
    input_cost: float 
    output_cost: float


class CostManager:

    def __init__(self):

        self.total_input_tokens = {}
        self.total_output_tokens = {} 
        self.total_tokens = {} 

        self.total_input_cost = {}
        self.total_output_cost = {}
        self.total_cost = {}

    def compute_total_cost(self):
        total_tokens, total_cost = 0, 0.0
        for _, value in self.total_tokens.items():
            total_tokens += value
        for _, value in self.total_cost.items():
            total_cost += value
        return total_tokens, total_cost

    def update_cost(self, cost: Cost, model: str):

        self.total_input_tokens[model] = self.total_input_tokens.get(model, 0) + cost.input_tokens
        self.total_output_tokens[model] = self.total_output_tokens.get(model, 0) + cost.output_tokens
        current_total_tokens = cost.input_tokens + cost.output_tokens
        self.total_tokens[model] = self.total_tokens.get(model, 0) + current_total_tokens

        self.total_input_cost[model] = self.total_input_cost.get(model, 0.0) + cost.input_cost
        self.total_output_cost[model] = self.total_output_cost.get(model, 0.0) + cost.output_cost
        current_total_cost = cost.input_cost + cost.output_cost
        self.total_cost[model] = self.total_cost.get(model, 0.0) + current_total_cost
        
        total_tokens, total_cost = self.compute_total_cost()
        print(f"Total cost: ${total_cost:.3f} | Total tokens: {total_tokens} | Current cost: ${current_total_cost:.3f} | Current tokens: {current_total_tokens}")


cost_manager = CostManager()


class OpenAIGPTGenerator:

    def __init__(self, model_name: str="gpt-4o-mini", max_new_tokens: int=256, num_outputs: int=1, **kwargs):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.num_outputs = num_outputs
        self.kwargs = kwargs

    def get_openaigpt_messages(self, instruction: str, user_inputs: Union[str, List[dict]]):
        messages = [{"role": "system", "content": instruction}]
        if isinstance(user_inputs, str):
            messages.append({"role": "user", "content": user_inputs})
            return messages
        for input_item in user_inputs:
            if "user" in input_item:
                messages.append({"role": "user", "content": input_item["user"]})
                continue
            if "assistant" in input_item:
                messages.append({"role": "assistant", "content": input_item["assistant"]})
        return messages

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(
        self, 
        instruction: str, 
        user_inputs: Union[str, List[dict]], 
        temperature: int = None,
        max_new_tokens: int = None,
        **kwargs
    ):
        """
        只能针对一个问题进行生成
        Input:
            instruction: 系统instruction(str)
            user_inputs: 可以直接是需要回答的问题(str), 或者是聊天的记录,以"user"和"system"表示是用户的输入还是模型的输出最后一个是需要回答的问题(List[str])
        Output:
            output: [[answer]*self.num_outputs] 一个或多个的回答
        """
        messages = self.get_openaigpt_messages(instruction=instruction, user_inputs=user_inputs)
        max_new_tokens = max_new_tokens or self.max_new_tokens 
        responses = self.client.chat.completions.create(
            model = self.model_name, 
            messages = messages, 
            temperature = temperature, 
            max_tokens = max_new_tokens,
            n = self.num_outputs,
        )
        outputs = [] 
        for choice in responses.choices:
            content = choice.message.content
            outputs.append(content)
        cost = self._completion_cost(response=responses)
        self._update_cost(cost=cost)
        return outputs
    
    def _completion_cost(self, response) -> Cost:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return self._compute_cost(input_tokens=input_tokens, output_tokens=output_tokens)
    
    def _compute_cost(self, input_tokens: int, output_tokens: int) -> Cost:
        # use LiteLLM to compute cost, require the model name to be a valid model name in LiteLLM.
        input_cost, output_cost = cost_per_token(
            model=self.model_name, 
            prompt_tokens=input_tokens, 
            completion_tokens=output_tokens, 
        )
        cost = Cost(input_tokens=input_tokens, output_tokens=output_tokens, input_cost=input_cost, output_cost=output_cost)
        return cost
    
    def _update_cost(self, cost: Cost):
        cost_manager.update_cost(cost=cost, model=self.model_name)

    
    
if __name__ == "__main__":

    generator = OpenAIGPTGenerator()

    separation = "(in the form of <head entity; relationship; tail entity> that describes the relationship between head entity and tail entity)"
    task_instruction = f"You are given a question and a list of knowledge triples {separation}. Your task is to generate a follow-up question based on the initial question and the provided knowledge triples. "\
        "The follow-up question should aim to retrieve additional information necessary for answering the initial question. Let's think step-by-step:\n"\
            "(1) Analyze the steps required to answer the initial question (using the information from initial question only), you MUST use the format \"Analyses:\" to output the analyzed steps.\n"\
                "(2) Investigate whether each triple can answer the previously analyzed steps, you MUST use the format \"Relevance:\" to output the relevance of each triple.\n"\
                    "(3) Analyze the unsolved steps, you MUST use the format \"Unsolved steps\" to output unsolved steps.\n"\
                        "(4) Generate follow-up question, you MUST use the format \"So the follow-up question is:\" to output the follow-up question. If existing triples are sufficient to answer the question, output \"So the follow-up question is: no additional follow-up question is needed.\""\
            
    examplars = [
        # single-hop example
        {
            "question": "Where was Georg Friedrich Haas born?", 
            "triples": ["<Georg Friedrich Haas; place of birth; Graz, Austria>"],
            # "output": "The first knowledge triple is relevant because it reveals that Georg Friedrich Haas was born in Graz, Austria. The reformulated question is: Georg Friedrich Haas was born in Graz, Austria."
            "output": "Analyses: (1) Identify the birthplace of Georg Friedrich Haas.\nRelevance: The first triple indicates that the birthplace of Georg Friedrich Haas was Graz, Austria, which can directly answer the question.\nUnsolved steps: existing triples are sufficient to answer the question. There are no unsolved steps.\nSo the follow-up question is: no additional follow-up question is needed.", 
        },
        # two-hop example 
        {
            "question": "What are three alternate names of the University where W.G. Godfrey received his BA and MA degrees?", 
            "triples": ["<W. G. Godfrey; education; BA and MA from the University of Waterloo, Ph.D. from Queen's University>"],
            "output": "Analyses: (1) Identify the university where W.G. Godfrey received his BA and MA degrees. (2) Determine the alternative names of this university.\nRelevance: The first triple indicates that W.G. Godfrey received his BA and MA degrees from the University of Waterloo, answering the first step.\nUnsolved steps: (1) Determine the alternative names for the University of Waterloo.\nSo the follow-up question is: what are three alternate names of the University of Waterloo?",
        },
        # three-hop example 
        # {
        #     "question": "Who distributed a movie whose cast includes an actor who acted in \"Him\"?",
        #     "triples": ["<Fionn Whitehead; notable work; \"Him\" (2016 ITV miniseries)>", "<Dunkirk (2017 film); cast; Fionn Whitehead, Mark Rylance, Tom Hardy>", "<Dunkirk (2017 film); director; Christopher Nolan>"], 
        #     # "output": "The first and second knowledge triples are relevant because the first triple indicates that the actor who acted in \"Him\" is Fionn Whitehead and the second triple reveals that Dunkirk is a movie whose cast include Fionn Whitehead. The reformulated question is: Who distributed the movie Dunkirk (2017 film)?",
        #     "output": "Analyses: (1) Identify an actor who acted in \"Him\". (2) Identify a movie that include this actor in its cast. (3) Determin the distributor of that movie.\nRelevance: The first triple indicates that Fionn Whitehead acted in \"Him\", answering the first step. The second triple indicates that Dunkirk (2017 film) is a movie whose cast include Fionn Whitehead, answersing the second step. The third triple provides information about the director of \"Dunkik\" but does not provide information about its distributor.\nUnsolved steps: (1) Determine the distributor of \"Dunkirk (2017 film)\".\nSo the follow-up question is: who distributed Dunkirk (2017 film)?",
        # },
        # # comparison example
        {
            "question": "Which magazine published papers more often; The Wittenburg Door or Sports Collectors Digest?", 
            "triples": ["<The Wittenburg Door; type; Christian satire and humor magazine>", "<Sports Collectors Digest; type; American advertising weekly paper>"],
            # "output": "The second knowledge triple is relevant because it shows that Sports Collectors Digest published weekly. The reformulated question is: Which magazine published papers more often; The Wittenburg Door or Sports Collectors Digest that published weekly?",
            "output": "Analyses: (1) Identify the publication frequency of \"The Wittenburg Door\". (2) Identify the publication frequency of \"Sports Collectors Digest\". (3) Compare the publication frequencies to determine which magazine published papers more often.\nRelevance: The first triple describes the type of maganize but does not provide information about its publication frequency. The second triple indicates that \"Sports Collectors Digest\" publihsed weekly, which addresses the second step.\nUnsolved steps: (1) Determine the publication frequency of \"The Wittenburg Door\". (2) Confirm and compare the publication frequencies of both magazines.\nSo the follow-up question is: What is the publication frequency of \"The Wittenburg Door\"?",
        },
    ]
    
    user_inputs = []
    for example in examplars:
        user = "Question: {}\nKnowledge Triples: {}".format(example["question"], ", ".join(example["triples"]))
        user_inputs.append({"user": user})
        user_inputs.append({"system": example["output"]})
    
    q = "Who distributed a movie whose cast includes an actor who acted in \"Him\"?"
    ts = ["<Fionn Whitehead; notable work; \"Him\" (2016 ITV miniseries)>", "<Dunkirk (2017 film); cast; Fionn Whitehead, Mark Rylance, Tom Hardy>", "<Dunkirk (2017 film); director; Christopher Nolan>"]
    user_inputs.append({"user": "Question: {}\nKnowledge Triples: {}".format(q, ", ".join(ts))})
    # example_user_input = "Question: {}\nKnowledge Triples: {}\n".format(examplars[-1]["question"], ", ".join(examplars[-1]["triples"]))
    generator.generate(
        instruction=task_instruction, 
        user_inputs=user_inputs, 
        # user_inputs=example_user_input
    )