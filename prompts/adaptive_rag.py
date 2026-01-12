
ANSWER_PROMPT = """Please answer the following question. You should think step-by-step to solve it. 

Provide your final answer in the format \\boxed{{{{YOUR_ANSWER}}}}.

Context: 
{{context}}

Question: {question}
"""

ANSWER_PROMPT_OPENROUTER = """Please answer the following question. You should think step-by-step to solve it. 

Provide your final answer in the format \\boxed{{{{YOUR_ANSWER}}}}.

NOTE: You might be provided with some of your previous thoughts. Do not repeat your previous thoughts, you should continue the thinking process towards answering the question. 

Context: 
{{context}}

Question: {question} 

<think>
"""

# Summarize your final long-form answer in the format \\boxed{{{{YOUR_LONG_FORM_ANSWER}}}}.

ANSWER_PROMPT_LONGFORMQA = """Answer the following ambiguous question, and output a long-form answer to the question. You should think step-by-step to solve it. 

Context: 
{{context}}

Question: {question}
"""


ANSWER_PROMPT_FOR_NON_REASONING_MODEL = """Please answer the following question. You should think step-by-step to solve it. 

You should first output all your internal thinking steps. Each thought shoud be put inside <think> and </think> tags. After reasoning, you must put your final answer inside <answer> and </answer> without detailed instructions.

Your output should look like: 
<think> [put your first thought here] </think>
<think> [put your second thought here] </think> 
... 
<think> [put your final thought here] </think>
<answer> [put your final answer (short phrase or keyword only) here] </answer>

---
Context: 
{{context}}

Question: {question}
"""

ANSWER_PROMPT_FOR_NON_REASONING_MODEL_OPENROUTER = """Please answer the following question. You should think step-by-step to solve it. 

You should first output all your internal thinking steps. Each thought shoud be put inside <think> and </think> tags. After reasoning, you must put your final answer inside <answer> and </answer> without detailed instructions. 

Your output should look like: 
<think> [put your first thought here] </think>
<think> [put your second thought here] </think> 
... 
<think> [put your final thought here] </think>
<answer> [put your final answer (short phrase or keyword only) here] </answer>

NOTE: You might be provided with some of your previous thoughts:
   - Do not repeat your previous thoughts, you should continue the thinking process towards answering the question. 
   - Or if you think the final answer can be derived from previous thoughts, you can directly output it within the <answer> and </answer> tags. 

---
Context: 
{{context}}

Question: {question}
"""

ANSWER_PROMPT_FOR_NON_REASONING_MODEL_WO_RETRIEVAL_DECISION_AGENT = """Please answer the following question. You should think step-by-step to solve it. 

You should first output all your internal thinking steps. Each thought shoud be put inside <think> and </think> tags. If you require external knowledge, output exactly '<search> retrieval </search>' to trigger a retrieval process. After reasoning, you must put your final answer inside <answer> and </answer> without detailed instructions.

Your output should look like: 
<think> [put your first thought here] </think>
<search> retrieval </search>
<think> [put your second thought here] </think> 
... 
<think> [put your final thought here] </think>
<answer> [put your final answer (short phrase or keyword only) here] </answer>

---
Context: 
{{context}}

Question: {question}
"""

# You should first output all your internal thinking steps. Each thought shoud be put inside <think> and </think> tags.
ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA = """Answer the following ambiguous question, and output a long-form answer to the question.

You should output all your internal thinking steps in a logical, structured, and detailed manner. Each thought should be wrapped in <think> and </think> tags. Your thoughts can include background knowledge, assumptions, logical deductions, or contextual interpretation to support a long-form, comprehensive answer. 

After reasoning, you must summarise your final long-form answers within <answer> xxx </answer> tags. 

Your output should look like: 
<think> [put your first thought here] </think>
<think> [put your second thought here] </think>
... 
<think> [put your final thought here] </think>
<answer> [summarize your final long-form answer here] </answer>

---
Context: 
{{context}}

Question: {question}
"""

ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA_OPENROUTER = """Answer the following ambiguous question, and output a long-form answer to the question.

You should output all your internal thinking steps in a logical, structured, and detailed manner. Each thought should be wrapped in <think> and </think> tags. Your thoughts can include background knowledge, assumptions, logical deductions, or contextual interpretation to support a long-form, comprehensive answer. 

After reasoning, you must summarise your final long-form answers within <answer> xxx </answer> tags. 

Your output should look like: 
<think> [put your first thought here] </think>
<think> [put your second thought here] </think>
... 
<think> [put your final thought here] </think>
<answer> [summarize your final long-form answer here] </answer>

NOTE: You might be provided with some of your previous thoughts. Do not repeat your previous thoughts, you should continue the thinking process towards answering the question. 

---
Context: 
{{context}}

Question: {question}
"""


ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA_WO_RETRIEVAL_DECISION_AGENT = """Answer the following ambiguous question, and output a long-form answer to the question.

You should output all your internal thinking steps in a logical, structured, and detailed manner. Each thought should be wrapped in <think> and </think> tags. Your thoughts can include background knowledge, assumptions, logical deductions, or contextual interpretation to support a long-form, comprehensive answer.

If you require external knowledge, output exactly '<search> retrieval </search> to trigger a retrieval process. 

After reasoning, you must summarise your final long-form answers within <answer> xxx </answer> tags. 

Your output should look like: 
<think> [put your first thought here] </think>
<think> [put your second thought here] </think>
... 
<think> [put your final thought here] </think>
<answer> [summarize your final long-form answer here] </answer>

---
Context: 
{{context}}

Question: {question}
"""

# ANSWER_PROMPT_FOR_NON_REASONING_MODEL_LONGFORMQA = """Please answer the following ambiguous question. 

# You should identify all possible interpretations of the question and provide corresponding answers through thinking step-by-step. Output all your internal thinking steps for addressing each possible interpretation. Each thought shoud be put inside <think> and </think> tags. If you need additional context, clearly say so in your thought. After exploring all interpretations, you must **Summarize** a final long-form answer within <answer> and </answer> tags. 

# Your output should look like:
# <think> [put possible interpretation 1 here] </think>
# <think> [put your reasoning step 1 for interpretation 1 here] </think>
# <think> [put your reasoning step 2 for interpretation 1 here] </think>
# ...
# <think> [put possible interpretation 2 here] </think>
# <think> [put your reasoning step 1 for interpretation 2 here] </think>
# <think> [put your reasoning step 2 for interpretation 2 here] </think>
# ...
# <answer> [put final integrated long-form answer addressing all interpretations here] </answer>

# ---
# Context: 
# {{context}}

# Question: {question}
# """


# >>>>>>>>>>>>>>>>>
# Determine Retrieval 
# >>>>>>>>>>>>>>>>>

DETERMINE_RETRIEVAL_INSTRUCTION = """You are an intelligent assistant. Your job is to decide whether external information retrieval is required to continue solving the problem.

You will be given: a question and the current reasoning progress (thought). 

Your decision should be based on the following rules:

[When to retrieve]
- If the current thought does not contain enough information to answer the question.
- If the model shows uncertainty, hesitation, or guesses in its reasoning. For example, if the thought contains words or phrases such as: "maybe", "I think", "probably", "not sure", "wait", "let me think", or other signs of hesitation.
- If external facts, names, dates, or specific knowledge are clearly needed to proceed.

[When not to retrieve]
- If the current thought already contains sufficient, relevant information to answer the question confidently.
- If the question is general knowledge or can be solved with basic reasoning based on what's already known.
- If the model is clearly confident and reaches a complete conclusion.

Your answer should be either "Yes" (retrieval is needed) or "No" (retrieval is not needed). 
Only output your answer and do not include any other text.
"""

DETERMINE_RETRIEVAL_DEMO = """--- 
Example 1:

Question: Who distributed a movie whose cast includes an actor who acted in 'Him'?

Thought so far: 
Alright, so I have this question: "Who distributed a movie whose cast includes an actor who acted in 'Him'?" Hmm, okay, let me try to break this down.

Do you need to retrieve external information?

Answer: Yes 

### Example 2: 

Question: Who distributed a movie whose cast includes an actor who acted in 'Him'? 

Thought so far: 
Alright, so I have this question: "Who distributed a movie whose cast includes an actor who acted in 'Him'?" Hmm, okay, let me try to break this down.

First, I need to understand what the question is asking. It's asking about a movie distributor. The movie in question has a cast that includes an actor who was in "Him." So, I think the steps are: find out who the actors in "Him" are, then find a movie they were in, and then determine who distributed that movie.

Do you need to retrieve external information?

Answer: Yes

### Example 3:

Question: Which film has the director who was born earlier, Ciguli Miguli or Last Hurrah For Chivalry? 

Thought so far: 
Okay, so I have to figure out which film has the director who was born earlier: Ciguli Miguli or Last Hurrah For Chivalry. Hmm, I'm not super familiar with these movies, but I'll try to break it down.

First, I need to find out who directed each of these films. Let's start with "Ciguli Miguli." I'm not sure who directed that, but maybe I can look it up. Wait, I think it's an Italian film, right? I recall that "Ciguli Miguli" is a 1995 movie. The director's name might be... I'm not sure. Maybe it's someone like Federico Veirojga? I think I've heard that name before in relation to Italian cinema.

Do you need to retrieve external information?

Answer: Yes"""

DETERMINE_RETRIEVAL_INPUTS = """---
Let's begin.

Question: {question}

Thought so far:
{thought}

Do you need to retrieve external information?

Answer: """

DETERMINE_RETRIEVAL_PROMPT_WO_DEMO = DETERMINE_RETRIEVAL_INSTRUCTION + "\n\n" + DETERMINE_RETRIEVAL_INPUTS

DETERMINE_RETRIEVAL_PROMPT = DETERMINE_RETRIEVAL_INSTRUCTION + "\n\n" + DETERMINE_RETRIEVAL_DEMO + "\n\n" + DETERMINE_RETRIEVAL_INPUTS

DETERMINE_RETRIEVAL_INSTRUCTION_WITH_THOUGHT = """You are an intelligent assistant. Your job is to decide whether external information retrieval is required to continue solving the problem.

You will be given: a question and the current reasoning progress ('thought so far'). 

Follow the rules below:

[When to retrieve]
- If the current thought does not contain enough information to answer the question.
- If the model shows uncertainty, hesitation, or guesses in its reasoning (e.g. “maybe”, “I think”, “not sure”…).
- If specific external facts (names, dates, definitions) are clearly needed.

[When not to retrieve]
- If the current thought already contains sufficient, relevant information to answer the question confidently.
- If basic reasoning over the existing content is enough.
- If the model sounds confident and has reached a complete conclusion.

**Important**
Do not use your own world knowledge to assess whether the information is correct or sufficient. 
Only judge based on how the thought is written — its language, confidence, and completeness. 

**Output format**

### Thought
<your step-by-step reasoning about whether retrieval is needed, based only on the language of the thought so far>

### Answer
<Yes or No>

Write nothing else outside this exact format."""

DETERMINE_RETRIEVAL_DEMO_WITH_THOUGHT = """---
Example 1:

Question: Who distributed a movie whose cast includes an actor who acted in 'Him'?

Thought so far: 
Alright, so I have this question: "Who distributed a movie whose cast includes an actor who acted in 'Him'?" Hmm, okay, let me try to break this down.

Do you need to retrieve external information?

### Thought
The thought so far does not include any concrete information about the file 'Him', its actors, or its distributors. 
No specific knowledge is present yet, so retrieval is needed to proceed.

### Answer
Yes 

### Example 2: 

Question: Who distributed a movie whose cast includes an actor who acted in 'Him'? 

Thought so far: 
Alright, so I have this question: "Who distributed a movie whose cast includes an actor who acted in 'Him'?" Hmm, okay, let me try to break this down.

First, I need to understand what the question is asking. It's asking about a movie distributor. The movie in question has a cast that includes an actor who was in "Him." So, I think the steps are: find out who the actors in "Him" are, then find a movie they were in, and then determine who distributed that movie.

Do you need to retrieve external information?

### Thought
The reasoning outlines a plan involving multiple steps, but it does not provide any actual answers. 
No actor from "Him" is identified, and no further factual progress is made. 
Therefore, external retrieval is necessary to move forward with the reasoning.

### Answer
Yes

### Example 3:

Question: Which film has the director who was born earlier, Ciguli Miguli or Last Hurrah For Chivalry? 

Thought so far: 
Okay, so I have to figure out which film has the director who was born earlier: Ciguli Miguli or Last Hurrah For Chivalry. Hmm, I'm not super familiar with these movies, but I'll try to break it down.

First, I need to find out who directed each of these films. Let's start with "Ciguli Miguli." I'm not sure who directed that, but maybe I can look it up. Wait, I think it's an Italian film, right? I recall that "Ciguli Miguli" is a 1995 movie. The director's name might be... I'm not sure. Maybe it's someone like Federico Veirojga? I think I've heard that name before in relation to Italian cinema.

Do you need to retrieve external information?

### Thought
The reasoning includes multiple signals of uncertainty such as “not sure”, “maybe”, and guesses about the director's name.  
It lacks concrete facts about either film's director or their birth dates.  
Based on this, retrieval is required to continue.

### Answer
Yes"""


DETERMINE_RETRIEVAL_INPUTS_WITH_THOUGHT = """---
Let's begin.

Question: {question}

Thought so far:
{thought}

Do you need to retrieve external information?"""

DETERMINE_RETRIEVAL_INSTRUCTION_WITH_CONTEXT = """You are an intelligent assistant. Your job is to decide whether external information retrieval is required to continue solving the problem. 

You will be given the following inputs:
- Question: The original user query.  
- Previous Reasoning: A list of prior reasoning thoughts the model has generated.
- Current Reasoning Step: The latest reasoning step to evaluate. 
- Retrieved Context: Text previously retrieved that may contain relevant information.

Your task is to answer whether the **current reasoning step** requires external retrieval.

---
### Respond **Yes (retrieval is required)** if any of these conditions is met:
1. **Missing entity/fact**: The current reasoning step requires, asks for, references, or depends on a specific entity / date / location / definition / fact, etc that is not already present (even paraphrased) in **Retrieved Context** or **Previous Reasoning** thought.
2. **Novel entity**: The current reasoning step mentions at least one new entity / date / number / title absent from prior context.
3. **Uncertainty signals**: The current reasoning step express uncertainty ("maybe", "I think", "probably", "not sure", "I guess", "unknown", "not provided", "[?]") **AND** requires knowledge that is not found in the **Previous Reasoning** steps or the **Retrieved Context**.
4. **Verifiability**: The missing fact cannot be safely inferred from general world knowledge or logic (e.g., arithmetic, common definitions).

### Respond **No(retrieval is not required) in the following common situations:
1. The information required in the current reasoning step is already present in or can be inferred from **Retrieved Context**.
2. The current reasoning step merely repeats the question, summarises, rephrases or draws a deduction from existing context.
3. The information requested is text-book general knowledge the model is expected to know (physical constants, dictionary definitions, etc.).
4. The step uses exploratory verbs ("let's see", "I think") but no new entity is introduced.

### Tips:
1. Synonyms / abbreviations count as already-present facts (e.g., William II vs Wm. II). 
2. If an uncertainty word appears but the required knowledge is in **Retrieved Context**, answer "No (retrieval is not required)" -- uncertainty alone is insufficient.
3. When unsure, first scan the context; if the needed fact is absent, answer "Yes (retrieval is required)". 

--- 
### Output Format 
ONLY output "Yes (retrieval is required)" or "No (retrieval is not required)" and do not include any other text.
"""

DETERMINE_RETRIEVAL_INSTRUCTION_WITH_CONTEXT_SIMPLIFIED = "Determine if the current reasoning step requires external information retrieval, output \"Yes (retrieval is required)\" or \"No (retrieval is not required)\"."

DETERMINE_RETRIEVAL_DEMO_WITH_CONTEXT = """---
### Example 1: 
Retrieved Context:

Question:
Which label released the album that includes the song "Make Me..."?

Previous Reasoning:  
I think "Make Me..." is a song by Britney Spears from her ninth studio album. 

Current Reasoning Step: 
I should double-check which record label released it.

Output:
Yes (retrieval is required)

### Example 2:
Retrieved Context:
"Make Me…" is a song by Britney Spears from her 2016 album "Glory", released under RCA Records.

Question:
Which label released the album that includes the song "Make Me..."?

Previous Reasoning:
I think "Make Me..." is by Britney Spears, and it came out around 2016.

Current Reasoning Step:
"Make Me..." is a song by Britney Spears from her 2016 album "Glory", released under RCA Records.

Output:
No (retrieval is not required)

### Example 3:
Retrieved Context:
The 56th episode of The Walking Dead is titled "Self Help" and features the character Abraham prominently.

Question:
Who does Lauren Cohan play in the 56th episode of the post-apocalyptic horror TV series "Self Help"?

Previous Reasoning:
I know Lauren Cohan appeared in The Walking Dead.

Current Reasoning Step:
I think she played Maggie Grimes, but I might be confusing that name. 

Output:
Yes (retrieval is required)
"""

DETERMINE_RETRIEVAL_INPUTS_WITH_CONTEXT = """---
Retrieved Context:  
{retrieved_context}

Question:  
{question}

Previous Reasoning:  
{reasoning_history}

Current Reasoning Step: 
{current_reasoning_step}

Output:
"""

DETERMINE_RETRIEVAL_PROMPT_WITH_CONTEXT = DETERMINE_RETRIEVAL_INSTRUCTION_WITH_CONTEXT + "\n\n" + DETERMINE_RETRIEVAL_DEMO_WITH_CONTEXT + "\n\n" + DETERMINE_RETRIEVAL_INPUTS_WITH_CONTEXT 

# >>>>>>>>>>>>>>>>>
# Query Formulation 
# >>>>>>>>>>>>>>>>>

QUERY_FORMULATION_INSTRUCTION = """You are an intelligent reasoning assistant. Your task is to help bridge the information gap between a current reasoning trace (thought) and the full answer to a given question by generating a high-quality retrieval query.

You will follow these three steps:

[Step 1: Generate Reasoning Chain]
Given the question and the corresponding thought, extract the information that is directly helpful for answering the question.
Represent each piece of information as a structured knowledge triple in the format <head; relation; tail>. 
Focus on facts, reasoning steps, or clues that can contribute to forming a correct and complete answer to the question.
**Ignore any uncertain, irrelevant or speculative content.**
**All triples must be strictly extracted from the given thought. Do not invent or infer new facts beyond what is explicitly stated in the thought.**

Output Format: <head1; relation1; tail1>, <head2; relation2; tail2>, ... <headN; relationN; tailN>

[Step 2: Identify Knowledge Gaps]
Based on the extracted reasoning chain, identify the **next piece of information** that is required to continue reasoning toward the answer to the question.  
This could be a missing fact, relation, or intermediate concept that, if known, would allow the reasoning to proceed.
If the current reasoning is already sufficient to answer the question, say so clearly.

Output Format:
To continue answering the question, we need to know: [describe the next necessary piece of information]

[Step 3: Formulate Retrieval Query]
Write a concise and effective natural language query that can help retrieve the specific information identified in Step 2.  
The query should be focused, contain the key entities and relations, and be suitable for a search engine or retrieval system.

If no additional information is needed, set the query to None.
"""

QUERY_FORMULATION_INSTRUCTION_SIMPLIFIED = "Generate a reasoning chain as knowledge triples from the given thought, identify the missing information needed to answer the question, and formulate a concise retrieval query. Output triples, missing information, and query."

QUERY_FORMULATION_DEMO = """--- 
Example 1: 
Question: Who distributed a movie whose cast includes an actor who acted in 'Him'? 

Current Thought: 
Alright, so I have this question: "Who distributed a movie whose cast includes an actor who acted in 'Him'?" Hmm, okay, let me try to break this down. 

First, I need to understand what the question is asking. It's asking about a movie distributor. The movie in question has an actor who was in "Him." So, I need to identify the movie "Him." I think "Him" is a movie, but I'm not exactly sure which one. There might be several movies with that title. Maybe I should look up some information about it.

Step 1: Reasoning Chain:
<"Him"; has actor; [unknown]> 

Step 2: Knowledge Gaps:
To continue answering the question, we need to know: Which actor acted in the movie "Him"? Identifying this actor allows us to look for other movies they were in and then determine the distributor of one of those movies.

Step 3: Retrieval Query:
Which actor acted in the movie "Him"?

Example 2: 
Question: What was the occupation of both Marguerite Radclyffe Hall and Charles Bukowski?

Current Thought:
Alright, so I need to figure out the occupations of both Marguerite Radclyffe Hall and Charles Bukowski. Hmm, I'm not too familiar with both of them, but I'll try to break it down.

First, Charles Bukowski. I'm more familiar with him. He was a poet, right? But wait, wasn't he also a novelist? I think he wrote a lot of poetry, and some novels too. His style was pretty distinctive, dealing with themes of alcoholism, urban life, and the underbelly of society. So his main occupation was as a writer, both in poetry and prose.

Step 1: Reasoning Chain:  
<Charles Bukowski; occupation; poet, novelist> 

Step 2: Knowledge Gaps:
To continue answering the question, we need to know: What was the occupation of Marguerite Radclyffe Hall? Once we have that, we can compare it with Bukowski's occupations to complete the answer.

Step 3: Retrieval Query:
What was Marguerite Radclyffe Hall's occupation?
"""

QUERY_FORMULATION_INPUT = """---
Let's begin.
Question: {question}

Current Thought:
{thought}

Step 1: Reasoning Chain: 
[Your output here]

Step 2: Knowledge Gaps:
[Your output here]

Step 3: Retrieval Query:
[Your output here]""" 


QUERY_FORMULATION_PROMPT_WO_DEMO = QUERY_FORMULATION_INSTRUCTION + "\n\n" + QUERY_FORMULATION_INPUT
QUERY_FORMULATION_PROMPT = QUERY_FORMULATION_INSTRUCTION + "\n\n" + QUERY_FORMULATION_DEMO + "\n\n" + QUERY_FORMULATION_INPUT


QUERY_FORMULATION_INSTRUCTION_WITH_THOUGHT = """You are an intelligent reasoning assistant. Your task is to help bridge the information gap between a current reasoning trace (thought) and the full answer to a given question by generating a high-quality retrieval query.

You will follow these three steps:

[Step 1: Generate Reasoning Chain]
Given the question and the corresponding thought, extract the information that is directly helpful for answering the question.
Represent each piece of information as a structured knowledge triple in the format <head; relation; tail>. 
Focus on facts, reasoning steps, or clues that can contribute to forming a correct and complete answer to the question.
**Ignore any uncertain, irrelevant or speculative content.**
**All triples must be strictly extracted from the given thought. Do not invent or infer new facts beyond what is explicitly stated in the thought.**

Output Format: <head1; relation1; tail1>, <head2; relation2; tail2>, ... <headN; relationN; tailN>

[Step 2: Identify Knowledge Gaps]
Based on the extracted reasoning chain, identify the **next piece of information** that is required to continue reasoning toward the answer to the question.  
This could be a missing fact, relation, or intermediate concept that, if known, would allow the reasoning to proceed.
If the current reasoning is already sufficient to answer the question, say so clearly.

Output Format:
To continue answering the question, we need to know: [describe the next necessary piece of information]

[Step 3: Formulate Retrieval Query]
Write a concise and effective natural language query that can help retrieve the specific information identified in Step 2.  
The query should be focused, contain the key entities and relations, and be suitable for a search engine or retrieval system.

If no additional information is needed, set the query to None.

**Output format** 

### Thought 
Explain your own reasoning about what has already been established, and what is still missing. 
Use clear, step-by-step explanation to reflect on what information is present and what is needed.

### Output 
Step 1: Reasoning Chain 
<your output here>

Step 2: Knowledge Gaps 
<your output here>

Step 3: Retrieval Query 
<your output here>

Write nothing else outside this exact format.""" 


QUERY_FORMULATION_DEMO_WITH_THOUGHT = """---
Example 1: 
Question: Who distributed a movie whose cast includes an actor who acted in 'Him'? 

Current Thought: 
Alright, so I have this question: "Who distributed a movie whose cast includes an actor who acted in 'Him'?" Hmm, okay, let me try to break this down. 

First, I need to understand what the question is asking. It's asking about a movie distributor. The movie in question has an actor who was in "Him." So, I need to identify the movie "Him." I think "Him" is a movie, but I'm not exactly sure which one. There might be several movies with that title. Maybe I should look up some information about it.

### Thought 
The thought identifies that the movie in question features an actor from "Him", and that the reasoning depends on figuring out who that actor is.  
However, the actor has not yet been identified, and the current reasoning expresses uncertainty about what "Him" refers to.  
This indicates that the next step requires retrieving the identity of an actor who acted in "Him".

### Output 
Step 1: Reasoning Chain:
<"Him"; has actor; [unknown]> 

Step 2: Knowledge Gaps:
To continue answering the question, we need to know: Which actor acted in the movie "Him"? Identifying this actor allows us to look for other movies they were in and then determine the distributor of one of those movies.

Step 3: Retrieval Query:
Which actor acted in the movie "Him"?

Example 2: 
Question: What was the occupation of both Marguerite Radclyffe Hall and Charles Bukowski?

Current Thought:
Alright, so I need to figure out the occupations of both Marguerite Radclyffe Hall and Charles Bukowski. Hmm, I'm not too familiar with both of them, but I'll try to break it down.

First, Charles Bukowski. I'm more familiar with him. He was a poet, right? But wait, wasn't he also a novelist? I think he wrote a lot of poetry, and some novels too. His style was pretty distinctive, dealing with themes of alcoholism, urban life, and the underbelly of society. So his main occupation was as a writer, both in poetry and prose.

### Thought 
The reasoning explores Charles Bukowski's occupation, proposing that he was a poet and novelist.  
However, it provides no information about Marguerite Radclyffe Hall.  
Since the question asks about both individuals, we still need to find out Hall's occupation to proceed.

### Output 
Step 1: Reasoning Chain:  
<Charles Bukowski; occupation; poet, novelist> 

Step 2: Knowledge Gaps:
To continue answering the question, we need to know: What was the occupation of Marguerite Radclyffe Hall? Once we have that, we can compare it with Bukowski's occupations to complete the answer.

Step 3: Retrieval Query:
What was Marguerite Radclyffe Hall's occupation?
""" 

QUERY_FORMULATION_INPUT_WITH_THOUGHT = """---
Let's begin.
Question: {question}

Current Thought:
{thought}
"""
 
# >>>>>>>>>>>>>>>>>
# Relevant Triples 
# >>>>>>>>>>>>>>>>>

RELEVANT_TRIPLES_INSTRUCTION = """You are an intelligent reasoning assistant. Your task is to select the most useful knowledge triple from a list of candidates to help continue a reasoning process toward answering a given question.

You will be given:
- A question
- A current reasoning chain represented as a sequence of knowledge triples
- A query, which may reflect an information need to guide the next reasoning step (note: the query may be helpful but can also be noisy or imprecise)
- A set of candidate triples

Your goal:
Select only one triple that is most useful for continuing the reasoning toward answering the question.
This selected triple should logically extend the current reasoning chain and help bridge the gap toward the final answer.

Selection criteria:
- The selected triple should logically extend the current reasoning chain. 
- You may use the query as a soft signal to guide your selection, but do not rely on it blindly - prioritize coherence with the reasoning chain and relevance to the question.
- Ignore candidates that are unrelated, redundant, or off-topic.

You should not modify or rewrite the selected triple. Only output the selected triple and do not include any other text. 
"""

RELEVANT_TRIPLES_INSTRUCTION_SIMPLIFIED = "Select the single most relevant knowledge triple from the candidate list that best extends the current reasoning chain toward answering the question. Output only the selected triple."

RELEVANT_TRIPLES_DEMO = """---
Example: 

Candidate Triples:
1. <Charles Bukowski; occupation; poet, novelist, short story writer>
2. <Charles Bukowski; occupation; poet>
3. <Radclyffe Hall; occupation; poet and author> 
4. <Radclyffe Hall; nationality; English> 
5. <Radclyffe Hall; full name; Marguerite Radclyffe Hall>
... 

Question: What was the occupation of both Marguerite Radclyffe Hall and Charles Bukowski?

Reasoning Chain:
<Charles Bukowski; occupation; poet, novelist, short story writer>

Query:
What was the occupation of Marguerite Radclyffe Hall?

Selected Triple:
<Radclyffe Hall; occupation; poet and author>
"""

RELEVANT_TRIPLES_INPUT = """---
Let's begin.

Candidate Triples:
{candidate_triples}

Question: {question}

Reasoning Chain:
{reasoning_chain}

Query:
{query}

Selected Triple:
"""

RELEVANT_TRIPLES_PROMPT = RELEVANT_TRIPLES_INSTRUCTION + "\n\n" + RELEVANT_TRIPLES_DEMO + "\n\n" + RELEVANT_TRIPLES_INPUT
RELEVANT_TRIPLES_PROMPT_WO_DEMO = RELEVANT_TRIPLES_INSTRUCTION + "\n\n" + RELEVANT_TRIPLES_INPUT 


RELEVANT_TRIPLES_INSTRUCTION_WITH_THOUGHT = """You are an intelligent reasoning assistant. Your task is to select the most useful knowledge triple from a list of candidates to help continue a reasoning process toward answering a given question.

You will be given:
- A question
- A current reasoning chain represented as a sequence of knowledge triples
- A query, which may reflect an information need to guide the next reasoning step (note: the query may be helpful but can also be noisy or imprecise)
- A set of candidate triples

Your goal:
Select only one triple that is most useful for continuing the reasoning toward answering the question.
This selected triple should logically extend the current reasoning chain and help bridge the gap toward the final answer.

Selection criteria:
- The selected triple should logically extend the current reasoning chain. 
- You may use the query as a soft signal to guide your selection, but do not rely on it blindly - prioritize coherence with the reasoning chain and relevance to the question.
- Ignore candidates that are unrelated, redundant, or off-topic.

**Output format** 

### Thought
Explain your own reasoning about what has already been established, and what is still missing. 
Use clear, step-by-step explanation to reflect on what information is present and what is needed.

### Selected Triple 
Put your selected triple here. 
DO NOT include any other text. 
Do NOT modify or rewrite the selected triple.
"""

RELEVANT_TRIPLES_DEMO_WITH_THOUGHT = """---
Example: 

Candidate Triples:
1. <Charles Bukowski; occupation; poet, novelist, short story writer>
2. <Charles Bukowski; occupation; poet>
3. <Radclyffe Hall; occupation; poet and author> 
4. <Radclyffe Hall; nationality; English> 
5. <Radclyffe Hall; full name; Marguerite Radclyffe Hall>
... 

Question: What was the occupation of both Marguerite Radclyffe Hall and Charles Bukowski?

Reasoning Chain:
<Charles Bukowski; occupation; poet, novelist, short story writer>

Query:
What was the occupation of Marguerite Radclyffe Hall?

### Thought
The goal is to find Radclyffe Hall's occupation in order to complete the comparison with Charles Bukowski.  
Among the candidate triples, only Triple 3 directly provides occupational information for Radclyffe Hall.  
Other triples mention nationality or full name, which are not relevant to the current reasoning goal.  
Therefore, Triple 3 is the most useful for continuing the reasoning.

### Selected Triple
<Radclyffe Hall; occupation; poet and author>
"""

RELEVANT_TRIPLES_INPUT_WITH_THOUGHT = """---
Let's begin.

Candidate Triples:
{candidate_triples}

Question: {question}

Reasoning Chain:
{reasoning_chain}

Query:
{query}
"""

RELEVANT_TRIPLES_INSTRUCTION_WITH_CONTEXT="""You are an intelligent reasoning assistant. Your task is to select the most useful knowledge triple from a list of candidates to help continue a reasoning process toward answering a given question.

You will be given:
- A question
- A current reasoning chain represented as a sequence of knowledge triples
- A query, which may reflect an information need to guide the next reasoning step (note: the query may be helpful but can also be noisy or imprecise)
- A set of candidate triples
- A context: the original text from which the candidate triples were extracted (to help assess meaning and correctness)

Your goal:
Select only one triple that is most useful for continuing the reasoning toward answering the question.
This selected triple should logically extend the current reasoning chain and help bridge the gap toward the final answer.

### Selection criteria:
- The selected triple should logically extend the current reasoning chain. 
- You may use the query as a soft signal to guide your selection, but do not rely on it blindly - prioritize coherence with the reasoning chain and relevance to the question.
- Use the context to resolve ambiguity or better understand the semantics of a candidate triple.
- Ignore candidates that are unrelated, redundant, or off-topic.

You should not modify or rewrite the selected triple. 

### Output Format (IMPORTANT)
- Only output the **selected triple**.
- DO NOT output explanations, context, or any other text.
- Format the output as: `<subject; predicate; object>` only.
- Do NOT copy or repeat context or metadata such as source title.
- Ignore redundant or unrelated triples.
"""

RELEVANT_TRIPLES_DEMO_WITH_CONTEXT="""---
Example:

Context:
Title: Charles Bukowski
Text: Charles Bukowski was an American poet, novelist, and short story writer.
Title: Radclyffe Hall 
Text: Radclyffe Hall (Marguerite Radclyffe Hall, born in England) was an English poet and author. 

Candidate Triples:
1. Source Title: Charles Bukowski
Text: <Charles Bukowski; occupation; poet, novelist, short story writer>
2. Source Title: Charles Bukowski
Text: <Charles Bukowski; occupation; poet>
3. Sorce Title: Radclyffe Hall 
Text: <Radclyffe Hall; occupation; poet and author> 
4. Sorce Title: Radclyffe Hall 
Text: <Radclyffe Hall; nationality; English> 
5. Sorce Title: Radclyffe Hall 
Text: <Radclyffe Hall; full name; Marguerite Radclyffe Hall>

Question: What was the occupation of both Marguerite Radclyffe Hall and Charles Bukowski?

Reasoning Chain:
<Charles Bukowski; occupation; poet, novelist, short story writer>

Query:
What was the occupation of Marguerite Radclyffe Hall?

Selected Triple:
<Radclyffe Hall; occupation; poet and author>
"""

RELEVANT_TRIPLES_INPUT_WITH_CONTEXT="""---
Let's begin.

Context:
{context}

Candidate Triples:
{candidate_triples}

Question: {question}

Reasoning Chain:
{reasoning_chain}

Query:
{query}

Selected Triple:
"""

# >>>>>>>>>>>>>>>>>
# Verbalization 
# >>>>>>>>>>>>>>>>>

VERBALIZATION_INSTRUCTION = """You are an intelligent reasoning assistant that assists with step-by-step reasoning.

You will be given:
- A question
- The current reasoning thought written in natural language
- A context containing a list of knowledge triples in the format <head; relation; tail>, each of which is associated with a source document

Your task:
Convert the most useful information from the context into fluent natural language text that can be **seamlessly appended** to the current reasoning thought. 
The context may contain irrelevant or noisy information—please focus only on the factual knowledge that contribute meaningfully to continuing the reasoning toward answering the question. 
The output should match the **tone, style, and flow** of the original thought.
You must **not include any thinking, reasoning, speculation, explanation, or conclusions** in your output.

Output Format:

Continuation Text (to append after the current thought):  
[your natural language text, style-matched and logically connected]

Only output the continuation text and do not include any other text. 
"""

VERBALIZATION_DEMO = """---
Example: 

Question: What was the occupation of both Marguerite Radclyffe Hall and Charles Bukowski?

Current Thought:
Alright, so I need to figure out the occupations of both Marguerite Radclyffe Hall and Charles Bukowski. Hmm, I'm not too familiar with both of them, but I'll try to break it down.

First, Charles Bukowski. I'm more familiar with him. He was a poet, right? But wait, wasn't he also a novelist? I think he wrote a lot of poetry, and some novels too. His style was pretty distinctive, dealing with themes of alcoholism, urban life, and the underbelly of society. So his main occupation was as a writer, both in poetry and prose.

Context:
Relevant Triple: <Radclyffe Hall; occupation; poet and author>
Source Document: Marguerite Antonia Radclyffe-Hall (12 August 1880 - 7 October 1943), more known under her pen name Radclyffe Hall, was an English poet and author, best known for the novel The Well of Loneliness, a groundbreaking work in lesbian literature. In adulthood, she often called herself John, rather than Marguerite.

Continuation Text (to append after the current thought):
Radclyffe Hall, on the other hand, was also a poet and author, best known for her novel The Well of Loneliness."""

VERBALIZATION_INPUT = """---
Let's begin.

Question: {question}

Current Thought:
{thought}

Context:
{ctx}

Continuation Text (to append after the current thought):"""


VERBALIZATION_PROMPT = VERBALIZATION_INSTRUCTION + "\n\n" + VERBALIZATION_DEMO + "\n\n" + VERBALIZATION_INPUT 
VERBALIZATION_PROMPT_WO_DEMO = VERBALIZATION_INSTRUCTION + "\n\n" + VERBALIZATION_INPUT 


# >>>>>>>>>>>>>>>>>
# Ablation 1: Directly Determine Retrieval & Formulate Retrieval Query  
# >>>>>>>>>>>>>>>>>

ABLATION1_INSTRUCTION = """
You are an intelligent assistant. Your job is to determine whether external information retrieval is needed to answer a question and, if so, to generate a useful retrieval query.

You will be given:
- A question
- The current reasoning progress ("thought so far")

Your task involves two steps:

1. **Determine if external retrieval is needed**:
   Answer "Yes" if:
   - The current reasoning lacks enough information to answer the question
   - There is uncertainty, guessing, or hesitation (e.g. "maybe", "I think", "not sure")
   - Specific factual information (names, dates, titles, etc.) is required to proceed

   Answer "No" if:
   - The current reasoning clearly and confidently answers the question
   - The question can be answered using general or basic reasoning already available

2. **If retrieval is needed, formulate a focused and effective search query**:
   The query should be concise and help obtain the specific missing information necessary to move reasoning forward.

Return your output as a JSON object in the following format:

{
  "retrieval_required": "Yes" or "No",
  "retrieval_query": "..." or null
}

Only return the JSON and nothing else.

---

Example:

Question: Who directed the movie "The Double Life of Véronique"?

Thought so far: I'm not sure who directed it. I think it's a European film, maybe French or Polish. The title sounds familiar, but I can't recall the director's name.

Expected Output:
{
  "retrieval_required": "Yes",
  "retrieval_query": "Who directed the movie The Double Life of Véronique?"
}
"""

ABLATION1_INPUT = """---

Now begin.

Question: {question}

Thought so far:
{thought}

Output:"""

# >>>>>>>>>>>>>>>>>
# Ablation 2: Use a single agent to make retrieval decision & formulate query 
# >>>>>>>>>>>>>>>>>

JOINT_DETERMINE_RETRIEVAL_QUERY_INSTRUCTION_WITH_CONTEXT = """
You are an intelligent assistant. Your job is to determine whether external information retrieval is needed to answer a question and, if so, to generate a useful retrieval query.

You will be given the following inputs:
- Question: The original user query.  
- Previous Reasoning: A list of prior reasoning thoughts the model has generated.
- Current Reasoning Step: The latest reasoning step to evaluate. 
- Retrieved Context: Text previously retrieved that may contain relevant information.

Your task involves two steps:

1. **Determine if external retrieval is needed**:
    Answer "Yes" if:
    - The current reasoning step requires, asks for, references, or depends on a specific entity / date / location / definition / fact, etc that is not already present (even paraphrased) in **Retrieved Context** or **Previous Reasoning** thought.
    - The current reasoning step mentions at least one new entity / date / number / title absent from prior context.
    - The current reasoning step express uncertainty ("maybe", "I think", "probably", "not sure", "I guess", "unknown", "not provided", "[?]") **AND** requires knowledge that is not found in the **Previous Reasoning** steps or the **Retrieved Context**.
    - The missing fact cannot be safely inferred from general world knowledge or logic (e.g., arithmetic, common definitions).

    Answer "No" if:
    - The information required in the current reasoning step is already present in or can be inferred from **Retrieved Context**.
    - The current reasoning step merely repeats the question, summarises, rephrases or draws a deduction from existing context.
    - The information requested is text-book general knowledge the model is expected to know (physical constants, dictionary definitions, etc.).
    - The step uses exploratory verbs ("let's see", "I think") but no new entity is introduced.
  
2. **If retrieval is needed, formulate a focused and effective search query**:
  The query should be concise and help obtain the specific missing information necessary to move reasoning forward.

Return your output as a JSON object in the following format:

{
  "retrieval_required": "Yes" or "No",
  "retrieval_query": "..." or null
}

Only return the JSON and nothing else.

---

Example:

Retrieved Context: 

Question: 
Who directed the movie "The Double Life of Véronique"?

Previous Reasoning:

Current Reasoning Step: 
I'm not sure who directed it. I think it's a European film, maybe French or Polish. The title sounds familiar, but I can't recall the director's name.

Expected Output:
{
  "retrieval_required": "Yes",
  "retrieval_query": "Who directed the movie The Double Life of Véronique?"
}
"""

# >>>>>>>>>>>>>>>>>
# Ablation 3: The query formulation directly output the query 
# >>>>>>>>>>>>>>>>>

DIRECE_QUERY_FORMULATION_INSTRUCTION = """
You are an intelligent reasoning assistant. You will be given a question and a current reasoning thought, your task is to formulate an appropriate query to retrieve external knowledge for answering the question. 

You should only output the query and DO NOT include any other text!
"""

DIRECT_QUERY_FORMULATION_INPUT = """
Let's begin.
Question: {question}

Current Thought:
{thought}
"""

# >>>>>>>>>>>>>>>>>
# Ablation 5: The knowledge integration agent directly selects a document
# >>>>>>>>>>>>>>>>>

DIRECT_KNOWLEDGE_INTEGRATION_INSTRUCTION = """
You are an intelligent reasoning assistant. You will be given a set of ranked retrieved documents, a question, and a current reasoning thought, your task is to select a relevant document that is helpful in answering the question. 

You should only output the index of the selected document and DO NOT include any other text!
"""

DIRECT_KNOWLEDGE_INTEGRATION_INPUT = """
Retrieved Context: 
{context}

Question:
{question}

Thought:
{thought}

Selected Document Index:"""

