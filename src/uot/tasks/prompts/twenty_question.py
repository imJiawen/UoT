from src.uot.tasks.prompts.general import *

# method
generate_prompt_rest_w_opt = '''Here are all the X:
{items_str}

{n} questions are designed to classify the possible X above based on the answer for these question:
{asked}
For each X under each question, if the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO. And your answer should be like:
Question 1: {Q1}
YES: ...
Count of YES: ...
NO: ...
Count of NO: ...
'''

generate_prompt_rest = '''{n} questions are designed to classify the possible X based on the answer for these question:
{asked}
For each X under each question, if the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO. And your answer should be like:
Question 1: {Q1}
YES: ...
Count of YES: ...
NO: ...
Count of NO: ...
'''

generate_prompt_w_opt = '''Here are all the X:
{items_str}

Please design a question about X and can only be answer by YES or NO. {asked} Then classify the possible X above based on this question. If the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO.
Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
You should think about best {n} questions to response. And your answer should be:
Question 1: Is X ...?
YES: aaa, bbb, ...
Count of YES: ...
NO: ccc, ddd, ...
Count of NO: ...
'''

generate_prompt = '''Please design a question about X and can only be answer by YES or NO. {asked} Then classify the possible X above based on this question. If the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO.
Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
You should think about best {n} questions to response. And your answer should be:
Question 1: Is X ...?
YES: aaa, bbb, ...
Count of YES: ...
NO: ccc, ddd, ...
Count of NO: ...
'''

# conversation
target_question = "Is X a '{target}'?"

targeting_prompt_free = """Note that you should guess and ask what X exactly is from now on.
The question must start with 'Is X ...'"""


targeting_prompt_set = """Note that you should guess and ask what X exactly is from now on. X is possible a:
{item_list_str}, or other.
The question must start with 'Is X ...'"""

targeting_prompt_set_wo_opt = """Note that you should guess and ask what X exactly is from now on. 
The question must start with 'Is X ...'"""

# guesser_prologue = '''Let us play the game of 20 questions. I am impersonating the thing, X. You will ask me up to {n} questions which start with 'Is X' and can only be answered by yes or no, and I will answer each one truthfully based on being X.
# Let us begin. Ask me the first question.
# '''

guesser_prologue = """
Let us play the game of 20 questions. I am impersonating the thing, X.

Your goal is to identify X within at most {n} turns.

Rules:
1. Ask exactly one question only.
2. Each question should start with "Is X" whenever possible.
3. Do not ask more than one question in a turn.
4. Do not add explanation, justification, or extra text.
5. If this is the final turn, do not ask a question. Instead, make exactly one final guess.

Final-turn output rule:
- On the final turn, output only:
  X is "<your best guess>".

Let us begin. Ask me the first question.
"""

oracle_guesser_prologue = """
Let us play the game of 20 questions. I am impersonating the thing, X.

Your goal is to identify X within at most {n} turns.

Rules:
1. Ask exactly one question only.
2. Each question should start with "Is X" whenever possible.
3. Do not ask more than one question in a turn.
4. Do not add explanation, justification, or extra text.
5. If this is the final turn, do not ask a question. Instead, make exactly one final guess.

Final-turn output rule:
- On the final turn, output only:
  X is "<your best guess>".

Let us begin. Ask me the first question.
"""

# oracle_guesser_prologue = """
# Let us play a pool-based identification game about X.

# The answerer does not start with one fixed hidden item. Instead, it maintains an internal belief over a candidate pool and answers according to an oracle policy.

# Your goal is to make the oracle accept one candidate within at most {n} turns.

# Rules:
# 1. On each non-final turn, output exactly one action only.
# 2. The action may be either one yes/no question or one direct guess.
# 3. Each question should start with "Is X" whenever possible.
# 4. Do not output both a question and a guess in the same turn.
# 5. Do not add explanation, justification, or extra text.
# 6. On the final turn, do not ask a question. Instead, make exactly one final guess.

# Direct-guess format:
# - X is "<your best guess>".

# Let us begin. Ask the first question or make the first direct guess.
# """

urge_prompt = ""

inform_prompt = "The X may be one of items below:\n{item_list_str}"


# # examiner
# examiner_prologue = '''Let us play the game of 20 questions. You are the answerer and I am guesser. X is '{item}'. 
# I will ask you up to 20 questions and you should answer each one truthfully based on being X. 
# If I guess correctly what X is, answer me "You guessed it. X is '{item}'."
# Note that never directly tell me what X is all the time.
# Let us begin. Here is my first question.
# '''

examiner_prologue = '''Let us play the game of 20 questions. You are the answerer and I am the guesser. X is "{item}".

You must answer each question truthfully based on being X, but you must follow this output rule strictly:

Allowed responses:
- "Yes"
- "No"
- "Partially"
- "Not applicable"
- "You guessed it. X is "{item}"."

Important rules:
1. Output exactly one of the allowed responses above.
2. Do not add any explanation, justification, hint, clarification, punctuation, or extra words.
3. Do not reveal any additional information about X beyond the allowed response.
4. If the guess is exactly correct, output: "You guessed it. X is "{item}"."
5. For all other questions, output only one of: "Yes", "No", "Partially", or "Not applicable".

Examples of forbidden outputs:
- "Yes, because it is a mammal."
- "No. It usually lives in Africa."
- "Partially, since that depends on context."

Examples of valid outputs:
- "Yes"
- "No"
- "Partially"
- "Not applicable"

Let us begin. Here is my first question.
'''

oracle_examiner_prologue = '''You are an entropy-driven oracle for a pool-based identification game.

Oracle mode: {mode}
Candidate pool: {pool_name}

There is no fixed hidden answer at the start of the episode. Instead, you maintain an internal posterior over the candidate pool.

Output rules:
- For ordinary yes/no questions, output exactly one of: "Yes.", "No.", or "Pass."
- For direct guesses, output exactly one of: "Correct, you guessed it." or "No, that is not correct."

Behavior rule:
- In adversarial mode, choose the feasible response that preserves more posterior entropy.
- In comparative mode, choose the feasible response that reduces posterior entropy more quickly.
'''

# open set
init_open_set_prompt = '''You are playing the game of 20 questions and you are the guesser. Based on the conversation history, please propose {size} things that you think the answerer might have in mind.
Your response should be: ["thing1", "thing2", ...]'''

renew_open_set_prompt = '''Based on the conversation history, please propose {size} things that the answerer of 20 question game might have in mind.
The list of {size} things should contains {item_list}
Your response should be: ["thing1", "thing2", ...]'''

final_guess_prompt = """
This is the final turn.
You must stop asking questions and make exactly one final guess now.

Output format:
X is "<your best guess>".

Do not ask a question.
Do not add any explanation.
Do not output anything else.
"""

final_guess_prompt_inform = """
This is the final turn.
You must stop asking questions and make exactly one final guess now.

Possible candidates:
{item_list_str}

Output format:
X is "<your best guess>".

Do not ask a question.
Do not add any explanation.
Do not output anything else.
"""


extract_guess_prompt = """
Rewrite the following response into exactly one final guess in this format:

X is "<guess>".

Rules:
1. Output exactly one line.
2. Do not output a question.
3. Do not add explanation or extra words.
4. Keep only the guessed entity.

Response:
{rsp}
"""