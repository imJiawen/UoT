import copy
import re
from typing import Any, Tuple

from src.uot.chat_utils import renew_open_set
from src.uot.models import get_response_method, unpack_model_response
from src.uot.uot import select, renew_node_to_root


def _split_thinking_content(text: str) -> Tuple[str, str]:
    """
    Backward-compatible parser for legacy wrappers that only return text.
    """
    if not isinstance(text, str):
        text = str(text)

    original_text = text

    if "</think>" in text:
        left, right = text.rsplit("</think>", 1)
        thinking_text = re.sub(r"</?think>", "", left, flags=re.IGNORECASE).strip()
        visible_text = right.strip()
        return visible_text, thinking_text

    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if think_blocks:
        thinking_text = "\n".join([b.strip() for b in think_blocks if b.strip()]).strip()
        visible_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return visible_text, thinking_text

    return original_text.strip(), ""


def _normalize_response_output(resp: Any) -> Tuple[Any, str, str, int]:
    """
    Normalize different model wrapper return formats into:
        (raw_response, visible_text, thinking_text, think_tokens)
    """
    raw, answer, cot, think_tokens = unpack_model_response(resp)

    if cot:
        return raw, str(answer), str(cot), int(think_tokens or 0)

    # backward-compatible fallback for legacy wrappers
    if answer:
        visible_text, thinking_text = _split_thinking_content(str(answer))
        if thinking_text and (think_tokens is None or int(think_tokens or 0) == 0):
            think_tokens = len(thinking_text.split())
        return raw, visible_text, thinking_text, int(think_tokens or 0)

    return raw, "", "", int(think_tokens or 0)


def _call_model(response_fn, messages, model: str, **kwargs) -> Tuple[Any, str, str, int]:
    """
    Unified model call wrapper.
    Always returns (raw_response, visible_text, thinking_text, think_tokens).
    """
    resp = response_fn(messages, model=model, **kwargs)
    return _normalize_response_output(resp)


def parse_yes_no(text: str):
    if not isinstance(text, str):
        return None

    t = text.strip().lower()
    t = re.sub(r"^[\"'\s]+|[\"'\s]+$", "", t)

    yes_patterns = [
        r"^yes\b",
        r"^yeah\b",
        r"^yep\b",
        r"^correct\b",
        r"^true\b",
        r"^indeed\b",
        r"^that is correct\b",
        r"^the answer is yes\b",
    ]
    no_patterns = [
        r"^no\b",
        r"^nope\b",
        r"^incorrect\b",
        r"^false\b",
        r"^that is not correct\b",
        r"^the answer is no\b",
    ]

    for p in yes_patterns:
        if re.match(p, t):
            return True
    for p in no_patterns:
        if re.match(p, t):
            return False
    return None


def is_winning_response(text: str) -> bool:
    if not isinstance(text, str):
        return False

    t = text.strip().lower()
    winning_phrases = [
        "you guessed it",
        "you are right",
        "you're right",
        "that is correct",
        "correct,",
        "correct.",
        "yes, that's right",
        "yes, you are right",
        "yes, that's correct",
    ]
    return any(p in t for p in winning_phrases)


def get_examiner_response(task, history):
    response_fn = get_response_method(task.examiner_model)

    if len(history) > 12:
        msg = [history[0]] + history[-11:]
    else:
        msg = history

    raw_response, visible_text, thinking_text, think_tokens = _call_model(
        response_fn,
        msg,
        model=task.examiner_model,
    )
    return raw_response, visible_text, thinking_text, think_tokens


def get_guesser_response(task, history, ques_id, node):
    response_fn = get_response_method(task.guesser_model)

    def simplify_rsp(rsp_text: str):
        examiner_fn = get_response_method(task.examiner_model)
        simplified_text = rsp_text
        simplify_thinking = ""
        simplify_think_tokens = 0

        if len(rsp_text.split()) > task.expected_action_tokens:
            m = [{
                "role": "user",
                "content": task.prompts.extract_q_prompt.format(rsp=rsp_text)
            }]
            _, simplified_text, simplify_thinking, simplify_think_tokens = _call_model(
                examiner_fn,
                m,
                model=task.examiner_model,
                max_tokens=task.expected_action_tokens
            )

        return simplified_text, simplified_text, simplify_thinking, simplify_think_tokens

    if len(node.items) == 1:
        target_question = (
            task.prompts.target_question_FA if task.free_answer
            else task.prompts.target_question
        )
        target_q = target_question.format(target=node.items[0])

        if target_q not in [h["content"] for h in history]:
            return node, target_q, target_q, "", 0, False
        else:
            targeting_prompt_free = (
                task.prompts.targeting_prompt_free_FA if task.free_answer
                else task.prompts.targeting_prompt_free
            )
            msg = copy.deepcopy(history) + [{
                "role": "user",
                "content": targeting_prompt_free
            }]
            _, resp_text, thinking_text, think_tokens = _call_model(
                response_fn,
                msg,
                model=task.guesser_model
            )
            simplified_resp, simplified_resp_text, simplify_thinking, simplify_think_tokens = simplify_rsp(resp_text)
            merged_thinking = "\n".join(x for x in [thinking_text, simplify_thinking] if x).strip()
            merged_think_tokens = int(think_tokens or 0) + int(simplify_think_tokens or 0)
            return node, simplified_resp, simplified_resp_text, merged_thinking, merged_think_tokens, False

    if ques_id < int(task.max_turn * 0.6):
        n = select(task, node)
        if n:
            return n, n.question, n.question, "", 0, True

    if task.inform:
        targeting_prompt_set = (
            task.prompts.targeting_prompt_set_FA if task.free_answer
            else task.prompts.targeting_prompt_set
        )
        msg = copy.deepcopy(history) + [{
            "role": "user",
            "content": targeting_prompt_set.format(item_list_str=', '.join(node.items))
        }]
    else:
        targeting_prompt_set_wo_opt = (
            task.prompts.targeting_prompt_set_FA_wo_opt if task.free_answer
            else task.prompts.targeting_prompt_set_wo_opt
        )
        msg = copy.deepcopy(history) + [{
            "role": "user",
            "content": targeting_prompt_set_wo_opt
        }]

    _, guesser_resp_text, thinking_text, think_tokens = _call_model(
        response_fn,
        msg,
        model=task.guesser_model
    )
    simplified_resp, simplified_resp_text, simplify_thinking, simplify_think_tokens = simplify_rsp(guesser_resp_text)
    merged_thinking = "\n".join(x for x in [thinking_text, simplify_thinking] if x).strip()
    merged_think_tokens = int(think_tokens or 0) + int(simplify_think_tokens or 0)
    return node, simplified_resp, simplified_resp_text, merged_thinking, merged_think_tokens, False


def get_guesser_naive_response(task, history, ques_id):
    response_fn = get_response_method(task.guesser_model)

    msg = copy.deepcopy(history)
    prompt = ""
    if ques_id > int(task.max_turn * 0.7):
        prompt += task.prompts.urge_prompt
        if task.inform:
            prompt += task.prompts.inform_prompt.format(item_list_str=', '.join(task.set))
    prompt += "\nYou must reply me with 1 question to ask only."

    if len(msg) == 0:
        msg = [{"role": "system", "content": ""}]
    else:
        msg[-1]["content"] += " " + prompt

    _, rsp_text, thinking_text, think_tokens = _call_model(
        response_fn,
        msg,
        model=task.guesser_model
    )

    def extract_ques(text: str) -> Tuple[str, str, int]:
        examiner_fn = get_response_method(task.examiner_model)
        message = [{
            "role": "user",
            "content": task.prompts.extract_q_prompt.format(rsp=text)
        }]
        _, extracted_text, extract_thinking, extract_think_tokens = _call_model(
            examiner_fn,
            message,
            model=task.examiner_model
        )
        return extracted_text, extract_thinking, extract_think_tokens

    if len(rsp_text.split()) > task.expected_action_tokens:
        extracted_text, extract_thinking, extract_think_tokens = extract_ques(rsp_text)
        merged_thinking = "\n".join(x for x in [thinking_text, extract_thinking] if x).strip()
        merged_think_tokens = int(think_tokens or 0) + int(extract_think_tokens or 0)
        return extracted_text, merged_thinking, merged_think_tokens

    return rsp_text, thinking_text, int(think_tokens or 0)


def converse(task, i):
    item = task.data[i]["target"]
    target_decl = task.prompts.target_declaration.format(target=item)
    print(target_decl)
    print("------ DIALOGUE START ------")

    count = 0
    state = 0

    thinking_g = []
    thinking_e = []
    thinking_tokens_g = 0
    thinking_tokens_e = 0

    if not task.free_answer:
        history_e = [{
            'role': 'system',
            'content': task.prompts.examiner_prologue.format(item=item)
        }]
    else:
        history_e = [{
            'role': 'system',
            'content': task.prompts.simulator_prologue.format(
                item=item,
                conv_hist=task.data[i]["conv_hist"]
            )
        }]

    if "self_repo" in task.data[i]:
        guesser_prologue = (
            task.prompts.guesser_prologue_FA if task.free_answer
            else task.prompts.guesser_prologue
        )
        history_g = [{
            'role': 'system',
            'content': guesser_prologue.format(repo=task.data[i]["self_repo"])
        }]
        print("Self-report:", task.data[i]["self_repo"])
        node = task.root.handle_self_repo(task, task.data[i]["self_repo"])
    else:
        history_g = [{
            'role': 'system',
            'content': task.prompts.guesser_prologue
        }]

        if task.open_set_size > 0 and task.n_pre_ask > 0:
            for _ in range(task.n_pre_ask):
                bot1_response_text, bot1_thinking_text, bot1_think_tokens = get_guesser_naive_response(task, history_g, count + 1)
                print("Bot 2:", bot1_response_text)

                if bot1_thinking_text:
                    thinking_g.append({
                        'turn': count + 1,
                        'content': bot1_thinking_text,
                        'tokens': int(bot1_think_tokens or 0),
                    })
                    thinking_tokens_g += int(bot1_think_tokens or 0)

                history_g.append({'role': 'assistant', 'content': bot1_response_text})
                history_e.append({'role': 'user', 'content': bot1_response_text})

                _, bot2_response_text, bot2_thinking_text, bot2_think_tokens = get_examiner_response(task, history_e)
                print("Bot 1:", bot2_response_text)

                if bot2_thinking_text:
                    thinking_e.append({
                        'turn': count + 1,
                        'content': bot2_thinking_text,
                        'tokens': int(bot2_think_tokens or 0),
                    })
                    thinking_tokens_e += int(bot2_think_tokens or 0)

                history_e.append({'role': 'assistant', 'content': bot2_response_text})
                history_g.append({'role': 'user', 'content': bot2_response_text})

                count += 1
                print('------', count, '-------------')

            node = task.root.handle_self_repo(task, history_g)
        else:
            node = task.root.handle_self_repo(task, history_g) if task.open_set_size > 0 else task.root

    node, bot1_response, bot1_response_text, bot1_thinking_text, bot1_think_tokens, flag = get_guesser_response(
        task, history_g, count + 1, node
    )
    print("Bot 2:", bot1_response_text)

    if bot1_thinking_text:
        thinking_g.append({
            'turn': count + 1,
            'content': bot1_thinking_text,
            'tokens': int(bot1_think_tokens or 0),
        })
        thinking_tokens_g += int(bot1_think_tokens or 0)

    history_g.append({'role': 'assistant', 'content': bot1_response_text})
    history_e.append({'role': 'user', 'content': bot1_response_text})

    while True:
        _, bot2_response_text, bot2_thinking_text, bot2_think_tokens = get_examiner_response(task, history_e)

        if task.free_answer and flag:
            node = node.handle_free_answer(task, bot1_response_text, bot2_response_text)
        else:
            yn = parse_yes_no(bot2_response_text)
            if yn is True:
                node = node.ans2node(True)
            elif yn is False:
                node = node.ans2node(False)

        if bot2_thinking_text:
            thinking_e.append({
                'turn': count + 1,
                'content': bot2_thinking_text,
                'tokens': int(bot2_think_tokens or 0),
            })
            thinking_tokens_e += int(bot2_think_tokens or 0)

        history_e.append({'role': 'assistant', 'content': bot2_response_text})
        history_g.append({'role': 'user', 'content': bot2_response_text})
        print("Bot 1:", bot2_response_text)

        if is_winning_response(bot2_response_text):
            state = 1
            break

        count += 1
        print('------', count, '-------------')

        if count >= task.max_turn:
            print("Bot 1: Sorry, time's up. You lose this game.", target_decl)
            state = -1
            break

        if (
            count <= int(task.max_turn * 0.3) + task.n_pre_ask
            and task.open_set_size > 0
            and len(node.items) < task.size_to_renew
        ):
            node = renew_node_to_root(task, node, history_g)

        node, bot1_response, bot1_response_text, bot1_thinking_text, bot1_think_tokens, flag = get_guesser_response(
            task, history_g, count + 1, node
        )
        print("Bot 2:", bot1_response_text)

        if bot1_thinking_text:
            thinking_g.append({
                'turn': count + 1,
                'content': bot1_thinking_text,
                'tokens': int(bot1_think_tokens or 0),
            })
            thinking_tokens_g += int(bot1_think_tokens or 0)

        history_g.append({'role': 'assistant', 'content': bot1_response_text})
        history_e.append({'role': 'user', 'content': bot1_response_text})

    return {
        'index': i,
        'turn': count,
        'history_g': history_g,
        'history_e': history_e,
        'thinking_g': thinking_g,
        'thinking_e': thinking_e,
        'thinking_tokens_g': thinking_tokens_g,
        'thinking_tokens_e': thinking_tokens_e,
        'state': state,
        'item': task.data[i]["target"]
    }


def naive_converse(task, i):
    item = task.data[i]["target"]
    target_decl = task.prompts.target_declaration.format(target=item)
    print(target_decl)

    thinking_g = []
    thinking_e = []
    thinking_tokens_g = 0
    thinking_tokens_e = 0

    if "self_repo" in task.data[i]:
        guesser_prologue = (
            task.prompts.guesser_prologue_FA if task.free_answer
            else task.prompts.guesser_prologue
        )
        history_g = [{
            'role': 'system',
            'content': guesser_prologue.format(repo=task.data[i]["self_repo"])
        }]
        print("Self-report:", task.data[i]["self_repo"])
    else:
        history_g = [{
            'role': 'system',
            'content': task.prompts.guesser_prologue
        }]

    if not task.free_answer:
        history_e = [{
            'role': 'system',
            'content': task.prompts.examiner_prologue.format(item=item)
        }]
    else:
        history_e = [{
            'role': 'system',
            'content': task.prompts.simulator_prologue.format(
                item=item,
                conv_hist=task.data[i]["conv_hist"]
            )
        }]

    print("------ DIALOGUE START ------")
    count = 0

    bot1_response_text, bot1_thinking_text, bot1_think_tokens = get_guesser_naive_response(task, history_g, count + 1)
    print("Bot 2:", bot1_response_text)

    if bot1_thinking_text:
        thinking_g.append({
            'turn': count + 1,
            'content': bot1_thinking_text,
            'tokens': int(bot1_think_tokens or 0),
        })
        thinking_tokens_g += int(bot1_think_tokens or 0)

    history_g.append({'role': 'assistant', 'content': bot1_response_text})
    history_e.append({'role': 'user', 'content': bot1_response_text})

    while True:
        _, bot2_response_text, bot2_thinking_text, bot2_think_tokens = get_examiner_response(task, history_e)

        if bot2_thinking_text:
            thinking_e.append({
                'turn': count + 1,
                'content': bot2_thinking_text,
                'tokens': int(bot2_think_tokens or 0),
            })
            thinking_tokens_e += int(bot2_think_tokens or 0)

        history_e.append({'role': 'assistant', 'content': bot2_response_text})
        history_g.append({'role': 'user', 'content': bot2_response_text})
        print("Bot 1:", bot2_response_text)

        if is_winning_response(bot2_response_text):
            state = 1
            break

        count += 1
        print('------', count, '-------------')

        if count >= task.max_turn:
            print("Bot 1: Sorry, time's up. You lose this game.", target_decl)
            state = -1
            break

        bot1_response_text, bot1_thinking_text, bot1_think_tokens = get_guesser_naive_response(task, history_g, count + 1)
        print("Bot 2:", bot1_response_text)

        if bot1_thinking_text:
            thinking_g.append({
                'turn': count + 1,
                'content': bot1_thinking_text,
                'tokens': int(bot1_think_tokens or 0),
            })
            thinking_tokens_g += int(bot1_think_tokens or 0)

        history_g.append({'role': 'assistant', 'content': bot1_response_text})
        history_e.append({'role': 'user', 'content': bot1_response_text})

    return {
        'index': i,
        'turn': count,
        'history_g': history_g,
        'history_e': history_e,
        'thinking_g': thinking_g,
        'thinking_e': thinking_e,
        'thinking_tokens_g': thinking_tokens_g,
        'thinking_tokens_e': thinking_tokens_e,
        'state': state,
        'item': task.data[i]["target"],
    }