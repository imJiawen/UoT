import os
import json
import argparse
import pickle
from typing import Any, List, Dict

from tqdm import tqdm
from transformers import AutoTokenizer

from src.uot.tasks import get_task
# from src.uot.method import converse, naive_converse
from src.uot.method import naive_converse
from src.uot.eval import evaluate_performance


def _safe_load_logs(log_file: str) -> List[Dict[str, Any]]:
    if not os.path.exists(log_file):
        return []

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            return []
        logs = json.loads(content)
        if isinstance(logs, list):
            return logs
        return []
    except Exception as e:
        print(f"Warning: failed to load existing log file {log_file}: {e}")
        return []


def _safe_save_logs(log_file: str, logs: List[Dict[str, Any]]) -> None:
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(logs, ensure_ascii=False) + '\n')


def _build_log_file(args) -> str:
    if args.naive_run:
        log_file = (
            f'./logs/{args.task}/{args.guesser_model}_as_guesser/'
            f'{args.dataset}_'
            f'examiner_{args.examiner_model}_{"" if args.inform else "un"}inform'
            f'_maxturn{args.max_turn}_{args.task_start_index}-{args.task_end_index}{args.add_info}.json'
        )
    else:
        log_file = (
            f'./logs/{args.task}/{args.guesser_model}_as_guesser/'
            f'{f"OS_init{args.open_set_size}_renew{args.size_to_renew}_" if args.open_set_size > 0 else ""}'
            f'{f"pre{args.n_pre_ask}_" if args.n_pre_ask > 0 else ""}'
            f'{args.dataset}_{args.temperature}_lambda{args.reward_lambda}_acc{not args.none_acc_reward}'
            f'_exp{args.expected_reward_method}_L{args.n_extend_layers}_K{args.n_potential_actions}'
            f'_PRUN{args.n_pruned_nodes}_{"" if args.inform else "un"}inform_EXAMINER{args.examiner_model}'
            f'_maxturn{args.max_turn}_{args.task_start_index}-{args.task_end_index}{args.add_info}.json'
        )
    return log_file


def _build_cot_log_file(args) -> str:
    return _build_log_file(args).replace(".json", "_cot.json")


def _build_root_file(args) -> str:
    return (
        f'./roots/{args.task}/{args.guesser_model}'
        f'{f"OS_init{args.open_set_size}_" if args.open_set_size > 0 else ""}'
        f'_{args.dataset}_{"" if args.inform else "un"}inform_{args.temperature}{args.add_info}_root.pickle'
    )


def _load_or_create_root(task, root_file: str) -> None:
    if os.path.exists(root_file):
        try:
            with open(root_file, 'rb') as r:
                root = pickle.load(r)
            task.create_root(root)
            return
        except Exception as e:
            print(f"Warning: failed to load root from {root_file}: {e}")
            print("Will rebuild root from scratch.")

    os.makedirs(os.path.dirname(root_file), exist_ok=True)
    task.create_root()
    with open(root_file, 'wb') as f:
        pickle.dump(task.root, f)


def _save_root(task, root_file: str) -> None:
    with open(root_file, 'wb') as f:
        pickle.dump(task.root, f)


def _resolve_tokenizer_path(model_name: str) -> str | None:
    model_name = str(model_name)

    mapping = {
        "qwen": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        "qwen_4b": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        "qwen_30b": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "qwen_thinking_4b": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        "qwen_thinking_30b": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "qwen_instruct_4b": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
        "qwen_instruct_30b": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "qwen3-4b-local": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        "qwen3-30b-local": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "qwen3-4b-instruct-local": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
        "qwen3-30b-instruct-local": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "llama3.1-8b-local": "/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct",
        "gpt_oss_20b": "/hpc2hdd/home/mpeng885/models/gpt-oss-20b",

        "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
        "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct": "/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct",
        "/hpc2hdd/home/mpeng885/models/gpt-oss-20b": "/hpc2hdd/home/mpeng885/models/gpt-oss-20b",
    }

    # OpenAI hosted models: no local tokenizer path here
    if model_name in {"gpt-5", "gpt-5-mini", "gpt-4", "gpt-3.5-turbo"}:
        return None

    if model_name in mapping:
        return mapping[model_name]

    return None


def _load_tokenizer(args):
    candidate_models = [args.guesser_model, args.examiner_model]

    for model_name in candidate_models:
        tokenizer_path = _resolve_tokenizer_path(model_name)
        if tokenizer_path is None:
            continue

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            print(f"Loaded tokenizer from {tokenizer_path} for token statistics.")
            return tokenizer
        except Exception as e:
            print(f"Warning: failed to load tokenizer from {tokenizer_path}: {e}")

    print(
        "Warning: no local tokenizer could be loaded for the selected models. "
        "Falling back to approximate whitespace token counting."
    )
    return None


def _count_tokens(text: Any, tokenizer) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)

    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass

    return len(text.split())


def _extract_cot_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cot_logs = []
    for idx, l in enumerate(logs):
        cot_logs.append({
            "sample_id": idx,
            "index": l.get("index", idx),
            "item": l.get("item"),
            "state": l.get("state"),
            "turn": l.get("turn"),
            "thinking_g": l.get("thinking_g", []),
            "thinking_e": l.get("thinking_e", []),
            "thinking_tokens_g": l.get("thinking_tokens_g", 0),
            "thinking_tokens_e": l.get("thinking_tokens_e", 0),
        })
    return cot_logs


def _compute_metrics(logs: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    total_guess_prompt = 0
    total_guess_output_visible = 0
    total_guess_thinking = 0

    total_exam_prompt = 0
    total_exam_output_visible = 0
    total_exam_thinking = 0

    num_guess_prompt_msgs = 0
    num_guess_output_visible_msgs = 0
    num_guess_thinking_msgs = 0

    num_exam_prompt_msgs = 0
    num_exam_output_visible_msgs = 0
    num_exam_thinking_msgs = 0

    num_samples = len(logs)

    if num_samples == 0:
        return {
            "num_samples": 0,
            "num_success": 0,
            "accuracy": 0.0,
            "avg_turn": 0.0,

            "total_guess_prompt_tokens": 0,
            "total_guess_output_visible_tokens": 0,
            "total_guess_thinking_tokens": 0,
            "total_guess_output_tokens": 0,
            "total_guess_tokens": 0,

            "total_exam_prompt_tokens": 0,
            "total_exam_output_visible_tokens": 0,
            "total_exam_thinking_tokens": 0,
            "total_exam_output_tokens": 0,
            "total_exam_tokens": 0,

            "avg_guess_prompt_tokens": 0,
            "avg_guess_output_visible_tokens": 0,
            "avg_guess_thinking_tokens": 0,

            "avg_exam_prompt_tokens": 0,
            "avg_exam_output_visible_tokens": 0,
            "avg_exam_thinking_tokens": 0,
        }

    num_success = sum(1 for l in logs if l.get("state") == 1)
    accuracy = num_success / num_samples
    avg_turn = sum(l.get("turn", 0) for l in logs) / num_samples

    for l in logs:
        history_g = l.get("history_g", [])
        history_e = l.get("history_e", [])
        thinking_g = l.get("thinking_g", [])
        thinking_e = l.get("thinking_e", [])

        for msg in history_g:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                total_guess_prompt += _count_tokens(content, tokenizer)
                num_guess_prompt_msgs += 1
            elif role == "assistant":
                total_guess_output_visible += _count_tokens(content, tokenizer)
                num_guess_output_visible_msgs += 1
            elif role == "system":
                total_guess_output_visible += _count_tokens(content, tokenizer)
                num_guess_output_visible_msgs += 1

        for msg in history_e:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                total_exam_prompt += _count_tokens(content, tokenizer)
                num_exam_prompt_msgs += 1
            elif role == "assistant":
                total_exam_output_visible += _count_tokens(content, tokenizer)
                num_exam_output_visible_msgs += 1
            elif role == "system":
                total_exam_output_visible += _count_tokens(content, tokenizer)
                num_exam_output_visible_msgs += 1

        sample_guess_thinking_tokens = l.get("thinking_tokens_g", None)
        sample_exam_thinking_tokens = l.get("thinking_tokens_e", None)

        if sample_guess_thinking_tokens is not None:
            total_guess_thinking += int(sample_guess_thinking_tokens or 0)
            num_guess_thinking_msgs += len(thinking_g)
        else:
            for msg in thinking_g:
                content = msg.get("content", "")
                total_guess_thinking += _count_tokens(content, tokenizer)
                num_guess_thinking_msgs += 1

        if sample_exam_thinking_tokens is not None:
            total_exam_thinking += int(sample_exam_thinking_tokens or 0)
            num_exam_thinking_msgs += len(thinking_e)
        else:
            for msg in thinking_e:
                content = msg.get("content", "")
                total_exam_thinking += _count_tokens(content, tokenizer)
                num_exam_thinking_msgs += 1

    total_guess_output_tokens = total_guess_output_visible + total_guess_thinking
    total_exam_output_tokens = total_exam_output_visible + total_exam_thinking

    metrics = {
        "num_samples": num_samples,
        "num_success": num_success,
        "accuracy": accuracy,
        "avg_turn": avg_turn,

        "total_guess_prompt_tokens": total_guess_prompt,
        "total_guess_output_visible_tokens": total_guess_output_visible,
        "total_guess_thinking_tokens": total_guess_thinking,
        "total_guess_output_tokens": total_guess_output_tokens,
        "total_guess_tokens": total_guess_prompt + total_guess_output_tokens,

        "total_exam_prompt_tokens": total_exam_prompt,
        "total_exam_output_visible_tokens": total_exam_output_visible,
        "total_exam_thinking_tokens": total_exam_thinking,
        "total_exam_output_tokens": total_exam_output_tokens,
        "total_exam_tokens": total_exam_prompt + total_exam_output_tokens,

        "avg_guess_prompt_tokens": total_guess_prompt / num_guess_prompt_msgs if num_guess_prompt_msgs else 0,
        "avg_guess_output_visible_tokens": total_guess_output_visible / num_guess_output_visible_msgs if num_guess_output_visible_msgs else 0,
        "avg_guess_thinking_tokens": total_guess_thinking / num_guess_thinking_msgs if num_guess_thinking_msgs else 0,

        "avg_exam_prompt_tokens": total_exam_prompt / num_exam_prompt_msgs if num_exam_prompt_msgs else 0,
        "avg_exam_output_visible_tokens": total_exam_output_visible / num_exam_output_visible_msgs if num_exam_output_visible_msgs else 0,
        "avg_exam_thinking_tokens": total_exam_thinking / num_exam_thinking_msgs if num_exam_thinking_msgs else 0,
    }
    return metrics


def run(args):
    task = get_task(args)

    original_start_index = max(args.task_start_index, 0)
    args.task_start_index = original_start_index

    if args.task_end_index < 0:
        args.task_end_index = len(task.data)
    else:
        args.task_end_index = min(args.task_end_index, len(task.data))

    log_file = _build_log_file(args)
    cot_log_file = _build_cot_log_file(args)
    root_file = None

    if not args.naive_run:
        root_file = _build_root_file(args)
        _load_or_create_root(task, root_file)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logs = _safe_load_logs(log_file)

    if len(logs) > 0:
        resumed_start = original_start_index + len(logs)
        if resumed_start > args.task_end_index:
            resumed_start = args.task_end_index
        print(
            f"Found existing log with {len(logs)} samples. "
            f"Resuming from index {resumed_start}."
        )
        args.task_start_index = resumed_start

    for i in tqdm(range(args.task_start_index, args.task_end_index)):
        # if args.naive_run:
        log = naive_converse(task, i)
        # else:
        #     log = converse(task, i)
        #     _save_root(task, root_file)

        logs.append(log)
        _safe_save_logs(log_file, logs)

        cot_logs = _extract_cot_logs(logs)
        _safe_save_logs(cot_log_file, cot_logs)

    evaluate_performance(log_file, task)

    tokenizer = _load_tokenizer(args)
    metrics = _compute_metrics(logs, tokenizer)

    metrics_file = log_file.replace(".json", "_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics to {metrics_file}")
    print(f"Saved CoT log to {cot_log_file}")


def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('--guesser_model', type=str, default='qwen')
    args.add_argument('--temperature', type=float, default=0)
    args.add_argument('--examiner_model', type=str, default='qwen')

    args.add_argument(
        '--task',
        type=str,
        default='20q',
        choices=['20q', 'md', 'tb', 'mediq']
    )
    args.add_argument(
        '--dataset',
        type=str,
        default='common',
        choices=['bigbench', 'common', 'thing', 'DX', 'MedDG', 'FloDial', 'icraftmd', 'imedqa']
    )
    args.add_argument('--task_start_index', type=int, default=-1)
    args.add_argument('--task_end_index', type=int, default=-1)
    args.add_argument('--open_set_size', type=int, default=-1)
    args.add_argument('--size_to_renew', type=int, default=-1)
    args.add_argument('--n_pre_ask', type=int, default=0)

    args.add_argument('--naive_run', action='store_true', default=False)
    args.add_argument('--inform', action='store_true', default=False)

    args.add_argument('--reward_lambda', type=float, default=0.4)
    args.add_argument('--n_extend_layers', type=int, default=3)
    args.add_argument('--n_potential_actions', type=int, default=3)
    args.add_argument('--max_turn', type=int, default=20)
    args.add_argument('--n_pruned_nodes', type=float, default=0)
    args.add_argument('--add_info', type=str, default='')

    args.add_argument('--expected_action_tokens', type=int, default=500)
    args.add_argument('--expected_target_tokens', type=int, default=20)

    args.add_argument('--none_acc_reward', action='store_true', default=False)
    args.add_argument('--expected_reward_method', type=str, default='avg', choices=['avg', 'max'])

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)