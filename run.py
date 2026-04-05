import os
import sys
import re
import hashlib
from pathlib import Path

# run.py lives at external/UoT/run.py; repo root is two levels up (contains `reasoning/`).
# Python only adds external/UoT to sys.path by default, so `reasoning.*` would not resolve.
_reasoning_root = Path(__file__).resolve().parent.parent.parent
_root = str(_reasoning_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import argparse
import json
import pickle
from typing import Any, List, Dict

from tqdm import tqdm
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

from reasoning.evaluate.twenty_question.offline_evaluator import (
    HostedGPTJudge,
    LocalVLLMJudge,
    POOL_MAP,
    normalize_entity,
)
from src.uot.tasks import get_task
from src.uot.method import naive_converse
from src.uot.eval import evaluate_performance

from src.uot.oracle_examiner import ComparativeEntropyOracle, OracleConfig
from src.uot.models import get_local_model_config, is_hosted_gpt_model


def _slugify_tag(text: Any) -> str:
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    tag = tag.strip("_")
    return tag or "unknown"


def _short_hash(text: Any) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]

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
    examiner_tag = args.examiner_model

    if args.examiner_mode != "fixed":
        resolved_oracle_model, resolved_oracle_base_url = _resolve_oracle_judge_model_and_base_url(args)
        examiner_tag = (
            f"{args.examiner_mode}_"
            f"{_slugify_tag(args.oracle_pool)}_"
            f"{_slugify_tag(resolved_oracle_model)}_"
            f"{_short_hash(resolved_oracle_base_url)}"
        )

    if args.naive_run:
        log_file = (
            f'./logs/{args.task}/{args.guesser_model}_as_guesser/'
            f'{args.dataset}_'
            f'examiner_{examiner_tag}_{"" if args.inform else "un"}inform'
            f'_maxturn{args.max_turn}_{args.task_start_index}-{args.task_end_index}{args.add_info}.json'
        )
    else:
        log_file = (
            f'./logs/{args.task}/{args.guesser_model}_as_guesser/'
            f'{f"OS_init{args.open_set_size}_renew{args.size_to_renew}_" if args.open_set_size > 0 else ""}'
            f'{f"pre{args.n_pre_ask}_" if args.n_pre_ask > 0 else ""}'
            f'{args.dataset}_{args.temperature}_lambda{args.reward_lambda}_acc{not args.none_acc_reward}'
            f'_exp{args.expected_reward_method}_L{args.n_extend_layers}_K{args.n_potential_actions}'
            f'_PRUN{args.n_pruned_nodes}_{"" if args.inform else "un"}inform_EXAMINER{examiner_tag}'
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

    if model_name in {"gpt-5", "gpt-5-mini", "gpt-4", "gpt-3.5-turbo"}:
        return None

    if model_name in mapping:
        return mapping[model_name]

    return None

def _resolve_oracle_judge_model_and_base_url(args) -> tuple[str, str]:
    """
    Resolve oracle_judge_model so that aliases in src.uot.models also work.

    Returns:
        resolved_model_name_for_vllm, resolved_base_url
    """
    raw_model = str(args.oracle_judge_model).strip() if args.oracle_judge_model is not None else ""
    raw_base_url = str(args.oracle_base_url).strip()

    if not raw_model:
        raise ValueError("oracle_judge_model must be provided for oracle mode.")

    if is_hosted_gpt_model(raw_model):
        resolved_base_url = raw_base_url if raw_base_url else os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        return raw_model, resolved_base_url

    local_cfg = get_local_model_config(raw_model)
    if local_cfg is not None:
        # Alias or full local path recognized by models.py
        resolved_model = local_cfg.served_model_name
        resolved_base_url = raw_base_url if raw_base_url else local_cfg.base_url
        return resolved_model, resolved_base_url

    # otherwise leave untouched, for example:
    # - remote API-exposed model names
    # - explicit full model identifiers already matching the server
    return raw_model, raw_base_url


def _resolve_oracle_cache_path(args, resolved_model: str, resolved_base_url: str) -> str:
    raw_cache_path = str(args.oracle_cache_path).strip() if args.oracle_cache_path is not None else ""
    if raw_cache_path and raw_cache_path != "judge_cache.json":
        return raw_cache_path

    return (
        f"./judge_caches/{args.task}/"
        f"{_slugify_tag(args.examiner_mode)}_"
        f"{_slugify_tag(args.oracle_pool)}_"
        f"{_slugify_tag(resolved_model)}_"
        f"{_short_hash(resolved_base_url)}.json"
    )


def _build_run_config(args) -> Dict[str, Any]:
    resolved_oracle_model = None
    resolved_oracle_base_url = None
    resolved_oracle_cache_path = None

    if args.examiner_mode != "fixed":
        resolved_oracle_model, resolved_oracle_base_url = _resolve_oracle_judge_model_and_base_url(args)
        resolved_oracle_cache_path = _resolve_oracle_cache_path(
            args,
            resolved_oracle_model,
            resolved_oracle_base_url,
        )

    return {
        "task": args.task,
        "dataset": args.dataset,
        "guesser_model": args.guesser_model,
        "examiner_model": args.examiner_model,
        "examiner_mode": args.examiner_mode,
        "naive_run": bool(args.naive_run),
        "inform": bool(args.inform),
        "max_turn": args.max_turn,
        "oracle_pool": None if args.examiner_mode == "fixed" else args.oracle_pool,
        "oracle_judge_model": resolved_oracle_model,
        "oracle_base_url": resolved_oracle_base_url,
        "oracle_cache_path": resolved_oracle_cache_path,
        "oracle_match_prob": None if args.examiner_mode == "fixed" else args.oracle_match_prob,
        "oracle_mismatch_prob": None if args.examiner_mode == "fixed" else args.oracle_mismatch_prob,
        "oracle_pass_prob": None if args.examiner_mode == "fixed" else args.oracle_pass_prob,
        "oracle_judge_batch_size": None if args.examiner_mode == "fixed" else args.oracle_judge_batch_size,
        "oracle_topk_prune": None if args.examiner_mode == "fixed" else args.oracle_topk_prune,
        "oracle_support_top_mass": None if args.examiner_mode == "fixed" else args.oracle_support_top_mass,
        "oracle_support_min_prob": None if args.examiner_mode == "fixed" else args.oracle_support_min_prob,
        "oracle_force_include_topk": None if args.examiner_mode == "fixed" else args.oracle_force_include_topk,
    }


def _build_run_signature(run_config: Dict[str, Any]) -> str:
    payload = json.dumps(run_config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _validate_oracle_dataset_alignment(task, args) -> None:
    if getattr(args, "examiner_mode", "fixed") == "fixed":
        return

    if args.oracle_pool not in POOL_MAP:
        raise ValueError(f"Unknown oracle_pool: {args.oracle_pool}")

    pool_entities = {normalize_entity(x) for x in POOL_MAP[args.oracle_pool]}
    missing = [
        row["target"] for row in task.data
        if normalize_entity(row["target"]) not in pool_entities
    ]
    if missing:
        preview = ", ".join(str(x) for x in missing[:5])
        raise ValueError(
            "Oracle mode is pool-defined, but the selected dataset is not contained "
            f"in oracle_pool={args.oracle_pool}. Missing examples include: {preview}"
        )


def _validate_existing_logs(
    logs: List[Dict[str, Any]],
    run_signature: str,
    run_config: Dict[str, Any],
    original_start_index: int,
    log_file: str,
) -> None:
    for offset, row in enumerate(logs):
        expected_index = original_start_index + offset
        actual_index = row.get("index")
        if actual_index != expected_index:
            raise ValueError(
                f"Existing log {log_file} is not contiguous from start index "
                f"{original_start_index}: expected index {expected_index}, found {actual_index}."
            )

        existing_signature = row.get("run_config_signature")
        if existing_signature != run_signature:
            raise ValueError(
                f"Existing log {log_file} does not match the current oracle/runtime configuration. "
                "Use a different --add_info or remove the old log before resuming."
            )

        existing_config = row.get("run_config")
        if existing_config != run_config:
            raise ValueError(
                f"Existing log {log_file} was created with a different run configuration. "
                "Refusing to silently resume."
            )

def _load_tokenizer(args):
    if AutoTokenizer is None:
        print(
            "Warning: transformers is not installed. "
            "Falling back to approximate whitespace token counting."
        )
        return None

    candidate_models = [args.guesser_model]

    if args.examiner_mode == "fixed":
        candidate_models.append(args.examiner_model)
    else:
        resolved_oracle_model, _ = _resolve_oracle_judge_model_and_base_url(args)
        candidate_models.append(resolved_oracle_model)

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
            "oracle_state": l.get("oracle_state"),
            "oracle_final_target": l.get("oracle_final_target"),
            "oracle_mode": l.get("oracle_mode"),
            "oracle_pool_name": l.get("oracle_pool_name"),
            "oracle_episode_outcome": l.get("oracle_episode_outcome"),
            "oracle_accepted_guess": l.get("oracle_accepted_guess"),
            "oracle_accepted_guess_matches_item": l.get("oracle_accepted_guess_matches_item"),
            "run_config_signature": l.get("run_config_signature"),
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
            "avg_oracle_final_entropy": 0.0,
            "avg_oracle_support_size_final": 0.0,
            "avg_oracle_top1_prob_final": 0.0,
            "oracle_accept_count": 0,
            "oracle_timeout_count": 0,
            "oracle_accepted_guess_matches_item_count": 0,
        }

    num_success = sum(1 for l in logs if l.get("state") == 1)
    accuracy = num_success / num_samples
    avg_turn = sum(l.get("turn", 0) for l in logs) / num_samples

    oracle_final_entropies = []
    oracle_support_sizes = []
    oracle_top1_probs = []
    oracle_accept_count = 0
    oracle_timeout_count = 0
    oracle_item_match_count = 0

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

        oracle_state = l.get("oracle_state")
        if isinstance(oracle_state, dict) and oracle_state.get("final_entropy") is not None:
            oracle_final_entropies.append(float(oracle_state["final_entropy"]))
        if isinstance(oracle_state, dict) and oracle_state.get("support_size_final") is not None:
            oracle_support_sizes.append(float(oracle_state["support_size_final"]))
        if isinstance(oracle_state, dict) and oracle_state.get("posterior_top1_prob_final") is not None:
            oracle_top1_probs.append(float(oracle_state["posterior_top1_prob_final"]))

        if l.get("oracle_episode_outcome") == "accepted_guess":
            oracle_accept_count += 1
        if l.get("oracle_episode_outcome") == "max_turn":
            oracle_timeout_count += 1
        if l.get("oracle_accepted_guess_matches_item") is True:
            oracle_item_match_count += 1

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

        "avg_oracle_final_entropy": (
            sum(oracle_final_entropies) / len(oracle_final_entropies)
            if oracle_final_entropies else 0.0
        ),
        "avg_oracle_support_size_final": (
            sum(oracle_support_sizes) / len(oracle_support_sizes)
            if oracle_support_sizes else 0.0
        ),
        "avg_oracle_top1_prob_final": (
            sum(oracle_top1_probs) / len(oracle_top1_probs)
            if oracle_top1_probs else 0.0
        ),
        "oracle_accept_count": oracle_accept_count,
        "oracle_timeout_count": oracle_timeout_count,
        "oracle_accepted_guess_matches_item_count": oracle_item_match_count,
    }
    return metrics


def _attach_runtime_config_to_task(task, args):
    resolved_oracle_model, resolved_oracle_base_url = _resolve_oracle_judge_model_and_base_url(args)
    resolved_oracle_cache_path = _resolve_oracle_cache_path(
        args,
        resolved_oracle_model,
        resolved_oracle_base_url,
    )

    task.examiner_mode = args.examiner_mode
    task.oracle_pool = args.oracle_pool
    task.oracle_judge_model = resolved_oracle_model
    task.oracle_base_url = resolved_oracle_base_url
    task.oracle_api_key = args.oracle_api_key
    task.oracle_cache_path = resolved_oracle_cache_path
    task.oracle_match_prob = args.oracle_match_prob
    task.oracle_mismatch_prob = args.oracle_mismatch_prob
    task.oracle_pass_prob = args.oracle_pass_prob
    task.oracle_judge_batch_size = args.oracle_judge_batch_size
    task.oracle_topk_prune = args.oracle_topk_prune
    task.oracle_support_top_mass = args.oracle_support_top_mass
    task.oracle_support_min_prob = args.oracle_support_min_prob
    task.oracle_force_include_topk = args.oracle_force_include_topk
    task.oracle_examiner = None

def _build_oracle_examiner_if_needed(task):
    if getattr(task, "examiner_mode", "fixed") == "fixed":
        task.oracle_examiner = None
        return

    mode = "adv" if task.examiner_mode == "adv_oracle" else "cop"

    if is_hosted_gpt_model(task.oracle_judge_model):
        judge = HostedGPTJudge(
            model=task.oracle_judge_model,
            cache_path=task.oracle_cache_path,
            temperature=0.0,
            max_tokens=32768,
            sleep_s=0.0,
            timeout=120.0,
            max_retries=5,
        )
    else:
        judge = LocalVLLMJudge(
            model=task.oracle_judge_model,
            base_url=task.oracle_base_url,
            api_key=task.oracle_api_key,
            cache_path=task.oracle_cache_path,
            temperature=0.0,
            max_tokens=32768,
            sleep_s=0.0,
            timeout=120.0,
            max_retries=5,
        )

    task.oracle_examiner = ComparativeEntropyOracle(
        cfg=OracleConfig(
            pool_name=task.oracle_pool,
            mode=mode,
            match_prob=task.oracle_match_prob,
            mismatch_prob=task.oracle_mismatch_prob,
            pass_prob=task.oracle_pass_prob,
            judge_batch_size=task.oracle_judge_batch_size,
            topk_prune=task.oracle_topk_prune,
            support_top_mass=task.oracle_support_top_mass,
            support_min_prob=task.oracle_support_min_prob,
            force_include_topk=task.oracle_force_include_topk,
        ),
        judge=judge,
    )


def run(args):
    task = get_task(args)
    _attach_runtime_config_to_task(task, args)
    _validate_oracle_dataset_alignment(task, args)

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
    run_config = _build_run_config(args)
    run_signature = _build_run_signature(run_config)

    if len(logs) > 0:
        _validate_existing_logs(
            logs,
            run_signature=run_signature,
            run_config=run_config,
            original_start_index=original_start_index,
            log_file=log_file,
        )
        resumed_start = original_start_index + len(logs)
        if resumed_start > args.task_end_index:
            resumed_start = args.task_end_index
        print(
            f"Found existing log with {len(logs)} samples. "
            f"Resuming from index {resumed_start}."
        )
        args.task_start_index = resumed_start

    for i in tqdm(range(args.task_start_index, args.task_end_index)):
        _build_oracle_examiner_if_needed(task)

        log = naive_converse(task, i)
        log["run_config"] = run_config
        log["run_config_signature"] = run_signature

        logs.append(log)
        _safe_save_logs(log_file, logs)

        cot_logs = _extract_cot_logs(logs)
        _safe_save_logs(cot_log_file, cot_logs)

        if getattr(task, "oracle_examiner", None) is not None:
            flush = getattr(task.oracle_examiner.judge, "flush", None)
            if callable(flush):
                flush()

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
        '--examiner_mode',
        type=str,
        default='fixed',
        choices=['fixed', 'adv_oracle', 'cop_oracle']
    )
    args.add_argument('--oracle_pool', type=str, default='common')
    # args.add_argument('--oracle_judge_model', type=str, default='/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507')
    args.add_argument('--oracle_judge_model', type=str, default='')
    args.add_argument('--oracle_base_url', type=str, default='http://127.0.0.1:8000/v1')
    args.add_argument('--oracle_api_key', type=str, default='EMPTY')
    args.add_argument('--oracle_cache_path', type=str, default='judge_cache.json')
    args.add_argument('--oracle_match_prob', type=float, default=0.98)
    args.add_argument('--oracle_mismatch_prob', type=float, default=0.02)
    args.add_argument('--oracle_pass_prob', type=float, default=0.50)
    args.add_argument('--oracle_judge_batch_size', type=int, default=8)
    args.add_argument('--oracle_topk_prune', type=int, default=0)

    args.add_argument('--oracle_support_top_mass', type=float, default=0.95)
    args.add_argument('--oracle_support_min_prob', type=float, default=1e-4)
    args.add_argument('--oracle_force_include_topk', type=int, default=0)

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
        choices=['common', 'thing', 'bigbench']
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