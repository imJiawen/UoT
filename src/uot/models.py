import os
import time
import copy
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

# ============================================================================
# API keys / clients
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CLAUDE2_API_KEY = os.getenv("CLAUDE2_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", "")

QWEN_INSTRUCT_4B_API_KEY = os.getenv("QWEN_INSTRUCT_4B_API_KEY", "")
QWEN_INSTRUCT_4B_IP = os.getenv("QWEN_INSTRUCT_4B_IP", "")
QWEN_INSTRUCT_4B_PORT = os.getenv("QWEN_INSTRUCT_4B_PORT", "")

QWEN_THINKING_4B_API_KEY = os.getenv("QWEN_THINKING_4B_API_KEY", "")
QWEN_THINKING_4B_IP = os.getenv("QWEN_THINKING_4B_IP", "")
QWEN_THINKING_4B_PORT = os.getenv("QWEN_THINKING_4B_PORT", "")

QWEN_THINKING_30B_API_KEY = os.getenv("QWEN_THINKING_30B_API_KEY", "")
QWEN_THINKING_30B_IP = os.getenv("QWEN_THINKING_30B_IP", "")
QWEN_THINKING_30B_PORT = os.getenv("QWEN_THINKING_30B_PORT", "")

GPT_OSS_20B_API_KEY = os.getenv("GPT_OSS_20B_API_KEY", "")
GPT_OSS_20B_IP = os.getenv("GPT_OSS_20B_IP", "")
GPT_OSS_20B_PORT = os.getenv("GPT_OSS_20B_PORT", "")

QWEN_INSTRUCT_30B_API_KEY = os.getenv("QWEN_INSTRUCT_30B_API_KEY", "")
QWEN_INSTRUCT_30B_IP = os.getenv("QWEN_INSTRUCT_30B_IP", "")
QWEN_INSTRUCT_30B_PORT = os.getenv("QWEN_INSTRUCT_30B_PORT", "")

# ============================================================================
# Unified generation defaults
# ============================================================================

MAX_TOKENS = 32768

DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": MAX_TOKENS,
    "reasoning_effort": "medium",
}

time_gap = {
    "gpt-4": 3,
    "gpt-3.5-turbo": 0.5,
    "claude-3-opus-20240229": 2,
    "claude-3-sonnet-20240229": 2,
    "gemini-1.0-pro": 2,
    "mistral-small-latest": 1,
    "mistral-medium-latest": 1,
    "mistral-large-latest": 1,
    "qwen_instruct_4b": 1,
    "qwen_thinking_4b": 1,
    "qwen_thinking_30b": 1,
    "qwen_4b": 1,
    "qwen_30b": 1,
    "qwen_instruct_30b": 1,
    "gpt_oss_20b": 1,
    "qwen3-4b-instruct-local": 0,
    "qwen3-4b-local": 0,
    "qwen3-30b-local": 0,
    "qwen3-30b-instruct-local": 0,
    "llama3.1-8b-local": 0,
}

openai_client = None
qwen_instruct_4b_api_client = None
qwen_thinking_4b_api_client = None
qwen_thinking_30b_api_client = None
qwen_instruct_30b_api_client = None
gpt_oss_20b_api_client = None

co = None
genai = None
glm = None
anthropic = None
claude_client = None
llama_client = None
mistral_client = None
ChatMessage = None

if OPENAI_API_KEY != "":
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"OPENAI_API_KEY: ****{OPENAI_API_KEY[-4:]}")

if COHERE_API_KEY != "":
    import cohere
    co = cohere.Client(COHERE_API_KEY)
    print(f"COHERE_API_KEY: ****{COHERE_API_KEY[-4:]}")

if GOOGLE_API_KEY != "":
    import google.generativeai as genai
    import google.ai.generativelanguage as glm
    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"GOOGLE_API_KEY: ****{GOOGLE_API_KEY[-4:]}")

if CLAUDE2_API_KEY != "":
    from anthropic import Anthropic
    anthropic = Anthropic(api_key=CLAUDE2_API_KEY, base_url="https://api.aiproxy.io")
    print(f"CLAUDE2_API_KEY: ****{CLAUDE2_API_KEY[-4:]}")

if ANTHROPIC_API_KEY != "":
    from anthropic import Anthropic
    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print(f"ANTHROPIC_API_KEY: ****{ANTHROPIC_API_KEY[-4:]}")

if TOGETHER_API_KEY != "":
    from openai import OpenAI
    llama_client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz")
    print(f"TOGETHER_API_KEY: ****{TOGETHER_API_KEY[-4:]}")

if MISTRAL_API_KEY != "":
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
    print(f"MISTRAL_API_KEY: ****{MISTRAL_API_KEY[-4:]}")

if QWEN_INSTRUCT_4B_API_KEY != "" and QWEN_INSTRUCT_4B_IP != "" and QWEN_INSTRUCT_4B_PORT != "":
    from openai import OpenAI
    qwen_instruct_4b_api_client = OpenAI(
        base_url=f"{QWEN_INSTRUCT_4B_IP}:{QWEN_INSTRUCT_4B_PORT}/v1",
        api_key=QWEN_INSTRUCT_4B_API_KEY,
    )
    print(f"QWEN_INSTRUCT_4B_API_KEY: ****{QWEN_INSTRUCT_4B_API_KEY[-4:]}")

if QWEN_THINKING_4B_API_KEY != "" and QWEN_THINKING_4B_IP != "" and QWEN_THINKING_4B_PORT != "":
    from openai import OpenAI
    qwen_thinking_4b_api_client = OpenAI(
        base_url=f"{QWEN_THINKING_4B_IP}:{QWEN_THINKING_4B_PORT}/v1",
        api_key=QWEN_THINKING_4B_API_KEY,
    )
    print(f"QWEN_THINKING_4B_API_KEY: ****{QWEN_THINKING_4B_API_KEY[-4:]}")

if QWEN_THINKING_30B_API_KEY != "" and QWEN_THINKING_30B_IP != "" and QWEN_THINKING_30B_PORT != "":
    from openai import OpenAI
    qwen_thinking_30b_api_client = OpenAI(
        base_url=f"{QWEN_THINKING_30B_IP}:{QWEN_THINKING_30B_PORT}/v1",
        api_key=QWEN_THINKING_30B_API_KEY,
    )
    print(f"QWEN_THINKING_30B_API_KEY: ****{QWEN_THINKING_30B_API_KEY[-4:]}")

if QWEN_INSTRUCT_30B_API_KEY != "" and QWEN_INSTRUCT_30B_IP != "" and QWEN_INSTRUCT_30B_PORT != "":
    from openai import OpenAI
    qwen_instruct_30b_api_client = OpenAI(
        base_url=f"{QWEN_INSTRUCT_30B_IP}:{QWEN_INSTRUCT_30B_PORT}/v1",
        api_key=QWEN_INSTRUCT_30B_API_KEY,
    )
    print(f"QWEN_INSTRUCT_30B_API_KEY: ****{QWEN_INSTRUCT_30B_API_KEY[-4:]}")

if GPT_OSS_20B_API_KEY != "" and GPT_OSS_20B_IP != "" and GPT_OSS_20B_PORT != "":
    from openai import OpenAI
    gpt_oss_20b_api_client = OpenAI(
        base_url=f"{GPT_OSS_20B_IP}:{GPT_OSS_20B_PORT}/v1",
        api_key=GPT_OSS_20B_API_KEY,
    )
    print(f"GPT_OSS_20B_API_KEY: ****{GPT_OSS_20B_API_KEY[-4:]}")

# ============================================================================
# Model config
# ============================================================================

@dataclass
class LocalModelConfig:
    name: str
    family: str
    tokenizer_name_or_path: str
    served_model_name: str
    base_url: str
    enable_reasoning: bool = True
    thinking_start_token: str = ""
    thinking_end_token: str = "</think>"


LOCAL_VLLM_HOST = os.getenv("LOCAL_VLLM_HOST", "http://127.0.0.1").rstrip("/")

LOCAL_MODEL_PATHS = {
    "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507": {
        "family": "qwen",
        "path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
    },
    "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507": {
        "family": "qwen",
        "path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
    },
    "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507": {
        "family": "qwen",
        "path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
    },
    "/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct": {
        "family": "llama",
        "path": "/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct",
    },
    "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "family": "qwen",
        "path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
    },
    "/hpc2hdd/home/mpeng885/models/gpt-oss-20b": {
        "family": "gpt_oss",
        "path": "/hpc2hdd/home/mpeng885/models/gpt-oss-20b",
    },
}

LOCAL_MODEL_ALIASES = {
    "qwen3-4b-instruct-local": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b-local": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-30b-local": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
    "llama3.1-8b-local": "/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct",
    "qwen3-30b-instruct-local": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
}

REMOTE_QWEN_SPECS = {
    "qwen_instruct_4b": {
        "tokenizer_path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
        "server_model_name": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
        "client_name": "qwen_instruct_4b",
        "enable_reasoning": False,
        "thinking_start_token": "",
        "thinking_end_token": "",
    },
    "qwen_thinking_4b": {
        "tokenizer_path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        "server_model_name": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        "client_name": "qwen_thinking_4b",
        "enable_reasoning": True,
        "thinking_start_token": "",
        "thinking_end_token": "</think>",
    },
    "qwen_thinking_30b": {
        "tokenizer_path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "server_model_name": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "client_name": "qwen_thinking_30b",
        "enable_reasoning": True,
        "thinking_start_token": "",
        "thinking_end_token": "</think>",
    },
    "qwen_instruct_30b": {
        "tokenizer_path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "server_model_name": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "client_name": "qwen_instruct_30b",
        "enable_reasoning": False,
        "thinking_start_token": "",
        "thinking_end_token": "",
    },
    "qwen_4b": {
        "tokenizer_path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        "server_model_name": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        "client_name": "qwen_thinking_4b",
        "enable_reasoning": True,
        "thinking_start_token": "",
        "thinking_end_token": "</think>",
    },
    "qwen_30b": {
        "tokenizer_path": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "server_model_name": "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "client_name": "qwen_thinking_30b",
        "enable_reasoning": True,
        "thinking_start_token": "",
        "thinking_end_token": "</think>",
    },
}

LOCAL_MODEL_CONFIGS: Dict[str, LocalModelConfig] = {
    "qwen3-4b-instruct-local": LocalModelConfig(
        name="qwen3-4b-instruct-local",
        family="qwen",
        tokenizer_name_or_path="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
        served_model_name="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
        base_url=f"{LOCAL_VLLM_HOST}:{os.getenv('QWEN3_4B_INSTRUCT_LOCAL_PORT', '8011')}/v1",
        enable_reasoning=False,
        thinking_start_token="",
        thinking_end_token="",
    ),
    "qwen3-4b-local": LocalModelConfig(
        name="qwen3-4b-local",
        family="qwen",
        tokenizer_name_or_path="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        served_model_name="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
        base_url=f"{LOCAL_VLLM_HOST}:{os.getenv('QWEN3_4B_THINKING_LOCAL_PORT', '8012')}/v1",
        enable_reasoning=True,
        thinking_start_token="",
        thinking_end_token="</think>",
    ),
    "qwen3-30b-local": LocalModelConfig(
        name="qwen3-30b-local",
        family="qwen",
        tokenizer_name_or_path="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        served_model_name="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
        base_url=f"{LOCAL_VLLM_HOST}:{os.getenv('QWEN3_30B_THINKING_LOCAL_PORT', '8013')}/v1",
        enable_reasoning=True,
        thinking_start_token="",
        thinking_end_token="</think>",
    ),
    "qwen3-30b-instruct-local": LocalModelConfig(
        name="qwen3-30b-instruct-local",
        family="qwen",
        tokenizer_name_or_path="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
        served_model_name="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
        base_url=f"{LOCAL_VLLM_HOST}:{os.getenv('QWEN3_30B_INSTRUCT_LOCAL_PORT', '8014')}/v1",
        enable_reasoning=False,
        thinking_start_token="",
        thinking_end_token="",
    ),
    "llama3.1-8b-local": LocalModelConfig(
        name="llama3.1-8b-local",
        family="llama",
        tokenizer_name_or_path="/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct",
        served_model_name="/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct",
        base_url=f"{LOCAL_VLLM_HOST}:{os.getenv('LLAMA31_8B_LOCAL_PORT', '8015')}/v1",
        enable_reasoning=False,
        thinking_start_token="",
        thinking_end_token="",
    ),
}

LOCAL_MODEL_CONFIGS["/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507"] = LOCAL_MODEL_CONFIGS["qwen3-4b-instruct-local"]
LOCAL_MODEL_CONFIGS["/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507"] = LOCAL_MODEL_CONFIGS["qwen3-4b-local"]
LOCAL_MODEL_CONFIGS["/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507"] = LOCAL_MODEL_CONFIGS["qwen3-30b-local"]
LOCAL_MODEL_CONFIGS["/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507"] = LOCAL_MODEL_CONFIGS["qwen3-30b-instruct-local"]
LOCAL_MODEL_CONFIGS["/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct"] = LOCAL_MODEL_CONFIGS["llama3.1-8b-local"]

_LOCAL_CLIENT_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}

# ============================================================================
# Helpers
# ============================================================================

def _sleep_for_model(model: str):
    time.sleep(time_gap.get(model, 1))


def _resolve_local_model_name(model: str) -> str:
    if model in LOCAL_MODEL_PATHS:
        return model
    if model in LOCAL_MODEL_ALIASES:
        return LOCAL_MODEL_ALIASES[model]
    return model


def get_local_model_config(model: str) -> Optional[LocalModelConfig]:
    resolved = _resolve_local_model_name(model)
    return LOCAL_MODEL_CONFIGS.get(model) or LOCAL_MODEL_CONFIGS.get(resolved)


def _get_or_create_openai_client(base_url: str, api_key: str = "EMPTY"):
    from openai import OpenAI

    cache_key = f"{base_url}|{api_key}"
    if cache_key not in _LOCAL_CLIENT_CACHE:
        _LOCAL_CLIENT_CACHE[cache_key] = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=1200.0,
        )
    return _LOCAL_CLIENT_CACHE[cache_key]


def _load_tokenizer_cached(tokenizer_path: str):
    from transformers import AutoTokenizer

    if tokenizer_path not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[tokenizer_path] = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=HUGGINGFACE_HUB_CACHE if HUGGINGFACE_HUB_CACHE else None,
        )
    return _TOKENIZER_CACHE[tokenizer_path]


def _count_tokens_with_tokenizer(text: str, tokenizer=None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(str(text).split())


def _get_remote_qwen_client(model: str):
    if model not in REMOTE_QWEN_SPECS:
        raise ValueError(f"Unsupported remote qwen model: {model}")

    client_name = REMOTE_QWEN_SPECS[model]["client_name"]
    if client_name == "qwen_instruct_4b":
        client = qwen_instruct_4b_api_client
    elif client_name == "qwen_instruct_30b":
        client = qwen_instruct_30b_api_client
    elif client_name == "qwen_thinking_4b":
        client = qwen_thinking_4b_api_client
    elif client_name == "qwen_thinking_30b":
        client = qwen_thinking_30b_api_client
    else:
        client = None

    if client is None:
        raise RuntimeError(f"Remote client for {model} is not initialized.")
    return client


def _apply_chat_template_safely(tokenizer, message, family: str, enable_reasoning: bool = True):
    if family == "qwen":
        if enable_reasoning:
            try:
                return tokenizer.apply_chat_template(
                    conversation=message,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_reasoning=True,
                    add_special_tokens=True,
                )
            except TypeError:
                pass

        try:
            return tokenizer.apply_chat_template(
                conversation=message,
                add_generation_prompt=True,
                tokenize=False,
                add_special_tokens=True,
            )
        except TypeError:
            pass

    try:
        return tokenizer.apply_chat_template(
            conversation=message,
            add_generation_prompt=True,
            tokenize=False,
        )
    except TypeError:
        pass

    try:
        return tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        pass

    text = ""
    for m in message:
        role = m.get("role", "user")
        content = m.get("content", "")
        text += f"{role}: {content}\n"
    text += "assistant: "
    return text


def _strip_start_token(text: str, start_token: str) -> str:
    if not text or not start_token:
        return text.strip()
    t = text.strip()
    if t.startswith(start_token):
        return t[len(start_token):].strip()
    idx = t.find(start_token)
    if idx >= 0:
        return t[idx + len(start_token):].strip()
    return t


def _split_reasoning_text(
    response_text: str,
    tokenizer=None,
    thinking_start_token: str = "",
    thinking_end_token: str = "</think>",
):
    response_text = (response_text or "").strip()
    if not response_text:
        return {
            "text": "",
            "cot": "",
            "num_thinking_tokens": 0,
        }

    cot = ""
    final_output = response_text

    if thinking_end_token and thinking_end_token in response_text:
        cot, final_output = response_text.split(thinking_end_token, 1)
        cot = _strip_start_token(cot, thinking_start_token)
        final_output = final_output.strip()
    else:
        final_output = response_text.strip()

    num_thinking_tokens = _count_tokens_with_tokenizer(cot, tokenizer) if cot else 0

    return {
        "text": final_output,
        "cot": cot,
        "num_thinking_tokens": num_thinking_tokens,
    }


_GPT_OSS_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")
_GPT_OSS_CHANNEL_BLOCK_RE = re.compile(
    r"<\|channel\|>(?P<channel>[^<]+?)(?:<\|constrain\|>[^<]+)?<\|message\|>(?P<body>.*?)(?=(?:<\|start\|>|<\|end\|>|$))",
    re.DOTALL,
)


def _clean_gpt_oss_payload(text: str) -> str:
    if not text:
        return ""
    s = _GPT_OSS_CONTROL_TOKEN_RE.sub(" ", text)
    s = s.replace("\\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _coerce_chat_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and item["text"] is not None:
                    parts.append(str(item["text"]))
                elif "content" in item and item["content"] is not None:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if "text" in content and content["text"] is not None:
            return str(content["text"])
        if "content" in content and content["content"] is not None:
            return str(content["content"])
    return str(content)


def _extract_gpt_oss_text_from_tool_calls(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    candidates = []
    for call in tool_calls:
        function = getattr(call, "function", None)
        if function is None and isinstance(call, dict):
            function = call.get("function")
        if function is None:
            continue

        arguments = getattr(function, "arguments", None)
        if arguments is None and isinstance(function, dict):
            arguments = function.get("arguments")
        if arguments is None:
            continue

        arg_text = str(arguments)
        candidates.append(arg_text)
    if not candidates:
        return ""
    return _clean_gpt_oss_payload("\n".join(candidates))


def extract_gpt_oss_content_and_cot(raw_content: Any, tool_calls: Any = None) -> Tuple[str, str]:
    text = _coerce_chat_content_to_text(raw_content)
    if not text:
        tool_text = _extract_gpt_oss_text_from_tool_calls(tool_calls)
        return tool_text, ""

    channel_blocks = _GPT_OSS_CHANNEL_BLOCK_RE.findall(text)
    if channel_blocks:
        final_parts = []
        commentary_parts = []
        analysis_parts = []
        tool_parts = []
        for channel_raw, body_raw in channel_blocks:
            channel = channel_raw.strip().lower()
            body = _clean_gpt_oss_payload(body_raw)
            if not body:
                continue
            if " to=" in channel:
                tool_parts.append(body)
            if channel.startswith("final"):
                final_parts.append(body)
            elif channel.startswith("commentary"):
                commentary_parts.append(body)
            elif channel.startswith("analysis"):
                analysis_parts.append(body)

        cot = "\n\n".join(analysis_parts).strip()
        if final_parts:
            return final_parts[-1], cot
        if commentary_parts:
            return commentary_parts[-1], cot
        if tool_parts:
            return tool_parts[-1], cot

        tool_text = _extract_gpt_oss_text_from_tool_calls(tool_calls)
        if tool_text:
            return tool_text, cot
        return "", cot

    return _clean_gpt_oss_payload(text), ""


def _usage_get_int(obj: Any, *keys: str) -> int:
    for key in keys:
        value = None
        if isinstance(obj, dict):
            value = obj.get(key)
        else:
            value = getattr(obj, key, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                continue
    return 0


def _extract_reasoning_tokens_from_usage(usage: Any) -> int:
    if usage is None:
        return 0

    direct = _usage_get_int(usage, "reasoning_tokens", "reasoningTokenCount")
    if direct > 0:
        return direct

    details_candidates = []
    if isinstance(usage, dict):
        details_candidates.extend([
            usage.get("completion_tokens_details"),
            usage.get("output_tokens_details"),
        ])
    else:
        details_candidates.extend([
            getattr(usage, "completion_tokens_details", None),
            getattr(usage, "output_tokens_details", None),
        ])

    for details in details_candidates:
        if details is None:
            continue
        value = _usage_get_int(details, "reasoning_tokens", "reasoningTokenCount")
        if value > 0:
            return value

    return 0


def _request_completion_with_fallback(
    client,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str = "medium",
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    extra_kwargs = extra_kwargs or {}

    request_kwargs = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    request_kwargs.update(extra_kwargs)

    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort

    try:
        return client.completions.create(**request_kwargs)
    except TypeError:
        request_kwargs.pop("reasoning_effort", None)
        request_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        return client.completions.create(**request_kwargs)


def _generate_vllm_completion(
    message: list,
    config: LocalModelConfig,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    tokenizer = _load_tokenizer_cached(config.tokenizer_name_or_path)
    raw_prompt_text = _apply_chat_template_safely(
        tokenizer,
        message,
        family=config.family,
        enable_reasoning=config.enable_reasoning,
    )

    client = _get_or_create_openai_client(config.base_url, api_key="EMPTY")
    res = _request_completion_with_fallback(
        client=client,
        model=config.served_model_name,
        prompt=raw_prompt_text,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort if config.enable_reasoning else None,
        extra_kwargs={"logprobs": 1},
    )

    response_text = res.choices[0].text if res.choices else ""
    parsed = _split_reasoning_text(
        response_text=response_text,
        tokenizer=tokenizer,
        thinking_start_token=config.thinking_start_token,
        thinking_end_token=config.thinking_end_token,
    )

    return res, parsed


def unpack_model_response(ret):
    """
    Normalize model return format to:
    (raw_response, answer, cot, think_tokens)
    """
    if isinstance(ret, tuple):
        if len(ret) == 4:
            return ret
        if len(ret) == 3:
            raw, answer, cot = ret
            return raw, answer, cot, 0
        if len(ret) == 2:
            raw, answer = ret
            return raw, answer, "", 0
        if len(ret) == 1:
            return None, str(ret[0]), "", 0

    if isinstance(ret, str):
        return None, ret, "", 0

    return None, str(ret), "", 0


# ============================================================================
# Remote API model responses
# ============================================================================

def gpt_response(
    message: list,
    model="gpt-4",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not set, but GPT model was requested.")

    _sleep_for_model(model)
    try:
        kwargs = {
            "model": model,
            "messages": message,
            "temperature": temperature,
            "n": 1,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p

        res = openai_client.chat.completions.create(**kwargs)
        try:
            text = res.choices[0].message.content
        except Exception:
            text = res.choices[0].text
        return res, text, "", 0
    except Exception as e:
        print(e)
        time.sleep(time_gap.get(model, 3) * 2)
        return gpt_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def cohere_response(
    message: list,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    if co is None:
        raise RuntimeError("COHERE_API_KEY is not set, but Cohere model was requested.")

    msg = copy.deepcopy(message[:-1])
    new_msg = message[-1]["content"]
    for m in msg:
        m.update({"role": "CHATBOT" if m["role"] in ["system", "assistant"] else "USER"})
        m.update({"message": m.pop("content")})

    try:
        text = co.chat(chat_history=msg, message=new_msg).text
        return None, text, "", 0
    except Exception as e:
        print(e)
        time.sleep(1)
        return cohere_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def palm_response(
    message: list,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    raise NotImplementedError("PaLM is not enabled in this file. Use Gemini instead.")


def gemini_response(
    message: list,
    model="gemini-1.0-pro",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    if genai is None or glm is None:
        raise RuntimeError("GOOGLE_API_KEY is not set, but Gemini model was requested.")

    genai_model = genai.GenerativeModel(model_name=model)
    try:
        chat = genai_model.start_chat()
        res = chat.send_message(message[-1]["content"])
        return res, res.text, "", 0
    except Exception as e:
        print(e)
        time.sleep(3)
        return gemini_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def claude_aiproxy_response(
    message,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    if anthropic is None:
        raise RuntimeError("CLAUDE2_API_KEY is not set, but Claude-2 proxy model was requested.")

    from anthropic import HUMAN_PROMPT, AI_PROMPT

    prompt = ""
    for m in message:
        prompt += AI_PROMPT if m["role"] in ["system", "assistant"] else HUMAN_PROMPT
        prompt += " " + m["content"]
    prompt += AI_PROMPT
    try:
        res = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
            prompt=prompt,
        )
        return res, res.completion, "", 0
    except Exception as e:
        print(e)
        time.sleep(1)
        return claude_aiproxy_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def claude_response(
    message,
    model="claude-3-sonnet-20240229",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    if claude_client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set, but Claude model was requested.")

    msg = []
    for m in message:
        role = m["role"] if m["role"] == "user" else "assistant"
        if msg and msg[-1]["role"] == role:
            msg[-1]["content"] += m["content"]
        else:
            msg.append({"role": role, "content": m["content"]})
    try:
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": msg,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        res = claude_client.messages.create(**kwargs)
        return res, res.content[0].text, "", 0
    except Exception as e:
        print(e)
        time.sleep(3)
        return claude_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def llama_response(
    message,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    if llama_client is None:
        raise RuntimeError("TOGETHER_API_KEY is not set, but remote Llama model was requested.")

    try:
        chat_completion = llama_client.chat.completions.create(
            messages=message,
            model="meta-llama/Llama-2-70b-chat-hf",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return chat_completion, chat_completion.choices[0].message.content, "", 0
    except Exception as e:
        print(e)
        time.sleep(1)
        return llama_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def mistral_response(
    message: list,
    model="mistral-large-latest",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    if mistral_client is None or ChatMessage is None:
        raise RuntimeError("MISTRAL_API_KEY is not set, but Mistral model was requested.")

    msg = [ChatMessage(role=m["role"], content=m["content"]) for m in message]
    try:
        res = mistral_client.chat(model=model, messages=msg)
        return res, res.choices[0].message.content, "", 0
    except Exception as e:
        print(e)
        time.sleep(1)
        return mistral_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def qwen_response(
    message: list,
    model="qwen_thinking_4b",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    if model not in REMOTE_QWEN_SPECS:
        raise ValueError(f"Unsupported qwen remote model: {model}")

    spec = REMOTE_QWEN_SPECS[model]
    tokenizer_path = spec["tokenizer_path"]
    server_model_name = spec["server_model_name"]
    enable_reasoning = spec["enable_reasoning"]
    thinking_start_token = spec.get("thinking_start_token", "")
    thinking_end_token = spec.get("thinking_end_token", "</think>")
    client = _get_remote_qwen_client(model)

    try:
        qwen_tokenizer = _load_tokenizer_cached(tokenizer_path)
        raw_prompt_text = _apply_chat_template_safely(
            qwen_tokenizer,
            message,
            family="qwen",
            enable_reasoning=enable_reasoning,
        )

        chat_completion = _request_completion_with_fallback(
            client=client,
            model=server_model_name,
            prompt=raw_prompt_text,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort if enable_reasoning else None,
            extra_kwargs={"logprobs": 1},
        )

        response_text = chat_completion.choices[0].text if chat_completion.choices else ""
        parsed = _split_reasoning_text(
            response_text=response_text,
            tokenizer=qwen_tokenizer,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        )
        return chat_completion, parsed["text"], parsed["cot"], parsed["num_thinking_tokens"]

    except Exception as e:
        print(e)
        time.sleep(1)
        return qwen_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def gemma_response(
    message: list,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    import uot.model_gemma as gm
    try:
        text = gm.gemma_response(history=message, output_len=max_tokens)
        return None, text, "", 0
    except Exception as e:
        print(e)
        time.sleep(1)
        return gemma_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def gpt_oss_20b_response(
    message: list,
    model="gpt_oss_20b",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    if gpt_oss_20b_api_client is None:
        raise RuntimeError("Remote client for gpt_oss_20b is not initialized.")

    server_model_name = "/hpc2hdd/home/mpeng885/models/gpt-oss-20b"

    try:
        request_kwargs = {
            "model": server_model_name,
            "messages": message,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        try:
            res = gpt_oss_20b_api_client.chat.completions.create(**request_kwargs)
        except TypeError:
            request_kwargs.pop("reasoning_effort", None)
            request_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
            res = gpt_oss_20b_api_client.chat.completions.create(**request_kwargs)

        raw_content = getattr(res.choices[0].message, "content", "")
        raw_tool_calls = getattr(res.choices[0].message, "tool_calls", None)
        final_output, cot = extract_gpt_oss_content_and_cot(raw_content, tool_calls=raw_tool_calls)

        num_thinking_tokens = 0
        usage = getattr(res, "usage", None)
        if usage is not None:
            num_thinking_tokens = _extract_reasoning_tokens_from_usage(usage)

        if num_thinking_tokens == 0 and cot:
            tok = None
            try:
                tok = _load_tokenizer_cached("/hpc2hdd/home/mpeng885/models/gpt-oss-20b")
            except Exception:
                tok = None
            num_thinking_tokens = _count_tokens_with_tokenizer(cot, tok)

        return res, final_output, cot, num_thinking_tokens
    except Exception as e:
        print(e)
        time.sleep(1)
        return gpt_oss_20b_response(message, model, temperature, top_p, max_tokens, reasoning_effort)

# ============================================================================
# Local model responses via vLLM
# ============================================================================

def _local_vllm_response(
    message: list,
    model: str,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    config = get_local_model_config(model)
    if config is None:
        raise ValueError(f"Unknown local model: {model}")

    try:
        res, parsed = _generate_vllm_completion(
            message=message,
            config=config,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return res, parsed["text"], parsed["cot"], parsed["num_thinking_tokens"]
    except Exception as e:
        print(e)
        time.sleep(1)
        return _local_vllm_response(
            message=message,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )


def local_qwen_30b_instruct_response(
    message: list,
    model="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def local_qwen_4b_instruct_response(
    message: list,
    model="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def local_qwen_4b_response(
    message: list,
    model="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def local_qwen_30b_response(
    message: list,
    model="/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)


def local_llama31_8b_response(
    message: list,
    model="/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct",
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    return _local_vllm_response(message, model, temperature, top_p, max_tokens, reasoning_effort)

# ============================================================================
# Router
# ============================================================================

def _unsupported_model_response(
    message,
    model=None,
    temperature=DEFAULT_GENERATION_CONFIG["temperature"],
    top_p=DEFAULT_GENERATION_CONFIG["top_p"],
    max_tokens=DEFAULT_GENERATION_CONFIG["max_tokens"],
    reasoning_effort=DEFAULT_GENERATION_CONFIG["reasoning_effort"],
):
    raise NotImplementedError(f"Unsupported model: {model}")


def get_response_method(model):
    exact_methods = {
        "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Instruct-2507": local_qwen_4b_instruct_response,
        "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-4B-Thinking-2507": local_qwen_4b_response,
        "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Thinking-2507": local_qwen_30b_response,
        "/hpc2hdd/home/mpeng885/models/LLaMa/Meta-Llama-3.1-8B-Instruct": local_llama31_8b_response,
        "/hpc2hdd/home/mpeng885/models/Qwen/Qwen3-30B-A3B-Instruct-2507": local_qwen_30b_instruct_response,

        "qwen3-4b-instruct-local": local_qwen_4b_instruct_response,
        "qwen3-4b-local": local_qwen_4b_response,
        "qwen3-30b-local": local_qwen_30b_response,
        "llama3.1-8b-local": local_llama31_8b_response,
        "qwen3-30b-instruct-local": local_qwen_30b_instruct_response,

        "qwen_instruct_4b": qwen_response,
        "qwen_thinking_4b": qwen_response,
        "qwen_thinking_30b": qwen_response,
        "qwen_instruct_30b": qwen_response,

        "qwen_4b": qwen_response,
        "qwen_30b": qwen_response,
    }

    if model in exact_methods:
        return exact_methods[model]

    model_lower = str(model).lower()

    if model_lower.startswith("gpt"):
        if model_lower == "gpt_oss_20b":
            return gpt_oss_20b_response
        return gpt_response
    if model_lower.startswith("cohere"):
        return cohere_response
    if model_lower.startswith("palm"):
        return palm_response
    if model_lower.startswith("_claude"):
        return claude_aiproxy_response
    if model_lower.startswith("claude"):
        return claude_response
    if model_lower.startswith("llama"):
        return llama_response
    if model_lower.startswith("mistral"):
        return mistral_response
    if model_lower.startswith("gemma"):
        return gemma_response
    if model_lower.startswith("gemini"):
        return gemini_response
    if model_lower.startswith("qwen_"):
        return qwen_response

    def _unsupported(*args, **kwargs):
        raise NotImplementedError(f"Unsupported model: {model}")

    return _unsupported