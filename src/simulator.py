import time
import tiktoken
from dataclasses import dataclass

# --- tunable timing model (the "GPU personality") ---
PREFILL_MS_PER_TOKEN = 0.5    
DECODE_MS_PER_TOKEN  = 30     
DEFAULT_MAX_TOKENS   = 64
TOKENS_PER_MESSAGE   = 3    

# --- expensive setup ONCE at import, not per request ---
_ENCODING = tiktoken.get_encoding("o200k_base")   # your fallback default

def _get_encoding(model: str):
    return tiktoken.encoding_for_model(model)

@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    prefill_ms: float
    decode_ms: float
    total_ms: float

def count_prompt_tokens(messages: list[dict], model: str) -> int:
    try:
        encoding = _get_encoding(model)
    except KeyError:
        encoding = _ENCODING
    num_tokens = 0
    for message in messages:
        num_tokens += TOKENS_PER_MESSAGE
        for value in message.values():
            num_tokens += len(encoding.encode(value))
    num_tokens += TOKENS_PER_MESSAGE
    return num_tokens


def generate(messages, model, max_tokens=None) -> GenerationResult:
    # 1. DECIDE
    prompt_tokens     = count_prompt_tokens(messages, model)
    output_tokens     = max_tokens or DEFAULT_MAX_TOKENS
    prefill_ms        = PREFILL_MS_PER_TOKEN * prompt_tokens
    decode_ms         = DECODE_MS_PER_TOKEN * output_tokens
    total_ms          = prefill_ms + decode_ms

    # 2. DO (consume the time — blocking sleep ON PURPOSE for now)
    time.sleep(total_ms / 1000.0)

    # 3. BUILD placeholder output (~output_tokens worth of words)
    text = " token" * output_tokens

    return GenerationResult(text, prompt_tokens, output_tokens,
                            prefill_ms, decode_ms, total_ms)