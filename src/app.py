from fastapi import FastAPI

from .schemas import Request
from .simulator import generate


app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def create_chat_completion(request: Request):
    messages_as_dicts = [m.model_dump() for m in request.messages]
    result = generate(messages_as_dicts, request.model, request.max_tokens)

    return {
        "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
        "object": "chat.completion",
        "created": 1741569952,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }
