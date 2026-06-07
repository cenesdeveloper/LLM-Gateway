from fastapi import FastAPI

from .schemas import Message, Request

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def create_chat_completion(request: Request):
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
                    "content": request.messages[-1].content,
                    "refusal": None,
                    "annotations": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 10,
            "total_tokens": 29,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": 0,
            },
        },
    }
