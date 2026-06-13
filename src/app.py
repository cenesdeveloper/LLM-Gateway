import time, json
from fastapi import FastAPI
from .schemas import Request
from .simulator import generate, generate_stream
from fastapi.responses import StreamingResponse


app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ok"}

async def _sse_stream(messages_as_dicts, request):
    created = int(time.time())
    async for token in generate_stream(messages_as_dicts, request.model, request.max_tokens):
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    messages_as_dicts = [m.model_dump() for m in request.messages]

    if request.stream:
        return StreamingResponse(_sse_stream(messages_as_dicts, request),
                                 media_type="text/event-stream")

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
