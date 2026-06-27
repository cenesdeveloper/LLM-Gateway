import time, json
from .metrics import request_latency_histogram, ttft_histogram, itl_histogram
from fastapi import FastAPI, Request, Response
from .schemas import ChatCompRequest
from .simulator import generate, generate_stream
from fastapi.responses import StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# src/app.py  (sketch of the new bits)
from contextlib import asynccontextmanager
from .scheduler import Router

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.router = Router()
    app.state.router.start()
    yield
    await app.state.router.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "ok"}

async def _sse_stream(messages_as_dicts, request):
    created = int(time.time())
    start = time.time()
    first = True
    prev = start

    async for token in generate_stream(messages_as_dicts, request.model, request.max_tokens):
        now = time.time()
        if first:
            ttft_histogram.observe(now - start)
            first = False
        else:
            itl_histogram.observe((now - prev) * 1000)
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        prev = now
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompRequest, http_req: Request):
    messages_as_dicts = [m.model_dump() for m in request.messages]

    if request.stream:
        return StreamingResponse(_sse_stream(messages_as_dicts, request),
                                 media_type="text/event-stream")

    start = time.time()
    result = await http_req.app.state.router.submit(messages_as_dicts, request.model, request.max_tokens)
    request_latency_histogram.observe(time.time() - start)
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

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)