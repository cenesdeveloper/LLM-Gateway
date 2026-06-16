# src/scheduler.py
import asyncio, time
from dataclasses import dataclass
from .simulator import generate, GenerationResult, count_prompt_tokens

PREFILL_MS_PER_TOKEN = 0.5
DECODE_MS_PER_TOKEN  = 30
DEFAULT_MAX_TOKENS   = 64
TOKENS_PER_MESSAGE   = 3
MAX_BATCH_SIZE       = 8     # most jobs the worker processes together at once



@dataclass
class Job:
    messages: list[dict]
    model: str
    max_tokens: int | None
    future: asyncio.Future   # the worker fulfills this


class Scheduler:
    def __init__(self):
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def submit(self, messages, model, max_tokens) -> GenerationResult:
        # 1. make a Future tied to the running event loop
        future = asyncio.get_running_loop().create_future()
        # 2. enqueue the job
        await self.queue.put(Job(messages, model, max_tokens, future))
        # 3. wait until the worker fulfills it, return the result
        return await future

    async def _worker(self):
        while True:
            # 1. block until at least one job is waiting
            batch = [await self.queue.get()]

            # 2. drain any other jobs already waiting, up to MAX_BATCH_SIZE
            while len(batch) < MAX_BATCH_SIZE:
                try:
                    batch.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # 3. compute each job's result (no sleeping); a bad job fails on its own
            done = []  # (job, result) pairs that succeeded
            for job in batch:
                try:
                    result = generate(job.messages, job.model, job.max_tokens)
                    done.append((job, result))
                except Exception as e:
                    job.future.set_exception(e)
                    self.queue.task_done()

            if not done:
                continue

            # 4. one sleep for the whole batch:
            #    prefill scales with the SUM of prompts, decode with the MAX output length
            results = [r for _, r in done]
            prefill_ms = PREFILL_MS_PER_TOKEN * sum(r.prompt_tokens for r in results)
            decode_ms = DECODE_MS_PER_TOKEN * max(r.completion_tokens for r in results)
            await asyncio.sleep((prefill_ms + decode_ms) / 1000.0)

            # 5. hand each job back its own result
            for job, result in done:
                job.future.set_result(result)
                self.queue.task_done()

    def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()