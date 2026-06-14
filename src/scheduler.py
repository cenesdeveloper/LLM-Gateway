# src/scheduler.py
import asyncio
from dataclasses import dataclass
from .simulator import generate, GenerationResult



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
        # infinite loop: pull one job, process it, fulfill its future
        while True:
            job = await self.queue.get()
            try:
                # run the blocking event in another thread to stay responsive
                result = await asyncio.to_thread(
                    generate, job.messages, job.model, job.max_tokens
                )
                job.future.set_result(result)
            except Exception as e:
                job.future.set_exception(e)
            finally:
                self.queue.task_done()

    def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()