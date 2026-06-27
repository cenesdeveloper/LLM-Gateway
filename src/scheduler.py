# src/scheduler.py
import asyncio, time
from dataclasses import dataclass
from .simulator import generate, GenerationResult, prefix_key, count_prompt_tokens
from .metrics import request_count, queue_depth_gauge, batch_size_histogram, prefill_time_histogram, decode_time_histogram, tokens_per_second_counter

PREFILL_MS_PER_TOKEN = 0.5
DECODE_MS_PER_TOKEN  = 30
DEFAULT_MAX_TOKENS   = 64
TOKENS_PER_MESSAGE   = 3
MAX_BATCH_SIZE       = 8     # most jobs the worker processes together at once
NUM_REPLICAS         = 3



@dataclass
class Job:
    messages: list[dict]
    model: str
    max_tokens: int | None
    future: asyncio.Future   # the worker fulfills this
    prefix: str


class Replica:
    """One simulated GPU: its own queue + batching worker."""

    def __init__(self, id):
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self.id = id
        self.cache = set()

    async def submit(self, messages, model, max_tokens, prefix) -> GenerationResult:
        # 1. make a Future tied to the running event loop
        future = asyncio.get_running_loop().create_future()
        # 2. enqueue the job
        await self.queue.put(Job(messages, model, max_tokens, future, prefix))
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
            batch_size_histogram.observe(len(batch))

            # 4. one sleep for the whole batch:
            #    prefill scales with the SUM of prompts, decode with the MAX output length
            results = [r for _, r in done]
            prefill_ms = 0

            for job, result in done:
                if job.prefix not in self.cache:
                    prefill_ms += PREFILL_MS_PER_TOKEN * result.prompt_tokens
                    self.cache.add(job.prefix)
            prefill_time_histogram.observe(prefill_ms)

            decode_ms = DECODE_MS_PER_TOKEN * max(r.completion_tokens for r in results)
            decode_time_histogram.observe(decode_ms)
            print(f"[replica {self.id}] handling batch of {len(done)}")
            await asyncio.sleep((prefill_ms + decode_ms) / 1000.0)
            queue_depth_gauge.labels(replica_id=self.id).set(self.queue.qsize())
            total_tokens = sum(r.completion_tokens for r in results)
            tokens_per_second_counter.labels(model=batch[0].model).inc(total_tokens)

            # 5. hand each job back its own result
            for job, result in done:
                job.future.set_result(result)
                self.queue.task_done()

    def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()

class Router:
    def __init__(self):
        self.replicas = [Replica(i) for i in range(NUM_REPLICAS)]
        self._next = 0   # round-robin pointer
        self.table = {}

    def start(self):
        for replica in self.replicas:
            replica.start()

    async def stop(self):
        for replica in self.replicas:
            await replica.stop()

    async def submit(self, messages, model, max_tokens):
        # pick the next replica in rotation, then advance the pointer
    
        prefix = prefix_key(messages)
        request_count.labels(model=model).inc()
        if prefix in self.table:
            replica = self.table[prefix]
        else:
            replica = self.replicas[self._next]
            self._next = (self._next + 1) % NUM_REPLICAS
            self.table[prefix] = replica
        return await replica.submit(messages, model, max_tokens, prefix)
    