import asyncio, time, argparse
import httpx, random
import statistics

BASE_URL = "http://localhost:8000"
PAYLOAD = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hello world"}],
    "stream": False
}



async def send_one(client: httpx.AsyncClient, latencies) -> float:
    start = time.time()
    await client.post(f"{BASE_URL}/v1/chat/completions", json=PAYLOAD)
    latencies.append(time.time() - start)
    return time.time() - start

async def load_test(rate: float, duration: float):
    latencies = []
    tasks = []
    start = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        while time.time() - start < duration:
            # spawn request without waiting for it
            task = asyncio.create_task(send_one(client, latencies))
            tasks.append(task)
            # Poisson inter-arrival time
            await asyncio.sleep(random.expovariate(rate))

        await asyncio.gather(*tasks)
    return latencies

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    latencies = await load_test(args.rate, args.duration)
    latencies.sort()
    p50 = statistics.median(latencies)
    p99 = latencies[int(len(latencies) * 0.99)]
    throughput = len(latencies) / args.duration
    print(f"p50: {p50:.3f}s  p99: {p99:.3f}s  throughput: {throughput:.1f} req/s")
    print(f"completed: {len(latencies)} requests")

asyncio.run(main())