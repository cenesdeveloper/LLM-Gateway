from prometheus_client import Histogram, Counter, Gauge

request_count = Counter('llm_requests_total', 'Total requests', ['model'])
queue_depth_gauge = Gauge('llm_queue_depth', 'Requests waiting', ['replica_id'])
batch_size_histogram = Histogram('llm_batch_size', 'Jobs per batch', buckets=[1,2,4,8])
prefill_time_histogram = Histogram('llm_prefill_ms', 'Prefill time in ms', buckets=[25, 50, 100, 500, 1000, 2000])
decode_time_histogram = Histogram('llm_decode_ms', 'Decode time in ms', buckets=[25, 50, 100, 500, 1000, 2000])
tokens_per_second_counter = Counter('llm_tokens_generated_total', 'Total tokens', ['model'])
request_latency_histogram = Histogram('llm_request_latency_seconds', 'End-to-end latency', buckets=[0.1, 0.5, 1.0, 2.0, 5.0])
ttft_histogram = Histogram('llm_ttft_seconds', 'Time to first token', buckets=[0.01, 0.05, 0.1, 0.5])
itl_histogram = Histogram('llm_itl_ms', 'Inter-token latency', buckets=[10, 25, 50, 100, 200])