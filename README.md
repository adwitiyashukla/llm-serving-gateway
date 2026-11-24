# llm-serving-gateway

An OpenAI compatible gateway that sits in front of local language models. It batches
requests that are in flight at the same time, reuses the KV cache of shared prompt
prefixes, serves repeated questions from a cache, and picks a backend per request.

Everything runs on CPU. I built it and measured it on my laptop.

## What it does

- Continuous batching. A request joins the decode batch as soon as there is a free slot
  and leaves it when it finishes, so a request that arrives late does not wait behind
  the whole batch.
- Prefix KV cache. Prompts are split into fixed size blocks and the KV for each block is
  stored under a hash chained with its parent block. A request that starts with the same
  system prompt as an earlier one skips prefilling that part.
- Response cache. Exact repeats come back from a hash lookup. Near repeats are matched by
  cosine similarity over the query embedding, above a threshold.
- Router. Each request picks a backend from prompt length, keywords and requested output
  length, and falls back to another backend if one fails.
- Admission control. Queue depth cap, per key request and token budgets, request timeouts,
  and 429 on backpressure.
- Metrics and a load generator, which is where the numbers below come from.

## API

`POST /v1/chat/completions` takes the OpenAI request shape and streams over SSE if you
ask it to. There is also `GET /v1/models`, `GET /metrics` in Prometheus text format,
`GET /stats` and `GET /healthz`.

## Results

Measured on an Intel Core 5 120U, 10 cores and 12 threads at 1.4 GHz base, 8 GB RAM,
Windows 11, no GPU. Backend is Qwen2.5-0.5B-Instruct in float32 with torch limited to
6 threads, which was the fastest setting I measured on this chip.

The load generator sends 16 requests at once, 24 output tokens each, over a shared
95 token system prompt. Each row turns one feature off. Every row is the median of two
runs and spread is the gap between the two runs of output tokens per second.

| variant | req/s | out tok/s | TTFT p50 | TTFT p95 | e2e p50 | mean batch | spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all features | 0.35 | 8.4 | 23.7 s | 31.1 s | 42.1 s | 6.94 | 5.2% |
| no continuous batching | 0.13 | 3.1 | 63.6 s | 109.6 s | 71.0 s | 1.00 | 0.3% |
| no prefix cache | 0.26 | 6.1 | 33.5 s | 47.7 s | 59.4 s | 6.94 | 6.0% |
| no response cache | 0.63 | 15.1 | 13.4 s | 18.6 s | 24.2 s | 6.94 | 32.1% |
| nothing enabled | 0.15 | 3.6 | 61.6 s | 93.6 s | 64.3 s | 1.00 | 21.7% |

What that says:

- Continuous batching is worth 2.73x the output tokens per second and cuts time to first
  token to 0.37x. The two runs of each of those rows differed by under 6 percent, so the
  gap is far bigger than the noise.
- The prefix cache is worth 1.37x throughput and drops time to first token to 0.71x,
  because every request in this workload shares the same system prompt.
- With nothing enabled the gateway does 3.6 output tokens per second. Plain transformers
  decoding the same model on the same machine does 3.65. So the gateway adds nothing
  measurable on top of the model, and the gains above come from the batching and the
  caches rather than from anything I did to the model call itself.
- The response cache costs throughput in this test. Every request here is unique and
  arrives at the same moment, so no lookup can hit, and each one still pays to embed its
  query. The next table is where it earns that back.

### Does batching change the answers

It should not, and it does not. The same 20 requests were run twice, once with the batch
size forced to 1 and once batched normally, both with the caches off. Every answer came
back identical, 20 out of 20. Reusing a cached prefix instead of prefilling it leaves the
KV tensors the same to within 3e-08, which is float32 rounding.

### What the response cache buys, and what it costs

Same 20 requests arriving at 0.25 per second, 30 percent of them repeats or near repeats
of an earlier one. Agreement is how often a cached answer matched what the same question
returned with the cache off.

| threshold | served from cache | agreement | median end to end |
| --- | --- | --- | --- |
| 0.90 | 11 of 20 | 64% | 0.09 s |
| 0.95 | 9 of 20 | 78% | 9.4 s |
| 0.99 | 5 of 20 | 100% | 13.1 s |

A hit returns in about 90 ms instead of 13 seconds, so the cache is worth a lot once
traffic repeats. Lower the threshold and it answers more questions from cache, but it
starts handing back an answer that belongs to a different question. 0.99 serves a quarter
of this traffic and never got one wrong, so that is the default in `config.toml`.

## How the batching works

Each sequence keeps its own KV cache. On every decode step the scheduler left pads the
live sequences to the longest one, stacks them into a single batch, runs one forward
pass, and splits the result back out. Left padding with a matching attention mask is what
makes a short sequence sit in the same batch as a long one without changing its output.
Sequences that finish free their slot straight away and waiting requests are prefilled
into it, which is the continuous part.

The prefix cache hashes each 16 token block together with the hash of the block before
it. A lookup walks that chain while the blocks match, so two prompts that share a system
prompt share its KV without the cache having to compare the prompts themselves. Entries
are evicted least recently used, and because a lookup touches the deepest block last, the
deep blocks go first and shared roots stay.

## Running it

```
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
```

The backends in `config.toml` are Hugging Face model ids, so the first start downloads
them. On Windows set `HF_HUB_DISABLE_SYMLINKS=1` first or the download fails on a symlink
privilege error.

Start the gateway:

```
.venv\Scripts\python -m llmserve --config config.toml serve
```

Ask it something:

```
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"what is a KV cache\"}],\"max_tokens\":64}"
```

Reproduce the tables above:

```
.venv\Scripts\python -m llmserve --config config.toml bench --requests 16 --rate 0 --max-tokens 24 --request-timeout 900 --ablation --repeats 2
.venv\Scripts\python -m llmserve --config config.toml bench --requests 20 --rate 0.25 --max-tokens 24 --request-timeout 900 --verify-cache --thresholds 0.90,0.95,0.99
```

There is a second config, `config.synthetic.toml`, that swaps the real model for a tiny
randomly initialized one built in memory. It needs no weights and no network, so the
tests and CI exercise the scheduler, the caches and the API without downloading anything.

## Config

Every tunable lives in `config.toml` and is validated on load, so a bad batch size or a
router rule naming a backend that does not exist fails at startup with a message saying
which key is wrong. `memory.budget_gb` caps how much model weight is resident at once,
and backends over that budget are loaded on demand and evicted least recently used. With
the 4 GB budget in this config, a request that routes to the 1.5B backend evicts the 0.5B
to make room, and that load took 35 seconds on this machine.

## Tests

```
.venv\Scripts\python -m pytest -q
```

122 tests covering config validation, the KV and prefix caches, the batching engine, the
response caches, the router, admission control, the API contract and the load generator.
