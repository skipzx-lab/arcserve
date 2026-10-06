# arcserve

A small OpenAI-compatible server for [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) on Intel Arc
GPUs, with continuous batching and tool calling. Built and tested with **Gemma 4 26B-A4B** on a single
**Arc Pro B60 (24 GB)**.

Status: experimental. One machine (Arc Pro B60, Ubuntu 26.04, xe driver), OpenVINO / GenAI 2026.4.1,
[`OpenVINO/gemma-4-26b-a4b-it-int4-ov`](https://huggingface.co/OpenVINO/gemma-4-26b-a4b-it-int4-ov).

## Findings first

- **arcserve is the fastest way we found to serve Gemma 4 26B-A4B on one B60**: ~60 tok/s single stream and
  ~117 tok/s across 3 streams, with working tool calls.
- **Gemma 4 26B-A4B can do agentic coding, but it is sensitive to KV-cache precision at long context.** In our
  OpenCode harness (the same harness on which a Qwen3.6-35B-A3B build passes the ticket first try), on llama.cpp with
  Google's QAT Q4_0 GGUF:
  - with a **q8_0 KV cache** it failed: at 20-40k tokens of context it quoted code for `edit` calls without the blank
    lines between methods, so 41 of 42 edits were rejected and it never recovered;
  - with an **f16 KV cache** the same ticket reached 23/26 acceptance tests on the first attempt;
  - at short context it edits correctly on both llama.cpp and arcserve (6/6 in a micro-test).
  So check the KV cache before blaming the model. OpenVINO's GPU default is `KV_CACHE_PRECISION=dynamic` (compressed);
  we haven't yet tested arcserve with an f16 KV cache.
- **But f16 KV on llama.cpp SYCL is slow at agent context lengths**: ~40 tok/s on a short prompt, ~9 tok/s per
  session with three sessions at 26-43k tokens. Not practical for a coding agent on one B60.
- **llama.cpp's host-RAM caches can exhaust system RAM** with SWA models like Gemma: `--ctx-checkpoints` defaults to 32
  per slot and `--cache-ram` to 8 GiB. With 3 slots of 64k and an f16 KV cache, llama-server grew from 1.6 to 16 GB of
  RSS in two minutes on our 32 GB host. `--ctx-checkpoints 4 --cache-ram 2048` kept it at ~4.5 GB.
- **For Gemma 4 chat on Arc, llama.cpp (SYCL) with Google's QAT Q4_0 GGUF works correctly out of the box** (tool calls,
  long-context recall, prompt caching): ~40 tok/s single stream.
- **Long context in fp16 is fragile** with the int4 OpenVINO export: at ~28k tokens the output depends on the prefill
  chunk size. Details and a repro in
  [openvino.genai#4146](https://github.com/openvinotoolkit/openvino.genai/issues/4146#issuecomment-6009326327).
- **Agent clients matter.** Under Qwen Code, Gemma copied the plain-text tool-call examples in Qwen Code's system prompt
  instead of using its native tool tokens; under OpenCode it uses native calls. OpenCode's `read` output prefixes
  lines with numbers (`3: `), which Gemma sometimes pastes into `oldString` (it corrects itself on retry).

## Why a custom server

On this card, the ready-made options each broke somewhere:

| Option | What happened |
|---|---|
| vLLM (Intel `llm-scaler-vllm` 0.26) with GPTQ / compressed-tensors Gemma 4 | ~20 tok/s (no fast MoE kernel for Gemma on XPU); XPU graph capture fails; the image ships transformers 5.8, and Gemma 4 needs >= 5.10.4 ([vllm#45259](https://github.com/vllm-project/vllm/issues/45259)) |
| OpenVINO Model Server 2026.4.0 | segfaults initialising Gemma 4 (VLM continuous batching) or on the first request |
| OpenArc | works, but serves one request at a time |
| OpenVINO GenAI directly | fast and correct, but a library, not a server |

## What arcserve does

- Runs `openvino_genai.ContinuousBatchingPipeline` in **its own process**. `step()` holds the Python GIL while the GPU
  works; in a thread of the web server it starved the HTTP threads (requests admitted one at a time, tokens arriving
  in a burst at the end). Requests and token ids cross over `multiprocessing` queues.
- Renders prompts with Hugging Face `transformers` (tools, thinking on/off) and passes text to the pipeline (VLM
  exports run the language model on `inputs_embeds`). **Keeps `<bos>`:** the OpenVINO tokenizer encodes it but never
  adds it, and Gemma degenerates without it.
- **Stops on the model's own stop tokens** (`generation_config.json`: for Gemma 4 that includes `<|tool_response>`,
  where the model hands over to the tool; without it the model invents its own tool results).
- **Gemma 4 tool calls** (`toolparse.py`): native `<|tool_call>call:name{...}<tool_call|>` with strings delimited by
  Gemma's quote token (commas, braces and quotes inside file contents are safe), plus a fallback for the plain-text
  form Gemma produces when it copies a client's prompt examples (`[tool_call: read_file {file_path: '...'}]`).
- Streaming that holds back tool-call text, sends a keep-alive chunk every 3 s (clients such as Qwen Code abort a
  stream after 240 s without data; long prefills take that long), and cancels a reply that keeps repeating the same
  tool call.
- `/metrics` with vLLM-style names; optional request/response logging for debugging (`ARC_LOG_DIR`).

## Two ways to run Gemma 4

| | Official export | RoPE-table patched export |
|---|---|---|
| Model | `OpenVINO/gemma-4-26b-a4b-it-int4-ov` as published | same, after `ov_rope_lut.py` (below) |
| `ARC_PREFIX_CACHE` | `1` (works) | **`0`** (the patch breaks prefix caching: wrong answers) |
| `ARC_BATCH_TOKENS` | up to `4096` | `512` or less |
| Speed in an agent loop | fast (cached prefixes, ~2,000 tok/s prefill) | slow (re-reads the whole context each turn, ~1,500 tok/s) |
| Long-context accuracy | fragile past ~20k tokens (#4146) | better: correct where the official export was not |

The patch: [DassaultFalconKing/OpenVino-For-Gemma-4](https://github.com/DassaultFalconKing/OpenVino-For-Gemma-4)
`patches/ov_rope_lut.py` replaces the runtime fp16 RoPE angle computation with precomputed sin/cos tables. For the
official exports it needs a 3-line change (proposed upstream in
[PR #3](https://github.com/DassaultFalconKing/OpenVino-For-Gemma-4/pull/3)); run it with `LUT_MAXPOS=65536`.
arcserve's defaults (`ARC_PREFIX_CACHE=0`, `ARC_BATCH_TOKENS=512`) are safe for both.

## Measured (Arc Pro B60, Gemma 4 26B-A4B int4, official export, prefix cache on, chunk 4096)

| Concurrent streams | Combined decode | Time to first token |
|---|---|---|
| 1 | ~60 tok/s | ~0.3 s |
| 3 | ~117 tok/s | ~0.4 s |

14k-token needle-in-a-haystack: correct, ~1,100 tok/s prefill. Measured with [`tools/llmbench.py`](tools/llmbench.py)
(tool call, concurrency, recall), which works against any OpenAI-compatible server.

## Run

```bash
docker build -t arcserve .
docker run -d --name arcserve --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
  -v /dev/dri/by-path:/dev/dri/by-path -v /path/to/models:/models:ro -p 127.0.0.1:8080:8080 \
  -e ARC_MODEL=/models/gemma-4-26b-a4b-it-int4-ov -e ARC_MAX_SEQS=3 \
  -e ARC_PREFIX_CACHE=1 -e ARC_BATCH_TOKENS=4096 arcserve
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"llm","messages":[{"role":"user","content":"Say hi"}],"max_tokens":20}'
```

The base image (`intel/llm-scaler-vllm`) is used only for its Intel GPU runtime, which works on the B60.

| Env var | Default | Meaning |
|---|---|---|
| `ARC_MODEL` | (required) | OpenVINO model directory |
| `ARC_NAME` | `llm` | model name served |
| `ARC_PORT` | `8080` | |
| `ARC_DEVICE` | `GPU` | |
| `ARC_MAX_SEQS` | `4` | concurrent sequences |
| `ARC_CACHE_GB` | `5` | KV cache size |
| `ARC_BATCH_TOKENS` | `512` | prefill chunk (tokens per scheduler step) |
| `ARC_PREFIX_CACHE` | `0` | `1` enables prefix caching (official export only) |
| `ARC_MAX_NEW` | `8192` | cap on tokens per reply |
| `ARC_MAX_CTX` | `65536` | prompt + reply limit |
| `ARC_THINK` | `0` | `1` enables Gemma's thinking |
| `ARC_CLOSE_THOUGHT_AFTER_TOOL` | `0` | `1` pre-closes an empty thought channel after tool results (experimental) |
| `ARC_PARSER` | `gemma4` | `none` disables tool-call parsing |
| `ARC_LOG_DIR` | unset | if set, saves each request and raw output there |

## Tests

```bash
python -m unittest discover tests   # parser tests; no model or GPU needed
```

## Credits

- [DassaultFalconKing/OpenVino-For-Gemma-4](https://github.com/DassaultFalconKing/OpenVino-For-Gemma-4): the RoPE
  table patch and the `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` setting used here.
- [OpenArc](https://github.com/SearchSavior/OpenArc): documented Gemma 4's tool-call protocol.
- [vpscloud's write-up](https://vpscloud.com.au/technical/two-flags-and-a-fortnight-getting-gemma-4-31b-to-fly-on-intel-arc-pro-b60s/)
  of Gemma 4 on multi-B60 vLLM.
