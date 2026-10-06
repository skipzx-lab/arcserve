# arcserve

A small OpenAI-compatible server for [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) on Intel Arc
GPUs, with continuous batching and tool calling. Built to run **Gemma 4 26B-A4B** on a single **Arc Pro B60 (24 GB)**
for agentic coding tools such as Qwen Code.

Status: experimental. Tested on one machine (Arc Pro B60, Ubuntu 26.04, xe driver) with
[`OpenVINO/gemma-4-26b-a4b-it-int4-ov`](https://huggingface.co/OpenVINO/gemma-4-26b-a4b-it-int4-ov).

## Why

On this card, the ready-made options each broke somewhere:

| Option | What happened |
|---|---|
| vLLM (Intel `llm-scaler-vllm` 0.26) with GPTQ / compressed-tensors Gemma 4 | ~20 tok/s: no fast MoE kernel for Gemma on XPU; XPU graph capture fails, so it needs eager mode |
| OpenVINO Model Server 2026.4.0 | segfaults initialising Gemma 4 (VLM continuous batching) or on the first request |
| OpenArc | works, but serves one request at a time; its tool-argument parser splits on commas |
| OpenVINO GenAI directly | **fast (55-65 tok/s) and correct**, but it's a library, not a server |

arcserve is the thin server around the part that works.

## What it does

- `openvino_genai.ContinuousBatchingPipeline` in **its own process**. `step()` holds the Python GIL while the GPU
  works; run in a thread of the web server it starved the HTTP threads (requests were admitted one at a time and tokens
  arrived in a burst at the end). Requests and token ids cross over `multiprocessing` queues.
- `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`, needed for continuous batching with these int4 MoE exports (found by
  [DassaultFalconKing/OpenVino-For-Gemma-4](https://github.com/DassaultFalconKing/OpenVino-For-Gemma-4)).
- Chat templating with Hugging Face `transformers` (tools, thinking off), prompts passed as text (VLM exports run the
  language model on `inputs_embeds`).
- **Gemma 4 tool calls** (`toolparse.py`): the native `<|tool_call>call:name{...}<tool_call|>` format, where strings are
  delimited by Gemma's quote token, so commas, braces and quotes inside file contents are safe; plus a fallback for
  the plain-text form Gemma produces when it copies examples from a client's system prompt
  (`[tool_call: read_file {file_path: '...'}]` and `[tool_call: run_shell_command for '...']`).
- Streaming that holds back tool-call text, sends keep-alive chunks while it does (clients such as Qwen Code abort a
  stream after 240 s without data), and cancels a reply that keeps repeating the same tool call.
- `/metrics` with vLLM-style names (`vllm:generation_tokens_total`, `vllm:num_requests_running`, ...).

## Measured (Arc Pro B60, Gemma 4 26B-A4B int4)

| Concurrent streams | Combined decode | Time to first token |
|---|---|---|
| 1 | ~60 tok/s | ~0.4-1 s |
| 2 | ~80 tok/s | ~0.3 s |
| 3 | ~105 tok/s | ~0.4-1.8 s |
| 4 | ~130 tok/s | ~1.3 s |

A 14k-token needle-in-a-haystack prompt is answered correctly (with the RoPE patch below), prefill ~1,000 tok/s.

## Long context: patch the model first

The official int4 export computes RoPE angles at runtime in fp16 on the GPU, which garbles output at long context.
DassaultFalconKing's `patches/ov_rope_lut.py` replaces that with precomputed sin/cos tables. For the official
`OpenVINO/gemma-4-*-int4-ov` exports it needs one change: in `trace_inv_freq`, also accept a `Constant` of shape
`[1, F, 1]` fed straight into the `MatMul` (their exports go through a `Broadcast`). Run it with `LUT_MAXPOS=65536` for
64k context.

## Run

```bash
docker build -t arcserve .
docker run -d --name arcserve --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
  -v /dev/dri/by-path:/dev/dri/by-path -v /path/to/models:/models:ro -p 127.0.0.1:8080:8080 \
  -e ARC_MODEL=/models/gemma-4-26b-a4b-it-int4-ov-ropelut -e ARC_MAX_SEQS=3 arcserve
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"llm","messages":[{"role":"user","content":"Say hi"}],"max_tokens":20}'
```

| Env var | Default | Meaning |
|---|---|---|
| `ARC_MODEL` | (required) | OpenVINO model directory |
| `ARC_NAME` | `llm` | model name served |
| `ARC_PORT` | `8080` | |
| `ARC_DEVICE` | `GPU` | |
| `ARC_MAX_SEQS` | `4` | concurrent sequences |
| `ARC_CACHE_GB` | `5` | KV cache size |
| `ARC_BATCH_TOKENS` | `4096` | max tokens per scheduler step |
| `ARC_MAX_NEW` | `8192` | cap on tokens per reply |
| `ARC_MAX_CTX` | `65536` | prompt + reply limit |
| `ARC_THINK` | `0` | `1` enables Gemma's thinking |
| `ARC_PARSER` | `gemma4` | `none` disables tool-call parsing |

## Tests

```bash
python -m unittest discover tests   # parser tests; no model or GPU needed
```

## Credits

- [DassaultFalconKing/OpenVino-For-Gemma-4](https://github.com/DassaultFalconKing/OpenVino-For-Gemma-4): the RoPE
  table patch and the `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` finding.
- [OpenArc](https://github.com/SearchSavior/OpenArc): documented Gemma 4's tool-call protocol.
