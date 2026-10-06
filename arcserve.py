#!/usr/bin/env python3
"""arcserve: a small OpenAI-compatible server for OpenVINO GenAI on Intel Arc (B60), with continuous batching.

Why it exists (2026-10-06): OVMS 2026.4.0 crashes on Gemma 4 on our B60 and OpenArc serves one request at a time.
This uses openvino_genai.ContinuousBatchingPipeline directly, with the workaround from
github.com/DassaultFalconKing/OpenVino-For-Gemma-4 (DYNAMIC_QUANTIZATION_GROUP_SIZE=0), HF chat templating (tools,
thinking off), and its own Gemma 4 tool-call parser (Gemma's quote tokens delimit strings, so commas inside file
contents are safe).

Env: ARC_MODEL (OpenVINO model dir), ARC_NAME (served model name, default "llm"), ARC_PORT (8080), ARC_DEVICE (GPU),
ARC_MAX_SEQS (4), ARC_CACHE_GB (5), ARC_BATCH_TOKENS (4096), ARC_MAX_NEW (8192: cap per reply), ARC_MAX_CTX (65536),
ARC_THINK (0), ARC_PARSER (gemma4|none).
Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (stream or not), GET /metrics (vllm:* names).
"""
import json, os, queue, re, threading, time, uuid

import numpy as np
import openvino as ov
import openvino_genai as og
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from transformers import AutoTokenizer

MODEL = os.environ["ARC_MODEL"]
NAME = os.environ.get("ARC_NAME", "llm")
PORT = int(os.environ.get("ARC_PORT", "8080"))
DEVICE = os.environ.get("ARC_DEVICE", "GPU")
MAX_SEQS = int(os.environ.get("ARC_MAX_SEQS", "4"))
CACHE_GB = int(os.environ.get("ARC_CACHE_GB", "5"))
BATCH_TOKENS = int(os.environ.get("ARC_BATCH_TOKENS", "4096"))
MAX_NEW = int(os.environ.get("ARC_MAX_NEW", "8192"))
MAX_CTX = int(os.environ.get("ARC_MAX_CTX", "65536"))
THINK = os.environ.get("ARC_THINK", "0") == "1"
PARSER = os.environ.get("ARC_PARSER", "gemma4")

import toolparse
from toolparse import CH_OPEN, QUOTE, STRIP, TEXT_CALL, TOOL_OPEN, parse_output, runaway, safe_prefix

toolparse.PARSER = PARSER

tok = AutoTokenizer.from_pretrained(MODEL)
STOP_IDS = {i for i in (tok.convert_tokens_to_ids(t) for t in ("<turn|>", "<eos>")) if isinstance(i, int) and i >= 0}

# ---------------------------------------------------------------- engine process
# openvino_genai's ContinuousBatchingPipeline.step() holds the GIL while the GPU works. Run back-to-back in a thread of
# the web server, it starved the HTTP threads: requests were admitted one at a time and tokens reached clients in a
# burst at the end. So the pipeline lives in its own process; requests and token ids cross over multiprocessing queues.
import multiprocessing as mp


def engine_main(inq, outq):
    sc = og.SchedulerConfig()
    sc.cache_size = CACHE_GB
    sc.max_num_seqs = MAX_SEQS
    sc.max_num_batched_tokens = BATCH_TOKENS
    sc.enable_prefix_caching = True
    sc.dynamic_split_fuse = True
    t0 = time.time()
    pipe = og.ContinuousBatchingPipeline(MODEL, sc, DEVICE, {"DYNAMIC_QUANTIZATION_GROUP_SIZE": 0})
    print(f"[arcserve] loaded {MODEL} on {DEVICE} in {time.time() - t0:.0f}s: max_seqs={MAX_SEQS} cache={CACHE_GB}GB "
          f"max_new={MAX_NEW} max_ctx={MAX_CTX} think={THINK} parser={PARSER}", flush=True)
    outq.put(("ready", None, None))
    handles = {}
    while True:
        while True:  # admit new requests (block only when idle)
            try:
                rid, prompt, params = inq.get(timeout=0.05) if not handles else inq.get_nowait()
            except queue.Empty:
                break
            if prompt is None:  # cancel request
                h = handles.pop(rid, None)
                if h is not None:
                    try:
                        h.drop()
                    except Exception:  # noqa: BLE001
                        pass
                    outq.put(("done", rid, "cancelled"))
                continue
            c = og.GenerationConfig()
            for k, v in params.items():
                setattr(c, k, v)
            handles[rid] = pipe.add_request(rid, prompt, [], c)
        if not handles:
            continue
        try:
            pipe.step()
        except Exception as e:  # noqa: BLE001
            print("[arcserve] step failed:", repr(e), flush=True)
            for rid in handles:
                outq.put(("error", rid, str(e)))
            handles.clear(); continue
        for rid, h in list(handles.items()):
            while h.can_read():
                for out in h.read().values():
                    outq.put(("ids", rid, list(out.generated_ids)))
            st = h.get_status()
            if st != og.GenerationStatus.RUNNING:
                outq.put(("done", rid, str(st)))
                del handles[rid]


MP = mp.get_context("spawn")
inq, outq = MP.Queue(), MP.Queue()
waiters = {}           # rid -> queue.Queue in this (web) process
stats = {"gen": 0, "prompt": 0}
next_id = [0]
id_lock = threading.Lock()


def router():
    while True:
        kind, rid, v = outq.get()
        if kind == "ready":
            ready.set(); continue
        if kind == "ids":
            stats["gen"] += len(v)
        q = waiters.get(rid)
        if q is not None:
            q.put((kind, v))
        if kind in ("done", "error"):
            waiters.pop(rid, None)


ready = threading.Event()


def cancel(rid):
    inq.put((rid, None, None))


def submit(prompt_text, cfg):
    """VLM exports run the language model on inputs_embeds, so the pipeline takes the rendered prompt text (it embeds it
    itself); our HF chat template already rendered it, so the pipeline's own templating is off and <bos> is left to its
    tokenizer."""
    if prompt_text.startswith("<bos>"):
        prompt_text = prompt_text[len("<bos>"):]
    q = queue.Queue()
    with id_lock:
        next_id[0] += 1
        rid = next_id[0]
        waiters[rid] = q
    inq.put((rid, prompt_text, cfg))
    q.rid = rid
    return q


# ---------------------------------------------------------------- prompt building
def to_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        fn["arguments"] = {}
                calls.append({**c, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def build_prompt(body):
    kw = {"tools": body.get("tools")} if body.get("tools") else {}
    text = tok.apply_chat_template(to_messages(body["messages"]), add_generation_prompt=True, tokenize=False,
                                   enable_thinking=THINK, **kw)
    return text, tok(text, add_special_tokens=False)["input_ids"]


def gen_config(body):
    c = {"max_new_tokens": min(int(body.get("max_tokens") or body.get("max_completion_tokens") or MAX_NEW), MAX_NEW),
         "stop_token_ids": set(STOP_IDS), "apply_chat_template": False}
    t = body.get("temperature")
    t = 1.0 if t is None else float(t)
    if t > 0:
        c.update(do_sample=True, temperature=t, top_p=float(body.get("top_p") or 0.95), top_k=int(body.get("top_k") or 64))
    else:
        c["do_sample"] = False
    if body.get("seed") is not None:
        c["rng_seed"] = int(body["seed"])
    return c


# ---------------------------------------------------------------- HTTP
app = FastAPI()


@app.get("/health")
def health():
    return PlainTextResponse("ok")


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": NAME, "object": "model", "owned_by": "arcserve"}]}


@app.get("/metrics")
def metrics():
    lines = [f'vllm:generation_tokens_total{{model_name="{NAME}"}} {stats["gen"]}',
             f'vllm:prompt_tokens_total{{model_name="{NAME}"}} {stats["prompt"]}',
             f'vllm:num_requests_running{{model_name="{NAME}"}} {len(waiters)}',
             f'vllm:num_requests_waiting{{model_name="{NAME}"}} 0']
    return PlainTextResponse("\n".join(lines) + "\n")


def err(code, msg):
    return JSONResponse({"error": {"message": msg, "type": "invalid_request_error"}}, status_code=code)


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    try:
        prompt_text, ids = build_prompt(body)
    except Exception as e:  # noqa: BLE001
        return err(400, f"could not apply chat template: {e}")
    cfg = gen_config(body)
    if len(ids) + cfg['max_new_tokens'] > MAX_CTX:
        if len(ids) >= MAX_CTX - 256:
            return err(400, f"This model's maximum context length is {MAX_CTX} tokens. "
                            f"However, your messages resulted in {len(ids)} tokens.")
        cfg['max_new_tokens'] = MAX_CTX - len(ids)
    stats["prompt"] += len(ids)
    q = submit(prompt_text, cfg)
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

    def collect():
        out_ids, status = [], None
        while True:
            kind, v = q.get()
            if kind == "ids":
                out_ids.extend(v); yield out_ids, None
            else:
                yield out_ids, (kind, v); return

    def finish(out_ids, text):
        content, calls = parse_output(text, body.get("tools"))
        if calls:  # drop exact duplicates (a looping reply repeats the same call)
            seen, uniq = set(), []
            for c in calls:
                k = (c["function"]["name"], c["function"]["arguments"])
                if k not in seen:
                    seen.add(k); uniq.append(c)
            calls = uniq
        hit_len = len(out_ids) >= cfg["max_new_tokens"]
        reason = "tool_calls" if calls else ("length" if hit_len else "stop")
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(out_ids), "total_tokens": len(ids) + len(out_ids)}
        return content, calls, reason, usage

    if not body.get("stream"):
        def drain():
            o, e = [], None
            for o, e in collect():
                pass
            return o, e
        out_ids, end = await run_in_threadpool(drain)
        if end and end[0] == "error":
            return err(500, "generation failed: " + end[1])
        content, calls, reason, usage = finish(out_ids, tok.decode(out_ids, skip_special_tokens=False))
        msg = {"role": "assistant", "content": content or (None if calls else "")}
        if calls:
            msg["tool_calls"] = calls
        return {"id": cid, "object": "chat.completion", "created": created, "model": NAME,
                "choices": [{"index": 0, "message": msg, "finish_reason": reason}], "usage": usage}

    def sse():
        def chunk(delta, reason=None, usage=None):
            d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": NAME,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
            if usage:
                d["usage"] = usage
            return "data: " + json.dumps(d) + "\n\n"
        yield chunk({"role": "assistant", "content": ""})
        sent, out_ids, beat, stopped = 0, [], time.time(), False
        for out_ids, end in collect():
            if end and end[0] == "error":
                yield "data: " + json.dumps({"error": {"message": "generation failed: " + end[1]}}) + "\n\n"; return
            text = tok.decode(out_ids, skip_special_tokens=False)
            safe = safe_prefix(text)
            if len(safe) > sent and not end:
                yield chunk({"content": safe[sent:]}); sent = len(safe); beat = time.time()
            elif not end and time.time() - beat > 3:   # keep-alive while tool-call text is held back
                yield chunk({"content": ""}); beat = time.time()
            if not end and not stopped and runaway(text):
                stopped = True; cancel(q.rid)
            if end:
                content, calls, reason, usage = finish(out_ids, text)
                if len(content) > sent:
                    yield chunk({"content": content[sent:]})
                for k, c in enumerate(calls or []):
                    yield chunk({"tool_calls": [{"index": k, **c}]})
                yield chunk({}, reason, usage)
                yield "data: [DONE]\n\n"
                return

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    MP.Process(target=engine_main, args=(inq, outq), daemon=True).start()
    threading.Thread(target=router, daemon=True).start()
    ready.wait()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
