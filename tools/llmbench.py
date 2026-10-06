#!/usr/bin/env python3
"""llmbench: quick correctness + speed check for any OpenAI-compatible server (vLLM, llama.cpp, arcserve, ...).

  llmbench.py [--url http://127.0.0.1:8006] [--model llm] [--streams 1,2,3,4] [--no-recall]

Checks: a tool call (get_weather for Point Cook), decode speed at each concurrency (combined tok/s and time to first
token), and a ~14k-token needle-in-a-haystack recall. Token counts come from the server's usage when it reports it
(stream_options.include_usage), otherwise from the number of content chunks."""
import argparse, json, random, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8006")
ap.add_argument("--model", default="llm")
ap.add_argument("--streams", default="1,3")
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--no-recall", action="store_true")
a = ap.parse_args()
U = a.url.rstrip("/") + "/v1/chat/completions"


def post(body, timeout=900):
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream(msgs, n, extra=None):
    body = {"model": a.model, "messages": msgs, "max_tokens": n, "stream": True,
            "stream_options": {"include_usage": True}, **(extra or {})}
    t0 = time.time(); first = last = None; chunks = 0; usage = None; text = ""
    for line in post(body):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        ch = d.get("choices") or []
        c = ch[0].get("delta", {}).get("content") if ch else None
        if c:
            chunks += 1; text += c; now = time.time(); first = first or now; last = now
    toks = (usage or {}).get("completion_tokens") or chunks
    return {"t0": t0, "first": first, "last": last, "toks": toks, "text": text}


def tool_check():
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a place",
              "parameters": {"type": "object", "properties": {"place": {"type": "string"}}, "required": ["place"]}}}]
    r = json.load(post({"model": a.model, "stream": False, "max_tokens": 300, "tools": tools,
                        "messages": [{"role": "user", "content": "What is the weather in Point Cook right now?"}]}))
    m = r["choices"][0]["message"]; calls = m.get("tool_calls") or []
    ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "Point Cook" in calls[0]["function"]["arguments"]
    print(f"tool call: {'PASS' if ok else 'FAIL'}  {json.dumps([c['function'] for c in calls])[:160]} "
          f"finish={r['choices'][0].get('finish_reason')} content={str(m.get('content'))[:60]!r}")


def concurrency(n):
    res = []
    def go(k):
        res.append(stream([{"role": "user", "content": f"Write a {a.tokens}-word story about robot number {k}."}], a.tokens))
    th = [threading.Thread(target=go, args=(k,)) for k in range(n)]
    T = time.time(); [t.start() for t in th]; [t.join() for t in th]
    ok = [r for r in res if r["first"]]
    if not ok:
        print(f"{n} streams: no output"); return
    span = max(r["last"] for r in ok) - min(r["first"] for r in ok)
    tot = sum(r["toks"] for r in ok)
    print(f"{n} stream(s): combined {tot / span:6.1f} tok/s | per stream {tot / span / n:5.1f} | "
          f"TTFT {[round(r['first'] - T, 2) for r in ok]} s")


def recall():
    random.seed(1)
    facts = [f"Item {i}: the code word for shelf {i} is {random.choice(['apple','river','copper','falcon','maple','orbit'])}-{random.randint(100,999)}." for i in range(1100)]
    want = facts[777].split()[-1].rstrip(".")
    r = stream([{"role": "user", "content": "\n".join(facts) + "\n\nWhat is the code word for shelf 777? Answer with just the code word."}], 30)
    got = r["text"].strip()
    print(f"recall ~14k tok: {'PASS' if want in got else 'FAIL'}  answer={got[:40]!r} expected={want!r} "
          f"TTFT {r['first'] - r['t0']:.1f}s (~{14100 / (r['first'] - r['t0']):.0f} tok/s prefill)")


tool_check()
for n in [int(x) for x in a.streams.split(",")]:
    concurrency(n)
if not a.no_recall:
    recall()
