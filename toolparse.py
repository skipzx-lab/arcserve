"""Output parsing for arcserve: Gemma 4 native tool calls, the text-form fallback, streaming hold-back and a loop guard.
Pure Python (no OpenVINO), so it can be unit-tested anywhere: python -m unittest discover tests"""
import json, re, uuid

PARSER = "gemma4"   # arcserve sets this from ARC_PARSER

TOOL_OPEN, TOOL_CLOSE = "<|tool_call>", "<tool_call|>"
CH_OPEN, CH_CLOSE = "<|channel>", "<channel|>"
QUOTE = '<|"|>'
STRIP = ("<turn|>", "<eos>", "<|turn>", "<bos>")

class _Args:
    """Gemma 4 tool-call arguments: {key:value,...}; strings are wrapped in <|"|> quote tokens; numbers, true/false/null,
    nested {...} and [...] are literal."""

    def __init__(self, s):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \n\t\r":
            self.i += 1

    def value(self):
        self.ws()
        s = self.s
        if s.startswith(QUOTE, self.i):
            j = s.find(QUOTE, self.i + len(QUOTE))
            if j == -1:
                raise ValueError("unterminated string")
            v = s[self.i + len(QUOTE):j]; self.i = j + len(QUOTE); return v
        if s[self.i] in "'\"":
            return self.qstr()
        if s[self.i] == "{":
            return self.obj()
        if s[self.i] == "[":
            self.i += 1; arr = []
            self.ws()
            if s[self.i] == "]":
                self.i += 1; return arr
            while True:
                arr.append(self.value()); self.ws()
                if s[self.i] == ",":
                    self.i += 1; continue
                if s[self.i] == "]":
                    self.i += 1; return arr
                raise ValueError("bad array")
        m = re.compile(r'[^,}\]]*').match(s, self.i)
        raw = m.group(0).strip(); self.i = m.end()
        try:
            return json.loads(raw)
        except ValueError:
            return raw

    def qstr(self):
        q, s, out = self.s[self.i], self.s, []
        self.i += 1
        while self.i < len(s):
            ch = s[self.i]
            if ch == "\\" and self.i + 1 < len(s):
                nxt = s[self.i + 1]
                out.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt)); self.i += 2; continue
            if ch == q:
                self.i += 1; return "".join(out)
            out.append(ch); self.i += 1
        raise ValueError("unterminated quoted string")

    def obj(self):
        s = self.s
        assert s[self.i] == "{"; self.i += 1; d = {}
        self.ws()
        if s[self.i] == "}":
            self.i += 1; return d
        while True:
            self.ws()
            m = re.compile(r'\s*["\']?([A-Za-z_][\w.-]*)["\']?\s*:').match(s, self.i)
            if not m:
                raise ValueError("bad key at %d" % self.i)
            self.i = m.end()
            d[m.group(1)] = self.value(); self.ws()
            if s[self.i] == ",":
                self.i += 1; continue
            if s[self.i] == "}":
                self.i += 1; return d
            raise ValueError("bad object at %d" % self.i)


TEXT_CALL = "[tool_call:"


def runaway(text):
    """True if a reply is stuck repeating the same tool call (Gemma did this up to the token limit)."""
    lines = [l.strip() for l in re.findall(r"(?:\[tool_call:[^\n]*|<\|tool_call>[^<]*)", text)]
    if len(lines) < 5:
        return False
    return any(lines.count(l) >= 5 for l in set(lines)) or len(lines) > 24


def parse_text_calls(text, tools):
    """Fallback for Gemma imitating Qwen Code's prompt examples instead of using native tool tokens:
    [tool_call: NAME {key: 'v', ...}]  or  [tool_call: NAME for KEY 'v' with KEY2 'v2']  or  [tool_call: NAME for 'v'].
    Returns (remaining text, calls)."""
    names = {t["function"]["name"]: t["function"] for t in (tools or []) if t.get("type") == "function"}
    calls, out, pos = [], [], 0
    while True:
        i = text.find(TEXT_CALL, pos)
        if i == -1:
            out.append(text[pos:]); break
        out.append(text[pos:i])
        m = re.compile(r"\[tool_call:\s*([\w.-]+)\s*").match(text, i)
        name = m.group(1) if m else None
        j = m.end() if m else i + len(TEXT_CALL)
        args, end = None, None
        try:
            if name and j < len(text) and text[j] == "{":
                p = _Args(text); p.i = j; args = p.obj(); p.ws()
                end = p.i + 1 if p.i < len(text) and text[p.i] == "]" else p.i
            elif name:
                k, q = j, None          # find the closing "]" outside quotes
                while k < len(text):
                    ch = text[k]
                    if q:
                        if ch == "\\":
                            k += 1
                        elif ch == q:
                            q = None
                    elif ch in "'\"":
                        q = ch
                    elif ch == "]":
                        break
                    k += 1
                prose = text[j:k]; end = k + 1
                args = {}
                for km in re.finditer(r"(?:for|with|and)\s+([A-Za-z_]\w*)\s+('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", prose):
                    args[km.group(1)] = _Args(km.group(2)).qstr()
                if not args:
                    vm = re.search(r"for\s+('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", prose)
                    req = (names.get(name, {}).get("parameters") or {}).get("required") or []
                    if vm and req:
                        args[req[0]] = _Args(vm.group(1)).qstr()
        except Exception:  # noqa: BLE001
            args = None
        if name and args is not None and (not names or name in names):
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args)}})
            pos = end if end is not None else len(text)
        else:
            out.append(text[i:j]); pos = j
    return "".join(out), calls


def parse_output(text, tools=None):
    """-> (content, tool_calls or None). Drops thought channels and control tokens."""
    for t in STRIP:
        text = text.replace(t, "")
    text = re.sub(re.escape(CH_OPEN) + r".*?" + re.escape(CH_CLOSE), "", text, flags=re.S)
    if PARSER != "gemma4":
        return text.strip(), None
    if TOOL_OPEN not in text:
        if TEXT_CALL in text:
            rest, calls = parse_text_calls(text, tools)
            if calls:
                return rest.strip(), calls
        return text.strip(), None
    calls, parts, pos = [], [], 0
    while True:
        i = text.find(TOOL_OPEN, pos)
        if i == -1:
            parts.append(text[pos:]); break
        parts.append(text[pos:i])
        j = text.find(TOOL_CLOSE, i)
        payload = text[i + len(TOOL_OPEN):j if j != -1 else len(text)].strip()
        pos = len(text) if j == -1 else j + len(TOOL_CLOSE)
        m = re.match(r"call:([\w.-]+)\s*(\{.*\})\s*$", payload, re.S)
        if not m:
            parts.append(payload); continue
        try:
            args = _Args(m.group(2)).obj()
        except Exception:  # noqa: BLE001 -- keep the raw text so the client sees something
            parts.append(payload); continue
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args)}})
    return "".join(parts).strip(), (calls or None)


def safe_prefix(text):
    """Content that can be streamed now: everything before the first tool-call/channel marker, minus control tokens."""
    cut = min([k for k in (text.find(TOOL_OPEN), text.find(CH_OPEN), text.find(TEXT_CALL)) if k != -1] or [len(text)])
    s = text[:cut]
    for n in range(len(TEXT_CALL) - 1, 0, -1):
        if s.endswith(TEXT_CALL[:n]):
            s = s[:-n]; break
    for t in STRIP:
        s = s.replace(t, "")
    return s.rstrip("�")


