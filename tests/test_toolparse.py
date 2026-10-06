"""Unit tests for toolparse (no model or GPU needed): python -m unittest discover tests"""
import json, os, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import toolparse as tp  # noqa: E402

Q = tp.QUOTE
TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "required": r, "properties": {}}}}
         for n, r in [("read_file", ["file_path"]), ("run_shell_command", ["command"]), ("write_file", ["file_path", "content"])]]


def calls(text, tools=TOOLS):
    content, cs = tp.parse_output(text, tools)
    return content, [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in cs or []]


class NativeCalls(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(calls("Sure.<|tool_call>call:read_file{file_path:%s/w/a.py%s}<tool_call|><turn|>" % (Q, Q)),
                         ("Sure.", [("read_file", {"file_path": "/w/a.py"})]))

    def test_string_with_commas_braces_quotes(self):
        body = 'a, b = {1: [2, 3]}\nprint("hi, there")'
        text = "<|channel>thought\nhmm<channel|><|tool_call>call:write_file{file_path:%sx.py%s,content:%s%s%s}<tool_call|>" % (Q, Q, Q, body, Q)
        self.assertEqual(calls(text), ("", [("write_file", {"file_path": "x.py", "content": body})]))

    def test_literals_nested_and_multiple(self):
        text = ("<|tool_call>call:run_shell_command{command:%sls -la%s,timeout:30,flags:[%sa%s,2,true],opts:{deep:{x:null}}}<tool_call|>"
                "<|tool_call>call:read_file{file_path:%s**/*.py%s}<tool_call|>") % (Q, Q, Q, Q, Q, Q)
        self.assertEqual(calls(text)[1], [("run_shell_command", {"command": "ls -la", "timeout": 30, "flags": ["a", 2, True],
                                                                "opts": {"deep": {"x": None}}}),
                                          ("read_file", {"file_path": "**/*.py"})])

    def test_plain_text(self):
        self.assertEqual(calls("plain answer, no tools<turn|>"), ("plain answer, no tools", []))


class TextFormCalls(unittest.TestCase):
    """Gemma copying Qwen Code's prompt examples instead of using native tool tokens."""

    def test_object_args(self):
        text = "I will read.\n[tool_call: read_file {file_path: '/w/T.md'}]\n[tool_call: run_shell_command {command: 'ls', description: \"List, files]\"}]"
        self.assertEqual(calls(text), ("I will read.", [("read_file", {"file_path": "/w/T.md"}),
                                                        ("run_shell_command", {"command": "ls", "description": "List, files]"})]))

    def test_prose_forms(self):
        self.assertEqual(calls("[tool_call: run_shell_command for 'ruff check && pytest']")[1],
                         [("run_shell_command", {"command": "ruff check && pytest"})])
        self.assertEqual(calls("[tool_call: write_file for file_path '/w/a.py' with content 'print(\\'hi\\')\\nx = [1, 2]']")[1],
                         [("write_file", {"file_path": "/w/a.py", "content": "print('hi')\nx = [1, 2]"})])

    def test_json_args(self):
        text = '[tool_call: write_file {"file_path": "/w/b.py", "content": "def f():\\n    return {\'a\': [1]}\\n"}]'
        self.assertEqual(calls(text)[1], [("write_file", {"file_path": "/w/b.py", "content": "def f():\n    return {'a': [1]}\n"})])

    def test_unknown_tool_left_as_text(self):
        self.assertEqual(calls("Just text mentioning [tool_call: nope] style."),
                         ("Just text mentioning [tool_call: nope] style.", []))


class Streaming(unittest.TestCase):
    def test_holds_back_partial_and_full_markers(self):
        self.assertEqual(tp.safe_prefix("Reading now. [tool_c"), "Reading now. ")
        self.assertEqual(tp.safe_prefix("ok [tool_call: x {a: 1}]"), "ok ")
        self.assertEqual(tp.safe_prefix("Hi<|tool_call>call:x{"), "Hi")

    def test_runaway(self):
        line = "[tool_call: read_file {file_path: '/w/T3-03.md'}]\n"
        self.assertTrue(tp.runaway(line * 5))
        self.assertFalse(tp.runaway(line * 2))


if __name__ == "__main__":
    unittest.main()
