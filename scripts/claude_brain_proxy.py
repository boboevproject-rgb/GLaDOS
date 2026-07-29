"""Local OpenAI-compatible proxy that gives the assistant a Claude brain.

GLaDOS speaks the OpenAI /v1/chat/completions streaming protocol; this server
translates each request into a headless `claude -p` call (Claude Code CLI,
already authenticated on this machine), and streams the reply back as SSE
deltas, so sentences reach the TTS as they are generated.

Design notes:
- stdlib only (http.server + subprocess) — no extra dependencies.
- The conversation arrives as a full message list every time, so each request
  is a stateless CLI call: system messages become --system-prompt (replacing
  the Claude Code default entirely — no coding persona, fewer tokens), the
  rest is flattened into a transcript fed via stdin (argv has length limits).
- All tools are disallowed: this brain only talks. Actions come in a later
  stage via MCP.

Env vars: CLAUDE_BRAIN_MODEL (default "sonnet"), CLAUDE_BRAIN_PORT (8555).
Run: uv run python scripts/claude_brain_proxy.py
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

MODEL = os.environ.get("CLAUDE_BRAIN_MODEL", "sonnet")
PORT = int(os.environ.get("CLAUDE_BRAIN_PORT", "8555"))
# Actions (presentations, long scripts) can take minutes
CLI_TIMEOUT_S = int(os.environ.get("CLAUDE_BRAIN_TIMEOUT", "300"))
HEARTBEAT_S = 10  # GLaDOS drops the socket after ~30s of silence

# The hands: tools the brain may use without prompts (headless mode cannot ask).
# Voice is an open channel — anything the microphone picks up reaches these
# tools, so the set is deliberately configurable and defaults to read-only.
# Grant write/exec via JARVIS_TOOLS once you trust the setup, e.g.
#   set JARVIS_TOOLS=Read,Glob,Grep,WebSearch,WebFetch,Write,Edit,Bash
SAFE_TOOLS = ["Read", "Glob", "Grep", "WebSearch", "WebFetch"]
FULL_TOOLS = [*SAFE_TOOLS, "Write", "Edit", "Bash"]
ACTION_TOOLS = [t.strip() for t in os.environ.get("JARVIS_TOOLS", ",".join(SAFE_TOOLS)).split(",") if t.strip()]

# Even with full access, a handful of operations are never a real voice command
# and are unrecoverable if a misheard phrase triggers one. Clear with
# JARVIS_NO_GUARDRAILS=1 if you want literally nothing blocked.
DENY_PATTERNS = [
    "Bash(format:*)",
    "Bash(diskpart:*)",
    "Bash(shutdown:*)",
    "Bash(vssadmin delete:*)",
    "Bash(cipher /w:*)",
    "Bash(reg delete HKLM:*)",
    "Bash(rm -rf /:*)",
    "Bash(rd /s /q C:\\Windows:*)",
    "Bash(rd /s /q C:\\Users:*)",
]
if os.environ.get("JARVIS_NO_GUARDRAILS") == "1":
    DENY_PATTERNS = []
WORKSPACE = os.environ.get("JARVIS_WORKSPACE", os.path.join(os.path.expanduser("~"), "Jarvis"))
TIMEOUT_APOLOGY = "Прошу прощения, сэр, задача заняла слишком много времени и была прервана."

CLAUDE_BIN = shutil.which("claude")
if not CLAUDE_BIN:
    sys.exit("claude CLI not found in PATH — install/login Claude Code first")

ROLE_LABELS = {"user": "Пользователь", "assistant": "Джарвис", "tool": "Инструмент"}


def build_prompt(messages: list[dict]) -> tuple[str, str]:
    """Split OpenAI messages into (system_prompt, transcript_for_stdin)."""
    system_parts: list[str] = []
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if not isinstance(content, str):  # ignore structured/tool payloads
            continue
        if role == "system":
            system_parts.append(content)
        elif content.strip():
            lines.append(f"{ROLE_LABELS.get(role, role)}: {content.strip()}")
    system_parts.append(
        "Ниже — стенограмма голосового диалога. Ответь СЛЕДУЮЩЕЙ репликой Джарвиса: "
        "только текст реплики, без имени говорящего, без кавычек и пояснений.\n"
        "У тебя есть руки: инструменты этого Windows-компьютера (запуск программ, "
        "файлы, документы, интернет). Если пользователь просит что-то сделать — "
        "сделай это инструментами, а не рассказывай как. Новые файлы сохраняй в "
        "текущую рабочую папку Jarvis и коротко называй, куда положил. "
        "ВЕСЬ твой текст озвучивается голосом: никакого маркдауна, кода, списков "
        "и технических подробностей в тексте — только короткие разговорные фразы. "
        "Перед долгим действием одной фразой скажи, что приступаешь."
    )
    return "\n\n".join(system_parts), "\n".join(lines) or "Пользователь: Привет"


_EOF = object()


def stream_claude(system_prompt: str, transcript: str):
    """Yield reply text pieces from a headless claude call.

    Yields None as a heartbeat when the CLI is busy with tools (the caller
    turns it into an empty SSE chunk so GLaDOS's read timeout never fires).
    """
    cmd = [
        CLAUDE_BIN,
        "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model", MODEL,
        "--system-prompt", system_prompt,
        "--allowedTools", *ACTION_TOOLS,
        "--permission-mode", "acceptEdits",
    ]
    if DENY_PATTERNS:
        cmd += ["--disallowedTools", *DENY_PATTERNS]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding="utf-8",
        errors="replace",
        cwd=WORKSPACE,
    )
    lines: queue.Queue = queue.Queue()

    def _reader() -> None:
        for raw in proc.stdout:
            lines.put(raw)
        lines.put(_EOF)

    threading.Thread(target=_reader, daemon=True).start()
    try:
        proc.stdin.write(transcript)
        proc.stdin.close()
        streamed_any = False
        deadline = time.time() + CLI_TIMEOUT_S
        while True:
            if time.time() > deadline:
                proc.kill()
                yield ("\n" if streamed_any else "") + TIMEOUT_APOLOGY
                break
            try:
                raw = lines.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                yield None  # heartbeat: tools are working, keep the socket warm
                continue
            if raw is _EOF:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "stream_event":
                inner = event.get("event", {})
                if inner.get("type") == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        streamed_any = True
                        yield delta["text"]
            elif etype == "assistant" and not streamed_any:
                # Fallback if partial events are unavailable: whole blocks
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        yield block["text"]
            elif etype == "result":
                break
    finally:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def sse_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console readable
        pass

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            messages = payload["messages"]
        except (json.JSONDecodeError, KeyError) as e:
            self.send_error(400, f"bad request: {e}")
            return

        system_prompt, transcript = build_prompt(messages)
        want_stream = bool(payload.get("stream", False))
        started = time.time()
        base = {"id": "brain", "object": "chat.completion.chunk", "model": MODEL}

        if want_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def write(data: bytes) -> None:
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            pieces: list[str] = []
            try:
                for piece in stream_claude(system_prompt, transcript):
                    if piece is None:  # heartbeat while tools are working
                        write(b"data: {}\n\n")
                        continue
                    pieces.append(piece)
                    write(sse_chunk({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}))
                write(sse_chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
                write(b"data: [DONE]\n\n")
                write(b"")  # final zero-length chunk ends the response
            except (ConnectionAbortedError, BrokenPipeError):
                pass  # GLaDOS dropped the stream (interruption) — fine
            reply = "".join(pieces)
        else:
            reply = "".join(p for p in stream_claude(system_prompt, transcript) if p)
            body = json.dumps(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        print(
            f"[brain] {time.time() - started:.1f}s | in: {len(transcript)} chars | out: {reply[:60]!r}",
            flush=True,
        )


if __name__ == "__main__":
    os.makedirs(WORKSPACE, exist_ok=True)
    print(
        f"[brain] Claude brain proxy on http://127.0.0.1:{PORT}/v1/chat/completions "
        f"(model={MODEL}, workspace={WORKSPACE}, tools={','.join(ACTION_TOOLS)})",
        flush=True,
    )
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
