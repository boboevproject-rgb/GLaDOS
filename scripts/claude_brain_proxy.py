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
- The child runs with the proxy's own virtualenv stripped from the
  environment, so `python` inside the agent's shell is the system interpreter
  the user installs libraries into.

Env vars: CLAUDE_BRAIN_MODEL (default "sonnet"), CLAUDE_BRAIN_PORT (8555).
Run: uv run python scripts/claude_brain_proxy.py
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os

from dotenv import load_dotenv

# Secrets (FISH_API_KEY for the voice, CLAUDE_CODE_OAUTH_TOKEN for the brain)
# live in the project's local .env; the CLI child inherits them via _child_env.
load_dotenv()
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

MODEL = os.environ.get("CLAUDE_BRAIN_MODEL", "sonnet")
PORT = int(os.environ.get("CLAUDE_BRAIN_PORT", "8555"))
# Actions (presentations, long scripts) can take minutes
CLI_TIMEOUT_S = int(os.environ.get("CLAUDE_BRAIN_TIMEOUT", "420"))
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
# The CLI reports failures as an assistant message whose text is in English —
# read aloud it is both confusing and useless, so known cases are replaced with
# something the owner can act on.
AUTH_APOLOGY = (
    "Сэр, срок моей авторизации истёк. Откройте терминал, запустите claude "
    "и войдите в аккаунт — после этого я снова в вашем распоряжении."
)
ERROR_APOLOGY = "Прошу прощения, сэр, мой мыслительный модуль вернул ошибку."

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
        "Офисные файлы делай в родных форматах установленными библиотеками: "
        "презентации — python-pptx в .pptx, документы — python-docx в .docx, "
        "таблицы — openpyxl в .xlsx (на компьютере есть Microsoft Office). "
        "HTML-страницу вместо этого делай, только если её прямо попросили. "
        "Имена файлов давай латиницей без пробелов. "
        "У тебя есть глаза: чтобы увидеть экран, выполни "
        "`python tools/screenshot.py` и затем прочитай полученный файл "
        "tools/screen.png инструментом Read — ты увидишь изображение. "
        "Делай так всегда, когда спрашивают про экран, окно, ошибку на нём "
        "или просят прочитать что-то с экрана; не отвечай, что не видишь. "
        "У тебя есть руки для мыши и клавиатуры: "
        "`python tools/click.py X Y` (можно --double, --right, --move) кликает "
        "по точке, где ты УВИДЕЛ её на снимке — координаты бери прямо с "
        "картинки, скрипт сам пересчитает их в экранные, сам не умножай. "
        "`python tools/type.py \"текст\"` вводит текст в активное окно "
        "(можно --enter), `--key enter` жмёт клавишу, `--hotkey ctrl s` — "
        "сочетание. Перед кликом ОБЯЗАТЕЛЬНО делай свежий снимок: окна двигаются. "
        "Перед вводом текста тоже сначала снимок — текст уходит в то окно, что "
        "сейчас в фокусе, и попасть не туда легко. Убедись по снимку, что "
        "активно нужное окно и курсор стоит в нужном поле; если нет — сначала "
        "кликни в это поле, сделай снимок ещё раз и только потом печатай. "
        "Сайты открывай командой `python tools/open_url.py АДРЕС` — она "
        "открывает страницу в обычном браузере пользователя, где он уже вошёл "
        "в свои аккаунты; не ищи браузер на экране и не набирай адрес вручную. "
        "Работа с формами — ВАЖНО, делай именно так, иначе уходит слишком много "
        "времени: открой страницу, сделай ОДИН снимок, кликни в первое поле и "
        "передай ВСЕ значения разом командой "
        "`python tools/fill_form.py \"значение1\" \"значение2\" ...` — она сама "
        "идёт по полям клавишей Tab в порядке формы и гасит подсказки "
        "автозаполнения. Значения перечисляй в том порядке, в каком поля идут "
        "на странице; для выпадающего списка просто передай текст пункта. "
        "Затем сделай ещё один снимок и проверь глазами результат. Не заполняй "
        "поля по одному со снимком после каждого — это слишком долго. "
        "ПРАВИЛО ПОДТВЕРЖДЕНИЯ: необратимые и внешние действия — отправка "
        "письма или сообщения, оплата и переводы, публикация, удаление файлов "
        "и писем, подтверждение форм и согласий, установка и удаление программ, "
        "изменение настроек системы — НЕ выполняй сразу. Подготовь всё до "
        "последнего шага, затем ОСТАНОВИСЬ и спроси голосом коротко, например "
        "«Письмо готово. Отправить, сэр?» — и жди ответа. Выполняй только "
        "после явного согласия в следующей реплике. Обычные действия "
        "(открыть, посмотреть, найти, создать файл, напечатать текст) делай "
        "сразу, без вопросов. "
        "ВЕСЬ твой текст озвучивается голосом: никакого маркдауна, кода, списков "
        "и технических подробностей в тексте — только короткие разговорные фразы. "
        "Перед долгим действием одной фразой скажи, что приступаешь."
    )
    return "\n\n".join(system_parts), "\n".join(lines) or "Пользователь: Привет"


def _child_env() -> dict[str, str]:
    """Environment for the agent, with our own virtualenv taken out of the way.

    The proxy itself is started through `uv run`, which exports VIRTUAL_ENV and
    puts that venv first on PATH. Those leak into every shell the agent opens,
    so `python` resolved to the assistant's own environment — which lacks the
    libraries the user installs system-wide — and scripts failed or got
    installed into the wrong place.
    """
    env = os.environ.copy()
    venv = env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    if venv:
        scripts = (Path(venv) / "Scripts").resolve()
        kept = []
        for part in env.get("PATH", "").split(os.pathsep):
            if not part:
                continue
            try:
                if Path(part).resolve() == scripts:
                    continue
            except OSError:
                pass
            kept.append(part)
        env["PATH"] = os.pathsep.join(kept)
    return env


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
        env=_child_env(),
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
        # Measured against what we send downstream, not against CLI chatter:
        # while a tool runs the CLI keeps emitting events we do not forward, so
        # the client would sit without a single byte and hit its read timeout.
        last_emit = time.monotonic()
        while True:
            if time.time() > deadline:
                proc.kill()
                yield ("\n" if streamed_any else "") + TIMEOUT_APOLOGY
                break
            try:
                raw = lines.get(timeout=1.0)
            except queue.Empty:
                raw = None
            if raw is _EOF:
                break
            if raw is None:
                if time.monotonic() - last_emit >= HEARTBEAT_S:
                    yield None  # keep the connection warm while tools work
                    last_emit = time.monotonic()
                continue
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            failure = event.get("error") if etype == "assistant" else None
            if etype == "result" and event.get("is_error"):
                failure = failure or "error"
            if failure:
                print(f"[brain] сбой Claude CLI: {failure}", flush=True)
                if not streamed_any:
                    yield AUTH_APOLOGY if "auth" in str(failure) else ERROR_APOLOGY
                break
            if etype == "stream_event":
                inner = event.get("event", {})
                if inner.get("type") == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        streamed_any = True
                        last_emit = time.monotonic()
                        yield delta["text"]
            elif etype == "assistant" and not streamed_any:
                # Fallback if partial events are unavailable: whole blocks
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        last_emit = time.monotonic()
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


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that does not dump a traceback when a client hangs up.

    GLaDOS keeps connections alive and drops them whenever a reply is
    cancelled or an idle socket is recycled, which is normal here and would
    otherwise flood the console with connection-reset stack traces.
    """

    def handle_error(self, request: object, client_address: object) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, ConnectionResetError | ConnectionAbortedError | BrokenPipeError):
            return
        super().handle_error(request, client_address)  # type: ignore[arg-type]


if __name__ == "__main__":
    os.makedirs(WORKSPACE, exist_ok=True)
    print(
        f"[brain] Claude brain proxy on http://127.0.0.1:{PORT}/v1/chat/completions "
        f"(model={MODEL}, workspace={WORKSPACE}, tools={','.join(ACTION_TOOLS)})",
        flush=True,
    )
    QuietThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
