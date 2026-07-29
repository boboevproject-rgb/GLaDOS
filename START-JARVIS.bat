@echo off
rem Jarvis launcher: starts the Claude brain proxy, then the voice assistant.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem Hands: full access (files + programs + shell), as chosen by the owner.
rem Read-only instead: Read,Glob,Grep,WebSearch,WebFetch
set JARVIS_TOOLS=Read,Glob,Grep,WebSearch,WebFetch,Write,Edit,Bash

start "Jarvis Brain (Claude proxy)" cmd /c "uv run --no-sync python scripts\claude_brain_proxy.py & pause"
timeout /t 3 /nobreak >nul
uv run --no-sync glados start --config configs\jarvis_config.yaml
