@echo off
rem Jarvis launcher: starts the Claude brain proxy, then the voice assistant.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
start "Jarvis Brain (Claude proxy)" cmd /c "uv run python scripts\claude_brain_proxy.py & pause"
timeout /t 3 /nobreak >nul
uv run glados start --config configs\jarvis_config.yaml
