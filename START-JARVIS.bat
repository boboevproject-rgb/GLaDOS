@echo off
rem Jarvis launcher: starts the Claude brain proxy, then the voice assistant.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem The project lives in OneDrive, which dehydrates files it considers idle.
rem A venv inside it breaks (uv sees a "non-existent interpreter", tries to
rem rebuild, and fails on locked files), so it is kept outside the synced folder.
set UV_PROJECT_ENVIRONMENT=C:\Users\BILOL\.venvs\glados

rem Hands: full access (files + programs + shell), as chosen by the owner.
rem Read-only instead: Read,Glob,Grep,WebSearch,WebFetch
set JARVIS_TOOLS=Read,Glob,Grep,WebSearch,WebFetch,Write,Edit,Bash

start "Jarvis Brain (Claude proxy)" cmd /c "uv run --extra cpu --extra gigaam python scripts\claude_brain_proxy.py & pause"
timeout /t 3 /nobreak >nul
uv run --extra cpu --extra gigaam glados start --config configs\jarvis_config.yaml
if errorlevel 1 pause
