@echo off
cd /d "%~dp0"
rem Interactive chat with the live trace. Extra arguments are passed through, e.g.  demo_chat.cmd --backend ollama
.venv\Scripts\python.exe -m dealer_core.cli.chat --backend openvino_gpu --debug %*
