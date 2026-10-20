@echo off
cd /d "%~dp0"
rem Interactive chat, HYBRID replies: a verified model-written lead-in sentence before low-risk templated answers (adds ~0.5-1 s per turn).
.venv\Scripts\python.exe -m dealer_core.cli.chat --backend openvino_gpu --hybrid --debug %*
