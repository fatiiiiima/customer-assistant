@echo off
cd /d "%~dp0"
rem Talk to the agent (microphone + speaker) with a live transcript of both sides. Add --headset if you wear one;
rem --save-transcript calls.txt also writes the transcript to a file; --list-devices to pick a device.
.venv\Scripts\python.exe -m dealer_core.cli.voice --backend openvino_gpu --debug --transcript %*
