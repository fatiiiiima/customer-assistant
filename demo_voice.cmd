@echo off
cd /d "%~dp0"
rem Talk to the agent (microphone + speaker). Add --headset if you wear one, --debug to see timings, --list-devices to pick a device.
.venv\Scripts\python.exe -m dealer_core.cli.voice --backend openvino_gpu --debug %*
