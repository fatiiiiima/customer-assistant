@echo off
cd /d "%~dp0"
rem The five scripted demo scenarios, paced for an audience. Extra arguments are passed through, e.g.  demo_replay.cmd --only 3
.venv\Scripts\python.exe -m dealer_core.cli.replay --backend openvino_gpu --pause 1.5 --debug %*
pause
