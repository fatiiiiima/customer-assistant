@echo off
rem Voice server for the AGMC website. Start this, then the website (bmw-portal\start.bat), then open http://localhost:8080 and press the mic.
cd /d "%~dp0"
.venv\Scripts\python.exe -m dealer_core.server %*
