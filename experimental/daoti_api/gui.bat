@echo off
cd /d "%~dp0"
echo Starting Basti...
start "Basti Server" python web_server.py
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8080
echo Basti is running. Close the server window to exit.
exit
