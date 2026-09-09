@echo off
rem start_worker_windows.bat — double-click to launch a MiniMPI student worker.
rem 1) Asks for the teacher's IP:Port (default shown in the prompt).
rem 2) Runs: python worker.py --server <ip>:<port>   (uses `py -3` if `python` missing)
rem 3) Keeps the window open so you can see errors.

chcp 65001 >nul
set "PYTHONUTF8=1"

set "DEFAULT=192.168.1.100:9000"   rem change to your teacher's IP
set "ADDR="
set /p "ADDR=Teacher IP:Port [%DEFAULT%]: "
if "%ADDR%"=="" set "ADDR=%DEFAULT%"
echo %ADDR% | findstr /r ":" >nul 2>nul
if errorlevel 1 set "ADDR=%ADDR%:9000"

where python >nul 2>nul
if %errorlevel%==0 ( set "PY=python" ) else ( set "PY=py -3" )

cd /d "%~dp0.."
echo.
echo Connecting to teacher %ADDR% ...
%PY% -u worker.py --server "%ADDR%"
echo.
echo Worker exited with code %errorlevel%.
pause
