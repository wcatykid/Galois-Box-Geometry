@echo off
setlocal
cd /d "%~dp0"
set HOST=127.0.0.1
set PORT=8000
set PYTHON_CMD=python
if not "%~1"=="" set PORT=%~1
where python >nul 2>nul && goto run
where py >nul 2>nul && set PYTHON_CMD=py && goto run
if exist "%USERPROFILE%\anaconda3\python.exe" set PYTHON_CMD="%USERPROFILE%\anaconda3\python.exe" && goto run
if exist "%USERPROFILE%\miniconda3\python.exe" set PYTHON_CMD="%USERPROFILE%\miniconda3\python.exe" && goto run
echo Could not find a Python interpreter. Install Python or update start_web_interface.bat.
exit /b 1
:run
echo Starting Galois NLP web interface on http://%HOST%:%PORT%
%PYTHON_CMD% webui_server.py --host %HOST% --port %PORT%
