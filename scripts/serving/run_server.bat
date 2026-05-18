@echo off
REM run_server.bat
REM Script chay FastAPI server cho Toxic Comment Classification (Windows)
REM
REM Cach dung:
REM     scripts\serving\run_server.bat
REM     scripts\serving\run_server.bat --port 5000
REM     scripts\serving\run_server.bat --model-dir C:\models\production

set HOST=0.0.0.0
set PORT=8000
set MODEL_DIR=

:parse
if "%1"=="" goto :run
if "%1"=="--host" set HOST=%2 & shift & shift & goto :parse
if "%1"=="--port" set PORT=%2 & shift & shift & goto :parse
if "%1"=="--model-dir" set MODEL_DIR=%2 & shift & shift & goto :parse
echo Unknown option: %1
exit /b 1

:run
echo ========================================
echo   Toxic Comment Classification API
echo ========================================
echo   Host: %HOST%
echo   Port: %PORT%
if not "%MODEL_DIR%"=="" echo   Model: %MODEL_DIR%
echo ========================================

python -m uvicorn src.serving.app:app --host %HOST% --port %PORT%
