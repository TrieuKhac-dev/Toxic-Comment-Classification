@echo off
REM deploy.bat — Tu dong deploy model 'ban3_baseline_lr' vao registry
REM Script duoc sinh tu dong boi ModelPackager

set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

echo ========================================
echo   🚀 Deploying model 'ban3_baseline_lr'...
echo ========================================

python scripts/serving/deploy_model.py ban3_baseline_lr --from-folder "%SCRIPT_DIR%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo   ✅ Deploy thanh cong!
    echo   🚀 Start server: python scripts/serving/run_server.py
    echo.
) else (
    echo.
    echo   ❌ Deploy that bai. Xem log o tren.
    echo.
)

pause
