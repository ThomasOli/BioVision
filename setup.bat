@echo off
setlocal

echo.
echo  ============================================
echo   BioVision Setup
echo  ============================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Download and install Python 3.10+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

:: Show Python version being used
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo Using: %%v
echo.

echo [1/2] Setting up Python backend...
python "%~dp0setup_backend.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed. See output above for details.
    echo.
    pause
    exit /b 1
)

if "%SKIP_NPM_INSTALL%"=="1" (
    echo.
    echo [2/2] Skipping npm install ^(SKIP_NPM_INSTALL=1^).
    goto :done
)

where npm >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARN] npm not found. Frontend dependencies were not installed.
    echo        Install Node.js LTS, then run: npm install
    goto :done
)

echo.
echo [2/2] Installing frontend dependencies ^(npm install^)...
call npm install
if errorlevel 1 (
    echo.
    echo [ERROR] npm install failed.
    echo.
    pause
    exit /b 1
)

:done
echo.
echo Setup complete.
echo Start app with: npm run dev

endlocal
