@echo off
cd /d "%~dp0"

REM Refresh PATH to include winget-installed ngrok
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "MACHINE_PATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "USER_PATH=%%b"
set "PATH=%MACHINE_PATH%;%USER_PATH%;%PATH%"

if not exist ".env" (
    echo WARNING: .env file not found. ngrok may not have auth token.
    echo Create .env with: NGROK_AUTH_TOKEN=your_token_here
    echo.
)

echo Starting Watch Timer...
uv run web_app.py

pause
