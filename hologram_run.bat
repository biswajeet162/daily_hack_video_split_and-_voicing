@echo off
setlocal EnableExtensions
cd /d "%~dp02-HOLO_GRAM_VOICE_OVER"

title Hologram Voice Over
color 0B
chcp 65001 >nul

set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "CONDA_NO_PLUGINS=true"

cls
echo.
echo ============================================================
echo   HOLOGRAM VOICE OVER
echo   Conda env : whisperx
echo   Project   : %~dp02-HOLO_GRAM_VOICE_OVER
echo   URL       : http://localhost:8000
echo ============================================================
echo.
echo   1. Select a video in the browser
echo   2. Set chunk size and click Process Video
echo   3. Record voice-over for each chunk
echo   4. Click Merge and download final_output.mp4
echo.
echo   Press Ctrl+C to stop the server.
echo ============================================================
echo.

start "" "http://localhost:8000"

echo   [..] Starting server with conda env: whisperx
echo.
echo ------------------------------------------------------------
call conda run --no-capture-output -n whisperx python server.py
set "EXIT_CODE=%ERRORLEVEL%"
echo ------------------------------------------------------------
echo.

if "%EXIT_CODE%"=="0" (
    echo   [OK]  Server stopped normally.
) else (
    echo   [XX]  Server exited with code %EXIT_CODE%.
)
echo.
pause
endlocal
exit /b %EXIT_CODE%
