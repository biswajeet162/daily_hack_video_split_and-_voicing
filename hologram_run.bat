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
echo   Which option do you want?
echo.
echo   [1] Split One Video
echo       Select one video, set chunk size, process, record, merge
echo.
echo   [2] Load Video Chunks
echo       Select pre-split videos, click each to transcribe/record,
echo       reorder with up/down, then merge
echo.
echo ============================================================
echo.

:ask_option
set "CHOICE="
set /p "CHOICE=Enter 1 or 2: "

if "%CHOICE%"=="1" goto option_one
if "%CHOICE%"=="2" goto option_two

echo.
echo   Invalid choice. Please type 1 or 2.
echo.
goto ask_option

:option_one
set "APP_URL=http://localhost:8000/?mode=split"
echo.
echo   Starting Option 1: Split One Video
goto start_app

:option_two
set "APP_URL=http://localhost:8000/?mode=chunks"
echo.
echo   Starting Option 2: Load Video Chunks
goto start_app

:start_app
echo   Press Ctrl+C to stop the server.
echo ============================================================
echo.

start "" "%APP_URL%"

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
