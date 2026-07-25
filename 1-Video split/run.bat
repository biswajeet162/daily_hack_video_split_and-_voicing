@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title Video Split - Task Runner
color 0A
chcp 65001 >nul

set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "CONDA_NO_PLUGINS=true"

title Video Split - Task Menu

:menu
cls
echo.
echo ============================================================
echo   VIDEO SPLIT - Task Runner
echo   Conda env : utube_env
echo   Project   : %~dp0
echo ============================================================
echo.
echo   [1] Download YouTube videos
echo       Input  : links.txt
echo       Output : input_videos\
echo       Tracker: input_videos\.download_archive.txt
echo.
echo   [2] Split videos by number markers (1-5)
echo       Input  : input_videos\*.mp4
echo       Refs   : reference_numbers\1.png .. 5.png
echo       Output : output_videos\^<video name^>\part-XX-*.mp4
echo       Tracker: input_videos\.split_processed.json
echo.
echo   [3] Transcribe split clips to Hindi (faster-whisper large-v3)
echo       Input  : output_videos\**\*.mp4
echo       Output : output_videos\**\*.transcription.json
echo       Tracker: output_videos\.transcribe_processed.json
echo.
echo   [0] Exit
echo.
echo ============================================================
set /p choice="Enter task number to run: "

if "%choice%"=="1" goto run_01
if "%choice%"=="2" goto run_02
if "%choice%"=="3" goto run_03
if "%choice%"=="0" goto end
echo.
echo   [XX] Invalid choice. Try again.
timeout /t 2 >nul
goto menu

:run_01
call :task_header 1 "Download YouTube videos" "links.txt  --^>  input_videos\"
call :run_python "01_download_youtube.py" --file links.txt
call :task_footer 1
pause
goto menu

:run_02
call :task_header 2 "Split videos by number markers" "input_videos\  --^>  output_videos\"
call :run_python "02_split_video.py"
call :task_footer 2
pause
goto menu

:run_03
call :task_header 3 "Transcribe split clips to Hindi" "output_videos\*.mp4  --^>  *.transcription.json"
call :run_python "03_transcribe_videos.py"
call :task_footer 3
pause
goto menu

:task_header
echo.
echo ============================================================
echo   TASK [%1] %~2
echo   Flow   : %~3
echo   Started: %date%  %time%
echo   Folder : %cd%
echo ============================================================
echo.
exit /b 0

:task_footer
echo.
echo ------------------------------------------------------------
echo   TASK [%1] finished at %date%  %time%
echo   Log above shows each step, file path, and output location.
echo ------------------------------------------------------------
echo.
exit /b 0

:run_python
echo   [..] Launching: python %*
echo   [..] Using conda env: utube_env  (live output below)
echo.
echo ------------------------------------------------------------
conda run --no-capture-output -n utube_env python %*
set "EXIT_CODE=%ERRORLEVEL%"
echo ------------------------------------------------------------
echo.
if "%EXIT_CODE%"=="0" (
    echo   [OK]  Python script finished successfully.
) else (
    echo   [XX]  Python script exited with code %EXIT_CODE%.
)
exit /b %EXIT_CODE%

:end
endlocal
exit /b 0
