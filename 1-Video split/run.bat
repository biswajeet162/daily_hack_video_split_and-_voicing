@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title Video Split - Task Menu
color 0A
chcp 65001 >nul

set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "CONDA_NO_PLUGINS=true"

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
echo   [3] Remove on-screen text from clips (per-frame, GPU OCR)
echo       Input  : output_videos\ (all .mp4 clips)
echo       Langs  : English, Hindi, Chinese (EasyOCR)
echo       Output : same clip files, text inpainted out
echo       Tracker: output_videos\.text_removed_processed.json
echo.
echo   [4] Transcribe split clips to Hindi (faster-whisper large-v3)
echo       Input  : output_videos\ (all .mp4 clips)
echo       Output : output_videos\Transcriptions\
echo       Tracker: output_videos\.transcribe_processed.json
echo.
echo   [5] Categorize transcriptions with Ollama (llama3.1:8b)
echo       Input  : output_videos\Transcriptions\*.transcription.json
echo       Output : same JSON file updated with categories
echo       Tracker: output_videos\.categorize_processed.json
echo.
echo   [6] Merge clips by category (interactive menu)
echo       Input  : output_videos\ clips + Transcriptions\ JSON
echo       Output : output_merged_videos\ (merged mp4 + dialogue json)
echo       Effects: merge_transition_config.json (random gap blink, no overlap)
echo       Tracker: output_videos\.merge_used_clips.json
echo.
echo   [0] Exit
echo.
echo ============================================================
set /p choice="Enter task number to run: "

if "%choice%"=="1" goto run_01
if "%choice%"=="2" goto run_02
if "%choice%"=="3" goto run_03
if "%choice%"=="4" goto run_04
if "%choice%"=="5" goto run_05
if "%choice%"=="6" goto run_06
if "%choice%"=="0" goto end
echo.
echo   [XX] Invalid choice. Try again.
timeout /t 2 >nul
goto menu

:run_01
call :print_task_header 1 "Download YouTube videos" "links.txt to input_videos"
call :run_python 01_download_youtube.py --file links.txt
call :print_task_footer 1
pause
goto menu

:run_02
call :print_task_header 2 "Split videos by number markers" "input_videos to output_videos"
call :run_python 02_split_video.py
call :print_task_footer 2
pause
goto menu

:run_03
call :print_task_header 3 "Remove on-screen text from clips" "per-frame GPU OCR + inpainting"
call :run_python 03_remove_text_from_videos.py
call :print_task_footer 3
pause
goto menu

:run_04
call :print_task_header 4 "Transcribe split clips to Hindi" "output_videos mp4 to transcription json"
call :run_python 04_transcribe_videos.py
call :print_task_footer 4
pause
goto menu

:run_05
call :print_task_header 5 "Categorize transcriptions with Ollama" "Hindi text to categories in same JSON"
call :run_python 05_categorize_transcriptions.py
call :print_task_footer 5
pause
goto menu

:run_06
call :print_task_header 6 "Merge clips by category" "pick category and clip count interactively"
call :run_python 06_merge_by_category.py --interactive
call :print_task_footer 6
pause
goto menu

:print_task_header
echo.
echo ============================================================
echo   TASK [%~1] %~2
echo   Flow   : %~3
echo   Started: %date%  %time%
echo   Folder : %cd%
echo ============================================================
echo.
exit /b 0

:print_task_footer
echo.
echo ------------------------------------------------------------
echo   TASK [%~1] finished at %date%  %time%
echo   Log above shows each step, file path, and output location.
echo ------------------------------------------------------------
echo.
exit /b 0

:run_python
echo   [..] Launching: python %*
echo   [..] Using conda env: utube_env  (live output below)
echo.
echo ------------------------------------------------------------
call conda run --no-capture-output -n utube_env python %*
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
