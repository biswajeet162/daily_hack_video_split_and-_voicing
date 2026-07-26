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
echo       Output : 1-input_videos\
echo       Tracker: trackers\download_archive.txt
echo.
echo   [2] Split videos by number markers (1-5)
echo       Input  : 1-input_videos\*.mp4
echo       Refs   : reference_numbers\1.png .. 5.png
echo       Config : configuration\split_trim_config.json
echo       Output : 2-output_videos\^<video name^>\part-XX-*.mp4
echo       Tracker: trackers\split_processed.json
echo.
echo   [3] Blur English text from clips (every frame, GPU OCR)
echo       Input  : 2-output_videos\ (all .mp4 clips)
echo       Config : configuration\remove_text_config.json
echo       Method : detect English text -^> Gaussian blur
echo       Tracker: trackers\text_removed_processed.json
echo.
echo   [4] Transcribe split clips to Hindi (faster-whisper large-v3)
echo       Input  : 2-output_videos\ (all .mp4 clips)
echo       Output : 2-output_videos\Transcriptions\
echo       Tracker: trackers\transcribe_processed.json
echo.
echo   [5] Categorize transcriptions with Ollama (llama3.1:8b)
echo       Input  : 2-output_videos\Transcriptions\*.transcription.json
echo       Output : same JSON file updated with categories
echo       Tracker: trackers\categorize_processed.json
echo.
echo   [6] Voice-over split clips (browser @ localhost:8001)
echo       Input  : 2-output_videos\ clips + Transcriptions\ JSON
echo       Output : 3-output_voiceover_videos\ (clip + your voice)
echo       Tracker: trackers\voiceover_processed.json
echo.
echo   [7] Merge clips by category (interactive menu)
echo       Input  : 2-output_videos\ clips + Transcriptions\ JSON
echo       Config : configuration\merge_transition_config.json
echo       Output : 4-output_merged_videos\ (merged mp4 + dialogue json)
echo       Tracker: trackers\merge_used_clips.json
echo       Prefers : 3-output_voiceover_videos\ when step 6 is done
echo       Note     : use --force if clips were merged before voice-over
echo.
echo   [A] Run ALL steps in order (1 -^> 5, then 7 auto merge)
echo       Steps 1-5 automated. Step 6 voice-over is manual in browser.
echo       Step 7 auto-merges 5 random clips after step 5.
echo.
echo   [0] Exit
echo.
echo ============================================================
set /p choice="Enter task number to run: "

if /i "%choice%"=="A" goto run_all
if /i "%choice%"=="all" goto run_all
if "%choice%"=="1" goto run_01
if "%choice%"=="2" goto run_02
if "%choice%"=="3" goto run_03
if "%choice%"=="4" goto run_04
if "%choice%"=="5" goto run_05
if "%choice%"=="6" goto run_06
if "%choice%"=="7" goto run_07
if "%choice%"=="0" goto end
echo.
echo   [XX] Invalid choice. Try again.
timeout /t 2 >nul
goto menu

:run_01
call :print_task_header 1 "Download YouTube videos" "links.txt to 1-input_videos"
call :run_python 01_download_youtube.py --file links.txt
call :print_task_footer 1
pause
goto menu

:run_02
call :print_task_header 2 "Split videos by number markers" "1-input_videos to 2-output_videos"
call :run_python 02_split_video.py
call :print_task_footer 2
pause
goto menu

:run_03
call :print_task_header 3 "Blur English text from clips" "every frame, detect and blur"
call :run_python 03_remove_text_from_videos.py
call :print_task_footer 3
pause
goto menu

:run_04
call :print_task_header 4 "Transcribe split clips to Hindi" "2-output_videos mp4 to transcription json"
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
call :print_task_header 6 "Voice-over split clips" "browser UI at localhost:8001"
echo   [..] Opening browser at http://localhost:8001
start "" "http://localhost:8001"
call :run_python 06_voiceover_server.py
call :print_task_footer 6
pause
goto menu

:run_07
call :print_task_header 7 "Merge clips by category" "pick category and clip count interactively"
call :run_python 06_merge_by_category.py --interactive
call :print_task_footer 7
pause
goto menu

:run_all
echo.
echo ============================================================
echo   FULL PIPELINE: steps 1 -^> 5, then 7 (auto merge)
echo   Step 6 voice-over is manual - run it separately when ready.
echo   Started: %date%  %time%
echo   Folder : %cd%
echo ============================================================
echo.

call :print_task_header 1 "Download YouTube videos" "links.txt to 1-input_videos"
call :run_python 01_download_youtube.py --file links.txt
if errorlevel 1 goto pipeline_fail
call :print_task_footer 1

call :print_task_header 2 "Split videos by number markers" "1-input_videos to 2-output_videos"
call :run_python 02_split_video.py
if errorlevel 1 goto pipeline_fail
call :print_task_footer 2

call :print_task_header 3 "Blur English text from clips" "every frame, detect and blur"
call :run_python 03_remove_text_from_videos.py
if errorlevel 1 goto pipeline_fail
call :print_task_footer 3

call :print_task_header 4 "Transcribe split clips to Hindi" "2-output_videos mp4 to transcription json"
call :run_python 04_transcribe_videos.py
if errorlevel 1 goto pipeline_fail
call :print_task_footer 4

call :print_task_header 5 "Categorize transcriptions with Ollama" "Hindi text to categories in same JSON"
call :run_python 05_categorize_transcriptions.py
if errorlevel 1 goto pipeline_fail
call :print_task_footer 5

call :print_task_header 7 "Merge clips by category (auto)" "random category, 5 clips (or best available)"
call :run_python 06_merge_by_category.py --auto --count 5
if errorlevel 1 goto pipeline_fail
call :print_task_footer 7

echo.
echo ============================================================
echo   FULL PIPELINE COMPLETED at %date%  %time%
echo   Steps 1-5 and 7 finished. Run step 6 for voice-over when ready.
echo ============================================================
echo.
pause
goto menu

:pipeline_fail
echo.
echo ============================================================
echo   FULL PIPELINE STOPPED - a step failed (see log above)
echo   Fix the issue, then run again or pick a single step.
echo ============================================================
echo.
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
