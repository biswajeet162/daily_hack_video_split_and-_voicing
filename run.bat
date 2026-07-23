@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title DAILY HACKS - Task Menu
color 0A

:menu
cls
echo ============================================
echo   DAILY HACKS VIDEOS - Task Runner
echo   Conda env: utube_env
echo ============================================
echo.
echo   [1] Download YouTube videos
echo       links.txt  -^>  input_videos\
echo.
echo   [2] Split video by number markers (1-5)
echo       input_videos\  -^>  output_videos\
echo.
echo   [0] Exit
echo.
echo ============================================
set /p choice="Enter file number to run: "

if "%choice%"=="1" goto run_01
if "%choice%"=="2" goto run_02
if "%choice%"=="0" goto end
echo.
echo Invalid choice. Try again.
timeout /t 2 >nul
goto menu

:run_01
echo.
echo Running [1] Download YouTube...
call conda run -n utube_env python "01_download_youtube.py" --file links.txt
echo.
echo --------------------------------------------
pause
goto menu

:run_02
echo.
echo Running [2] Split video by number markers...
call conda run -n utube_env python "02_split_video.py" --latest-only
echo.
echo --------------------------------------------
pause
goto menu

:end
endlocal
exit /b 0
