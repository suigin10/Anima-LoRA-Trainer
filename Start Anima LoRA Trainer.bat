@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
  py -3.12 launch.py --background
  if errorlevel 1 py -3.11 launch.py --background
) else (
  python launch.py --background
)
if errorlevel 1 pause
