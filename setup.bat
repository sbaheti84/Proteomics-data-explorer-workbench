@echo off
rem One-time setup for Proteomics Data Explorer Workbench (Windows).
rem Double-click this file, or run it in a terminal. Options, e.g.:
rem   setup.bat -InstallPrereqs     also install Python, R and Rtools with winget
rem   setup.bat -SkipR              Python only (no NAguideR imputation)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set "SETUP_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %SETUP_EXIT%
