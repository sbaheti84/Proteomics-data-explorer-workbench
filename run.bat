@echo off
rem Start Proteomics Data Explorer Workbench (Windows). Run setup.bat once first.
rem Close this window (or press Ctrl+C) to stop the app.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 pause
