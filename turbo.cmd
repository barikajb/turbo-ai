@echo off
rem Run TURBO with the project's virtual environment from any folder.
rem Add this folder to PATH to use `turbo fix app.py` anywhere.
"%~dp0.venv\Scripts\python.exe" "%~dp0turbo.py" %*
