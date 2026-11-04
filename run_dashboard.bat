@echo off
REM Starts the TransitVet dashboard in your browser. Run install_windows.bat once first.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (echo Run install_windows.bat first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
streamlit run app.py
pause
