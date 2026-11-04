@echo off
REM Analyses every star listed in stars.txt (one ID per line) and writes results\summary.csv.
REM Safe to stop and start again: finished stars are skipped.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (echo Run install_windows.bat first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
if not exist stars.txt (echo TIC 1333933> stars.txt & echo KIC 11442793>> stars.txt)
python -m transitvet run --file stars.txt --out results
pause
