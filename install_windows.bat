@echo off
REM TransitVet one-click install for Windows. Double-click this file once.
REM Needs Python 3.11, 3.12 or 3.13 from python.org (tick "Add python.exe to PATH" when installing).
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version || (echo Python not found. Install it from https://www.python.org/downloads/ and run this again. & pause & exit /b 1)
if not exist .venv (
  echo Creating a private Python environment in .venv ...
  %PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || (echo Install failed: see the messages above. & pause & exit /b 1)
echo.
echo Checking the install with the automated tests (about 3 minutes)...
python -m transitvet test
echo.
echo Done. Double-click run_dashboard.bat to start TransitVet.
pause
