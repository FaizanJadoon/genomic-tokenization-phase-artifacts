@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: Virtual environment not found at "%PY%"
  echo Extract this package into D:\research\Faizan\paper-2\genomic-tokenization-phase-artifacts and ensure the existing .venv is present.
  pause
  exit /b 2
)
echo Paper 2: rerunning local experiments and generating HTML report...
"%PY%" "%~dp0reproduce_paper2.py"
set "EXITCODE=%ERRORLEVEL%"
if exist "%~dp0results\publication_report\index.html" (
  start "" "%~dp0results\publication_report\index.html"
)
if not "%EXITCODE%"=="0" echo ATTENTION: Some steps failed or the audit was incomplete. Read the report and logs.
echo.
echo Finished. Exit code: %EXITCODE%
pause
exit /b %EXITCODE%
