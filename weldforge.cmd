@echo off
rem ===========================================================================
rem  WeldForge -- one command.
rem
rem    weldforge.cmd                 install what is missing, build the docs
rem                                  and the executables, then start the studio
rem    weldforge.cmd run             just start it
rem    weldforge.cmd install docs    install and build the docs, do not start
rem    weldforge.cmd --force         reinstall from scratch
rem    weldforge.cmd --help          every option
rem
rem  Everything it does lives in tools\bootstrap.py, which imports nothing but
rem  the standard library so that it can run before anything is installed.
rem ===========================================================================
setlocal

cd /d "%~dp0"

rem Find a Python to bootstrap with. The launcher (py.exe) ships with every
rem python.org install and picks the newest interpreter, so it is tried first.
set "WF_PY="
where py >nul 2>nul && set "WF_PY=py -3"
if not defined WF_PY (
  where python >nul 2>nul && set "WF_PY=python"
)
if not defined WF_PY (
  echo.
  echo   Python was not found on PATH.
  echo.
  echo   Install Python 3.10 or newer from https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH" in the installer, then run this again.
  echo.
  exit /b 1
)

rem Once the virtual environment exists, use its interpreter directly: it
rem already has everything, and it keeps the system Python out of the way.
if exist "%~dp0.venv\Scripts\python.exe" set "WF_PY=%~dp0.venv\Scripts\python.exe"

"%WF_PY%" "%~dp0tools\bootstrap.py" %*
exit /b %ERRORLEVEL%
