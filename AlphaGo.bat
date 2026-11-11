@echo off
rem ============================================================================================
rem  AlphaGo launcher and installer. Double-click it.
rem    first time : finds Python 3.11+ (offers to install it), creates .venv, installs AlphaGo,
rem                 offers a desktop shortcut, then starts the app
rem    afterwards : starts the app (re-installs by itself when pyproject.toml changed)
rem  Options (from a shortcut or a terminal):
rem    AlphaGo.bat --web        the app in your browser (use this if the window does not open)
rem    AlphaGo.bat --headless   the engine without a window (UI at http://127.0.0.1:8090)
rem    AlphaGo.bat --update     re-install the packages, then exit
rem    AlphaGo.bat --shortcut   (re)create the desktop shortcut
rem    AlphaGo.bat --test       run the test suite
rem    AlphaGo.bat --build      build dist\AlphaGo\AlphaGo.exe
rem ============================================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title AlphaGo
set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"
set "MARK=%VENV%\.alphago-installed"
set "MODE=%~1"

if /i "%MODE%"=="--run" goto :run
if /i "%MODE%"=="--help" goto :help
if /i "%MODE%"=="/?" goto :help

if not exist "%PY%" goto :install
if /i "%MODE%"=="--update" goto :deps
fc /b "pyproject.toml" "%MARK%" >nul 2>&1 || goto :deps
goto :launch

rem ------------------------------------------------------------------------------ install
:install
echo.
echo  Setting up AlphaGo for the first time. This takes a few minutes, once.
echo.
call :findpython
if not defined BASEPY goto :nopython
echo  Using Python: %BASEPY%
"%BASEPY%" -m venv "%VENV%"
if errorlevel 1 goto :fail
set "FIRST=1"

:deps
echo.
echo  Installing AlphaGo's packages (internet needed)...
"%PY%" -m pip install --disable-pip-version-check --upgrade pip >nul
"%PY%" -m pip install --disable-pip-version-check -e .
if errorlevel 1 goto :fail
copy /y "pyproject.toml" "%MARK%" >nul
echo  Done.
if defined FIRST (
    if not exist "%USERPROFILE%\Desktop\AlphaGo.lnk" (
        choice /c YN /m "  Create a desktop shortcut"
        if not errorlevel 2 call :shortcut
    )
)
if /i "%MODE%"=="--update" (
    echo  AlphaGo is up to date.
    pause
    exit /b 0
)

rem ------------------------------------------------------------------------------ launch
:launch
if /i "%MODE%"=="--shortcut" (
    call :shortcut
    pause
    exit /b 0
)
if /i "%MODE%"=="--test" (
    "%PY%" -m pip install --disable-pip-version-check -q -e ".[dev]"
    "%PY%" -m pytest -q
    pause
    exit /b
)
if /i "%MODE%"=="--build" (
    "%PY%" -m pip install --disable-pip-version-check -q -e ".[dev]"
    "%PY%" scripts\build_exe.py
    pause
    exit /b
)
if /i "%MODE%"=="--web" (
    echo  AlphaGo is running at http://127.0.0.1:8090  -  close this window or press Ctrl+C to stop it.
    "%PY%" -m alphago --web
    exit /b
)
if /i "%MODE%"=="--headless" (
    echo  AlphaGo engine running, UI at http://127.0.0.1:8090  -  close this window to stop it.
    "%PY%" -m alphago --headless
    exit /b
)
rem the window app: run it from a minimised console so a crash can still be reported
start "AlphaGo" /min "%~f0" --run
exit /b 0

:run
"%PY%" -m alphago
if errorlevel 1 (
    powershell -NoProfile -Command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('AlphaGo stopped with an error.' + [Environment]::NewLine + 'Details: %APPDATA%\AlphaGo\logs\alphago.log' + [Environment]::NewLine + [Environment]::NewLine + 'If the window does not open at all, start AlphaGo.bat --web instead.', 'AlphaGo') | Out-Null"
)
exit /b

rem ------------------------------------------------------------------------------ helpers
:findpython
set "BASEPY="
for %%V in (3.13 3.12 3.11 3.14) do (
    if not defined BASEPY (
        py -%%V -c "import sys" >nul 2>&1 && for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)"') do set "BASEPY=%%P"
    )
)
if not defined BASEPY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)"') do set "BASEPY=%%P"
)
exit /b 0

:nopython
echo.
echo  Python 3.11 or newer was not found.
where winget >nul 2>&1
if errorlevel 1 (
    echo  Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then double-click AlphaGo.bat again.
    start "" "https://www.python.org/downloads/"
    pause
    exit /b 1
)
choice /c YN /m "  Install Python 3.13 now with winget"
if errorlevel 2 (
    echo  Install Python 3.11+ from https://www.python.org/downloads/ and double-click AlphaGo.bat again.
    pause
    exit /b 1
)
winget install -e --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements
echo.
echo  Python is installed. Close this window and double-click AlphaGo.bat again to finish the setup.
pause
exit /b 0

:shortcut
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\AlphaGo.lnk'); $s.TargetPath='%~f0'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.IconLocation='%SystemRoot%\System32\imageres.dll,186'; $s.Description='AlphaGo - WorldQuant BRAIN research'; $s.Save()"
if errorlevel 1 (
    echo  Could not create the shortcut.
) else (
    echo  Desktop shortcut AlphaGo created.
)
exit /b 0

:fail
echo.
echo  Setup failed (see the messages above). Check your internet connection and run AlphaGo.bat again.
echo  To start over, delete the .venv folder next to AlphaGo.bat.
pause
exit /b 1

:help
echo.
echo  AlphaGo.bat              start AlphaGo (installs it the first time)
echo  AlphaGo.bat --web        start it in your browser
echo  AlphaGo.bat --headless   start the engine without a window
echo  AlphaGo.bat --update     re-install the packages
echo  AlphaGo.bat --shortcut   create the desktop shortcut
echo  AlphaGo.bat --test       run the tests
echo  AlphaGo.bat --build      build dist\AlphaGo\AlphaGo.exe
exit /b 0
