@echo off
rem One-time setup: a private Python environment beside the tools, and two
rem shortcuts that open them without a console window.
rem
rem   setup.bat                       uses the Python on this machine
rem   setup.bat C:\path\to\python.exe uses that one instead
rem
rem Run it again after moving this folder: the shortcuts hold its path.

setlocal
cd /d "%~dp0"

set "PY="
if not "%~1"=="" set "PY="%~1""
if not defined PY (
    where py >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY goto :no_python

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto :no_python

echo.
echo === Creating the Python environment in .venv
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 goto :failed
)

echo.
echo === Installing numpy, scipy, astropy and pillow
".venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto :failed

".venv\Scripts\python.exe" -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo.
    echo This Python has no tkinter, which the windows need. Reinstall Python
    echo from python.org with "tcl/tk and IDLE" ticked, then run setup again.
    goto :failed
)

echo.
echo === Creating the shortcuts
choice /c YN /m "Also put the shortcuts on the desktop"
if errorlevel 2 (set "DESKTOP=0") else (set "DESKTOP=1")

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$here = (Resolve-Path '.').Path;" ^
  "$pyw = Join-Path $here '.venv\Scripts\pythonw.exe';" ^
  "$shell = New-Object -ComObject WScript.Shell;" ^
  "$targets = @($here);" ^
  "if ('%DESKTOP%' -eq '1') { $targets += [Environment]::GetFolderPath('Desktop') };" ^
  "foreach ($folder in $targets) {" ^
  "  foreach ($tool in @(@('Subframe Inspector', 'inspector_app.py'), @('Subframe Labeller', 'label_frames.py'))) {" ^
  "    $link = $shell.CreateShortcut((Join-Path $folder ($tool[0] + '.lnk')));" ^
  "    $link.TargetPath = $pyw;" ^
  "    $link.Arguments = '\"' + (Join-Path $here $tool[1]) + '\"';" ^
  "    $link.WorkingDirectory = $here;" ^
  "    $link.Save() } }"
if errorlevel 1 goto :failed

echo.
echo Done. Open "Subframe Inspector" to analyse a night,
echo "Subframe Labeller" to go through the frames by eye.
echo.
pause
exit /b 0

:no_python
echo.
echo Python 3.10 or newer was not found.
echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe
echo to PATH"^), then run setup.bat again -- or give it the path of a Python
echo you already have:  setup.bat C:\path\to\python.exe
echo.
pause
exit /b 1

:failed
echo.
echo Setup did not complete -- see the messages above.
echo.
pause
exit /b 1
