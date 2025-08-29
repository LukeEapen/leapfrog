@echo off
REM ==============================================================
REM Stop all POCs (1,2,3,3b,4,5) launched by start_all_pocs.bat
REM This locates cmd windows by their titles ("POC 1", etc.) and
REM kills the entire process tree so child Python processes exit.
REM Usage:  stop_all_pocs.bat
REM ==============================================================

setlocal enabledelayedexpansion

echo Stopping all POC windows...

REM Important: stop "POC 3b" before "POC 3" to avoid partial matches
for %%T in ("POC 3b" "POC 1" "POC 2" "POC 3" "POC 4" "POC 5") do (
  echo.
  echo - Looking for %%~T windows
  REM Find cmd.exe processes whose window title contains the POC label, then kill by PID with tree
  for /f "tokens=2" %%P in ('tasklist /v /fi "imagename eq cmd.exe" ^| findstr /i /c:"%%~T"') do (
    echo   Killing cmd.exe PID %%P for %%~T
    taskkill /pid %%P /t /f >nul 2>&1
  )

  REM Fallback: if any python consoles have the title (rare), kill them too
  for /f "tokens=2" %%P in ('tasklist /v /fi "imagename eq python.exe" ^| findstr /i /c:"%%~T"') do (
    echo   Killing python.exe PID %%P for %%~T
    taskkill /pid %%P /t /f >nul 2>&1
  )
  for /f "tokens=2" %%P in ('tasklist /v /fi "imagename eq pythonw.exe" ^| findstr /i /c:"%%~T"') do (
    echo   Killing pythonw.exe PID %%P for %%~T
    taskkill /pid %%P /t /f >nul 2>&1
  )
)

echo.
echo All POC windows signaled to stop.
echo If some remain, close them manually or rerun this script.
echo.
pause
