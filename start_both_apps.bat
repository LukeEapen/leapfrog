@echo off
echo ===============================================
echo   COMBINED FLASK APPLICATIONS LAUNCHER
echo ===============================================
echo Starting both Flask applications...
echo.
echo Optimized Backend: http://localhost:5000
echo Three-Section UI:  http://localhost:5001/three-section
echo.
echo Press Ctrl+C to stop both applications
echo ===============================================

REM Start both applications in parallel using PowerShell
powershell -Command "Start-Process python -ArgumentList 'poc2_backend_processor_optimized.py' -WindowStyle Normal; Start-Process python -ArgumentList 'launch_three_section.py' -WindowStyle Normal; Write-Host 'Both applications started! Check the opened windows.'; Read-Host 'Press Enter to continue (this will not stop the applications)'"

echo.
echo Both applications have been started in separate windows.
echo Use the windows that opened to monitor each application.
echo Close those windows or press Ctrl+C in them to stop the applications.
pause
