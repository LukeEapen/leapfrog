@echo off
REM ==============================================================
REM Launch all POCs (1,2,3,3b,4,5) in separate CMD windows
REM Usage:  start_all_pocs.bat
REM NOTE: Ensure virtual env / dependencies installed beforehand.
REM ==============================================================

set ROOT=%~dp0

REM POC 1 (Requirement Definition)
start "POC 1" cmd /k "cd /d %ROOT% && python product_workbench_requirement_definition.py"
REM POC 2 (Backlog Management)
start "POC 2" cmd /k "cd /d %ROOT% && python product_workbench_backlog_management.py"
REM POC 3 (Original Synth / etc)
start "POC 3" cmd /k "cd /d %ROOT%poc3\backend && python app.py"
REM POC 3b (Enhanced Synth)
start "POC 3b" cmd /k "cd /d %ROOT%poc3b\backend && python app.py"
REM POC 4 (Migration Reconciliation)
start "POC 4" cmd /k "cd /d %ROOT%poc4 && python migration_reconciliation.py"
REM POC 5 (Architecture Definition)
start "POC 5" cmd /k "cd /d %ROOT%poc5 && python product_architecture_definition.py"

echo.
echo All POCs launched. Accessible URLs (default):
echo   POC1: http://127.0.0.1:5001/
echo   POC2: http://127.0.0.1:5002/tabbed-layout
echo   POC3: http://127.0.0.1:5050
echo   POC3b:http://127.0.0.1:5051
echo   POC4: http://127.0.0.1:5000
echo   POC5: http://127.0.0.1:6060
echo.
pause
