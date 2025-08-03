@echo off
REM Start all POC backends in separate windows

REM POC 1
start "POC 1" cmd /k "cd /d %~dp0 && python product_workbench_requirement_definition.py"
REM POC 2
start "POC 2" cmd /k "cd /d %~dp0 && python product_workbench_backlog_management.py"
REM POC 3
start "POC 3" cmd /k "cd /d %~dp0\poc3\backend && python app.py"
REM POC 4
start "POC 4" cmd /k "cd /d %~dp0\poc4 && python migration_reconciliation.py"
REM POC 5
start "POC 5" cmd /k "cd /d %~dp0\poc5 && python product_architecture_definition.py"

echo All POC backends started in separate windows.
pause
