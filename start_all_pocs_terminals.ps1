# PowerShell script to start all POC backends in separate VS Code terminals
$commands = @(
    'cd $PSScriptRoot; python .\product_workbench_requirement_definition.py',
    'cd $PSScriptRoot; python .\product_workbench_backlog_management.py',
    'cd $PSScriptRoot\poc3\backend; python app.py',
    'cd $PSScriptRoot\poc4\backend; python migration_reconciliation.py',
    'cd $PSScriptRoot\poc5; python product_architecture_definition.py'
)

for ($i=0; $i -lt $commands.Count; $i++) {
    $name = "POC $($i+1)"
    $cmd = $commands[$i]
    # Open a new VS Code terminal tab and run the command
    code -r --new-window . # Ensures VS Code is open in workspace
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd -WindowStyle Normal
}
