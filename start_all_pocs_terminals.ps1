<#
Starts ALL POCs (1,2,3,3b,4,5) each in its own PowerShell window.
Usage (from repo root):
    powershell -ExecutionPolicy Bypass -File .\start_all_pocs_terminals.ps1

Logs: Each window shows live output. Optional: uncomment Out-File section to capture logs.
#>

$root = $PSScriptRoot

$items = @(
        @{ Name = 'POC 1'; Path = "$root";                Cmd = 'python .\\product_workbench_requirement_definition.py'; Url='http://127.0.0.1:5001/' },
        @{ Name = 'POC 2'; Path = "$root";                Cmd = 'python .\\product_workbench_backlog_management.py';     Url='http://127.0.0.1:5002/tabbed-layout' },
        @{ Name = 'POC 3'; Path = "$root\\poc3\\backend"; Cmd = 'python app.py';                                         Url='http://127.0.0.1:5050' },
        @{ Name = 'POC 3b';Path = "$root\\poc3b\\backend";Cmd = 'python app.py';                                         Url='http://127.0.0.1:5051' },
        @{ Name = 'POC 4'; Path = "$root\\poc4";          Cmd = 'python .\\migration_reconciliation.py';                 Url='http://127.0.0.1:5000' },
        @{ Name = 'POC 5'; Path = "$root\\poc5";          Cmd = 'python product_architecture_definition.py';              Url='http://127.0.0.1:6000' }
)

Write-Host 'Starting all POCs...' -ForegroundColor Cyan
foreach ($i in $items) {
        $startCmd = "cd `"$($i.Path)`"; $($i.Cmd)"
        Write-Host ("Launching {0}: {1}" -f $i.Name, $i.Url) -ForegroundColor Yellow
        Start-Process powershell -ArgumentList '-NoExit', '-Command', $startCmd -WindowStyle Normal -WorkingDirectory $i.Path
}

Write-Host 'All POC processes launched.' -ForegroundColor Green
Write-Host ('URLs:' + [Environment]::NewLine + ($items | ForEach-Object { " - {0}: {1}" -f $_.Name, $_.Url }) )
