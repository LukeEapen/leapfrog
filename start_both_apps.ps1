# Combined Flask Applications Launcher (PowerShell)
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  COMBINED FLASK APPLICATIONS LAUNCHER" -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting both Flask applications..." -ForegroundColor Green
Write-Host ""
Write-Host "Optimized Backend: http://localhost:5000" -ForegroundColor Blue
Write-Host "Three-Section UI:  http://localhost:5001/three-section" -ForegroundColor Blue
Write-Host ""
Write-Host "Each application will open in a separate window..." -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Cyan

try {
    # Start the optimized backend
    Write-Host "Starting Optimized Backend (Port 5000)..." -ForegroundColor Green
    $optimizedProcess = Start-Process python -ArgumentList "poc2_backend_processor_optimized.py" -PassThru -WindowStyle Normal
    
    # Wait a moment before starting the second app
    Start-Sleep -Seconds 2
    
    # Start the three-section backend
    Write-Host "Starting Three-Section Backend (Port 5001)..." -ForegroundColor Green  
    $threeSectionProcess = Start-Process python -ArgumentList "launch_three_section.py" -PassThru -WindowStyle Normal
    
    Write-Host ""
    Write-Host "✅ Both applications started successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🔗 Access URLs:" -ForegroundColor Yellow
    Write-Host "   • Optimized Backend: http://localhost:5000" -ForegroundColor Blue
    Write-Host "   • Three-Section UI: http://localhost:5001/three-section" -ForegroundColor Blue
    Write-Host ""
    Write-Host "💡 Both applications are running in separate windows." -ForegroundColor Cyan
    Write-Host "   Close those windows or press Ctrl+C in them to stop the applications." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Process IDs:" -ForegroundColor Yellow
    Write-Host "   Optimized Backend: $($optimizedProcess.Id)" -ForegroundColor White
    Write-Host "   Three-Section Backend: $($threeSectionProcess.Id)" -ForegroundColor White
    
} catch {
    Write-Host "❌ Error starting applications: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Read-Host "Press Enter to continue (this will not stop the applications)"
