# === K-AI Tracker setup ===
# Paste this whole block into the VS Code terminal (PowerShell), once.

$py = (Get-Command python -ErrorAction SilentlyContinue).Source

if ($py -like "*WindowsApps*") {
    Write-Host "PROBLEM: Python points to the Microsoft Store app (breaks venv)." -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix (30 seconds):" -ForegroundColor Yellow
    Write-Host "1. Windows + I -> Apps -> Advanced app settings -> App execution aliases"
    Write-Host "2. Turn off 'python.exe' and 'python3.exe'"
    Write-Host "3. Close the VS Code window completely and reopen it"
    Write-Host "4. Paste this script again"
    exit
}

Write-Host "Python OK: $py" -ForegroundColor Green

# Allow running scripts for this session (safe, user scope only)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

# Create the venv only if it doesn't exist yet
if (-not (Test-Path ".venv")) {
    Write-Host "Creating the virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies (can take a few minutes, torch is large)..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Your prompt should now start with (.venv)"
Write-Host "Next, run: python scripts\run_ingestion.py"
