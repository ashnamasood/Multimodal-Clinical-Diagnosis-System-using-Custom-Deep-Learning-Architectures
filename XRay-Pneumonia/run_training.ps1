# Overnight chest X-ray training (CPU-friendly, logs to outputs/xray/training.log)
# Run from project root:  powershell -ExecutionPolicy Bypass -File XRay-Pneumonia/run_training.ps1

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "XRay-Pneumonia\outputs\xray"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "training.log"

function Write-Log($Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Write-Log "=== X-ray training started ==="
Write-Log "Project: $ProjectRoot"
Write-Log "Device: CPU (128px, batch 64 — tuned for overnight completion on CPU)"

$Args = @(
    "-u",
    "XRay-Pneumonia/train_xray_cnn.py",
    "--epochs", "20",
    "--image-size", "128",
    "--batch-size", "64",
    "--early-stopping-patience", "5",
    "--gradcam-samples", "6"
)

try {
    & python @Args 2>&1 | ForEach-Object {
        Write-Host $_
        Add-Content -Path $LogFile -Value $_
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Log "=== Training finished successfully (exit 0) ==="
    } else {
        Write-Log "=== Training failed (exit $LASTEXITCODE) ==="
        exit $LASTEXITCODE
    }
} catch {
    Write-Log "=== Training error: $_ ==="
    exit 1
}

# Post-train smoke test
Write-Log "Running smoke test..."
& python "XRay-Pneumonia/test_model.py" --samples-per-class 5 2>&1 | ForEach-Object {
    Write-Host $_
    Add-Content -Path $LogFile -Value $_
}
Write-Log "=== All done ==="
