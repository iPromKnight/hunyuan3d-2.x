<# 
 .SYNOPSIS
   Setup + run PromGen on Windows (PowerShell), with CUDA or CPU wheels.

 .PARAMETER Cuda
   One of: cpu | cu121 | cu122 | cu124 | cu126 (default: cu126)

 .PARAMETER Python
   Python version to create venv with (default: 3.10)

 .PARAMETER Port
   Gradio/Uvicorn port (default: 8080)

 .PARAMETER ServerHost
   Bind address (default: 0.0.0.0)
#>

[CmdletBinding()]
param(
  [ValidateSet("cpu","cu121","cu122","cu124","cu126")]
  [string]$Cuda = "cu126",
  [string]$Python = "3.10",
  [int]$Port = 8080,
  [string]$ServerHost = "0.0.0.0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ensure relative paths resolve correctly
if ($PSScriptRoot) { Set-Location $PSScriptRoot }

Write-Host "🪟 Setup PromGen for Windows..."
Write-Host "• Cuda: $Cuda | Python: $Python | Host: $ServerHost | Port: $Port"

# 1) Guard: Windows only
$onWindows = $false

if ($PSVersionTable.PSEdition -eq 'Desktop') {
    # Windows PowerShell 5.x
    if ($env:OS -eq 'Windows_NT') { $onWindows = $true }
}
elseif ($PSVersionTable.PSEdition -eq 'Core') {
    # PowerShell 6/7+
    if ($IsWindows) { $onWindows = $true }
}

if (-not $onWindows) {
    Write-Host "❌ This script is designed for Windows only"
    return
}

# 2) Require uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "❌ 'uv' command not found. Install UV: https://docs.astral.sh/uv/getting-started/installation"
  return
}

# 3) Ensure a venv exists
if (-not (Test-Path ".\.venv")) {
  Write-Host "[..] Creating virtual environment..."
  uv venv --python=$Python
}

# Path to the venv's Python
$VenvPython = Join-Path (Join-Path $PSScriptRoot ".venv") "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
  throw "Virtual environment Python not found at: $VenvPython"
}

# 4) Install deps if missing
$depsCode = @"
import importlib, sys
mods = ("torch","torchvision","torchaudio","transformers","gradio")
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    sys.exit(1)
"@

$depsOk = $true
& uv run --python "$VenvPython" python -c $depsCode | Out-Null
if ($LASTEXITCODE -ne 0) { $depsOk = $false }

if ($depsOk) {
  Write-Host "[OK] Required packages already installed"
} else {
  Write-Host "[..] Installing required packages..."

  # Choose the correct PyTorch wheel index for CPU/CUDA
  $indexUrl = if ($Cuda -eq "cpu") { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/$Cuda" }

  # Install pinned torch trio
  & uv pip install --python "$VenvPython" --index-url $indexUrl `
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

  # Then install the rest
  & uv pip install --python "$VenvPython" -r (Join-Path $PSScriptRoot "requirements.txt")

  Write-Host "[OK] Required packages installed"
}

# 5) Run the app in foreground so Ctrl+C works
Write-Host "🚀 Setup complete - Starting PromGen..."
& uv run --python "$VenvPython" `
  python gradio_app.py `
  --enable_flashvdm `
  --mc_algo=mc `
  --host $ServerHost `
  --port $Port

$exitCode = $LASTEXITCODE
exit $exitCode
