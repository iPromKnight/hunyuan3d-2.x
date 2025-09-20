<# 
 .SYNOPSIS
   Setup + run PromGen on Windows (PowerShell), with CUDA or CPU wheels.

 .PARAMETER Cuda
   One of: cpu | cu121 | cu122 | cu124 | cu126 (default: cu126)

 .PARAMETER Python
   Python version to create venv with (default: 3.10)

 .PARAMETER Port
   Gradio/Uvicorn port (default: 8080)

 .PARAMETER Host
   Bind address (default: 0.0.0.0)
#>

[CmdletBinding()]
param(
  [ValidateSet("cpu","cu121","cu122","cu124","cu126")]
  [string]$Cuda = "cu126",
  [string]$Python = "3.10",
  [int]$Port = 8080,
  [string]$Host = "0.0.0.0"
)

$ErrorActionPreference = "Stop"

Write-Host "🪟 Setup PromGen for Windows..."

# 1) Guard: Windows only
if (-not $IsWindows) {
  Write-Host "❌ This script is designed for Windows only"
  exit 1
}

# 2) Require uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "❌ 'uv' command not found. Install UV: https://docs.astral.sh/uv/getting-started/installation"
  exit 1
}

# 3) Ensure a venv exists & is activated
if (-not (Test-Path ".\.venv")) {
  Write-Host "📦 Creating virtual environment..."
  uv venv --python=$Python
}
if (-not $env:VIRTUAL_ENV) {
  Write-Host "✅ Activating virtual environment"
  . .\.venv\Scripts\Activate.ps1
}

# 4) Install deps if missing
$depsOk = $true
try {
  uv run python - <<'PY'
import importlib
for m in ("torch","torchvision","torchaudio","transformers","gradio"):
    importlib.import_module(m)
PY
} catch {
  $depsOk = $false
}

if ($depsOk) {
  Write-Host "✅ Required packages already installed"
} else {
  Write-Host "📦 Installing required packages..."

  # Choose the correct PyTorch wheel index for CPU/CUDA
  $indexUrl = if ($Cuda -eq "cpu") { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/$Cuda" }

  # Pin to your known-good trio (Windows CUDA wheels live on the PyTorch index)
  uv pip install --index-url $indexUrl `
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

  # Then the rest from requirements.txt (Torch also appears there but is already satisfied)
  uv pip install -r requirements.txt

  Write-Host "✅ Required packages installed"
}

# 5) Graceful shutdown helper (PowerShell)
# Ctrl+C normally flows through; this is just a safety net if needed.
$script:child = $null
$onExit = {
  try {
    if ($script:child -and -not $script:child.HasExited) {
      Write-Host "🛑 Stopping PromGen..."
      $script:child.Kill()
    }
  } catch {}
}
Register-EngineEvent PowerShell.Exiting -Action $onExit | Out-Null

# 6) Run the app
Write-Host "🚀 Setup complete - Starting PromGen..."
# (No MPS on Windows; no special env vars needed here)

# Use Start-Process -PassThru to ensure Ctrl+C works cleanly in the same window
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName  = "uv"
$psi.Arguments = "run gradio_app.py --enable_flashvdm --mc_algo=mc --host $Host --port $Port"
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError  = $false
$psi.UseShellExecute = $true
$script:child = [System.Diagnostics.Process]::Start($psi)
$script:child.WaitForExit()
exit $script:child.ExitCode