# wormsam2 installer for Windows.  Right-click -> "Run with PowerShell", or in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass; .\install.ps1
# Creates .\.venv with torch (CUDA if an NVIDIA GPU is present, otherwise CPU), SAM2 and the wormsam2 tool,
# downloads the model on first selftest (~180 MB), and leaves wormsam2.cmd next to this file.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host ">> installing uv (python package manager)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Write-Host ">> creating virtual environment (.venv, Python 3.11)..."
uv venv --python 3.11 .venv --allow-existing
$PY = ".venv\Scripts\python.exe"
Write-Host ">> installing torch..."
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "   NVIDIA GPU found -> CUDA build"
    uv pip install --python $PY torch torchvision --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "   no NVIDIA GPU -> CPU build (SAM2-tiny will be used by default)"
    uv pip install --python $PY torch torchvision --index-url https://download.pytorch.org/whl/cpu
}
Write-Host ">> installing SAM2 and wormsam2..."
uv pip install --python $PY -e .
Set-Content -Path "wormsam2.cmd" -Value "@`"%~dp0.venv\Scripts\wormsam2.exe`" %*"
# make 'wormsam2' callable from any folder: add this directory to the user's PATH (takes effect in new terminals)
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$PSScriptRoot*") {
    [Environment]::SetEnvironmentVariable("Path", "$PSScriptRoot;$userPath", "User")
    Write-Host "   added $PSScriptRoot to the user PATH (open a new terminal to use 'wormsam2')"
}
Write-Host ">> self-test (downloads the SAM2 model on first run)..."
& ".\.venv\Scripts\wormsam2.exe" selftest
Write-Host ""
Write-Host "Done. Usage (in a new terminal):  wormsam2 run episodes_<video>.zip"
