param(
    [string]$EnvName = "biocore-omicverse",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found. Install Miniconda or Mambaforge first."
}

conda create -n $EnvName "python=$PythonVersion" -y
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install omicverse
conda run -n $EnvName python -c "import omicverse as ov; print('omicverse ok')"

$pythonPath = (conda run -n $EnvName python -c "import sys; print(sys.executable)").Trim()
Write-Host ""
Write-Host "Add these lines to your .env:"
Write-Host "SCICORE_OMICVERSE_ENABLED=1"
Write-Host "SCICORE_OMICVERSE_PYTHON=$pythonPath"
