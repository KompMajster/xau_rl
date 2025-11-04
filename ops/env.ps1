$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path
$Py = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Cfg = Join-Path $RepoRoot "config.yaml"
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
function RunPy([string]$script, [string]$args="") {
& $Py $script $args 2>&1 | Tee-Object -FilePath (Join-Path $LogDir ((Split-Path $script -Leaf) + ".log")) -Append
}
