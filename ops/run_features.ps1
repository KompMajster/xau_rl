. "$PSScriptRoot\env.ps1"
while ($true) { try { RunPy "$RepoRoot\features\build_features.py" } catch { Write-Host "Features error: $_" } Start-Sleep -Seconds 600 }
