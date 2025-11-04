. "$PSScriptRoot\env.ps1"
while ($true) { try { RunPy "$RepoRoot\fetch__mt5_data.py" } catch { Write-Host "Collector error: $_" } Start-Sleep -Seconds 300 }
