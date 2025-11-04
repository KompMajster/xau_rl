. "$PSScriptRoot\env.ps1"
while ($true) { try { RunPy "$RepoRoot\paper_demo\paper_loop_mt5_demo.py" } catch { Write-Host "Paper loop crash: $_"; Start-Sleep -Seconds 10 } Start-Sleep -Seconds 2 }
