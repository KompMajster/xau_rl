. "$PSScriptRoot\env.ps1"
while ($true) {
try {
while ( (Get-Date).ToUniversalTime().Hour -ne 23 ) { Start-Sleep -Seconds 600 }
Start-Sleep -Seconds 300
RunPy "$RepoRoot\rl\train_ppo.py" "--timesteps 2000000 --seed 42 --eval_freq 200000"
RunPy "$RepoRoot\rl\walk_forward.py" "--segments 6 --train_days 120 --val_days 30 --timesteps 500000 --out_dir reports_wf"
RunPy "$RepoRoot\rl\evaluate.py" "--max_eval_steps 3000 --out_dir reports_nightly"
RunPy "$RepoRoot\ops\save_candidate.py"
& $Py "$RepoRoot\ops\promote_challenger.py"
} catch { Write-Host "Nightly training error: $_" }
Start-Sleep -Seconds 3600
}
