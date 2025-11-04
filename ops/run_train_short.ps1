. "$PSScriptRoot\env.ps1"
while ($true) {
try {
RunPy "$RepoRoot\rl\train_ppo.py" "--timesteps 500000 --seed 42 --eval_freq 100000"
RunPy "$RepoRoot\rl\evaluate.py" "--max_eval_steps 3000 --out_dir reports_live"
RunPy "$RepoRoot\ops\save_candidate.py"
& $Py "$RepoRoot\ops\promote_challenger.py"
} catch { Write-Host "Short training error: $_" }
Start-Sleep -Seconds 14400
}
