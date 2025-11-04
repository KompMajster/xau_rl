$nssm = "C:\nssm\nssm.exe"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ps = "powershell.exe"
& $nssm install XAU_Collector   $ps "-ExecutionPolicy Bypass -File "$root\run_collector.ps1""
& $nssm set    XAU_Collector   AppDirectory (Split-Path -Parent $root)
& $nssm set    XAU_Collector   Start SERVICE_AUTO_START
& $nssm install XAU_Features    $ps "-ExecutionPolicy Bypass -File "$root\run_features.ps1""
& $nssm set    XAU_Features    AppDirectory (Split-Path -Parent $root)
& $nssm set    XAU_Features    Start SERVICE_AUTO_START
& $nssm install XAU_Paper       $ps "-ExecutionPolicy Bypass -File "$root\run_paper.ps1""
& $nssm set    XAU_Paper       AppDirectory (Split-Path -Parent $root)
& $nssm set    XAU_Paper       Start SERVICE_AUTO_START
& $nssm install XAU_TrainShort  $ps "-ExecutionPolicy Bypass -File "$root\run_train_short.ps1""
& $nssm set    XAU_TrainShort  AppDirectory (Split-Path -Parent $root)
& $nssm set    XAU_TrainShort  Start SERVICE_AUTO_START
& $nssm install XAU_TrainNightly $ps "-ExecutionPolicy Bypass -File "$root\run_train_nightly.ps1""
& $nssm set    XAU_TrainNightly AppDirectory (Split-Path -Parent $root)
& $nssm set    XAU_TrainNightly Start SERVICE_AUTO_START
