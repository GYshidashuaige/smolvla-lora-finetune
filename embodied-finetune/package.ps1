$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& .venv/Scripts/python.exe report.py
if ($LASTEXITCODE) { throw 'Report generation failed' }
$files = @('README.md','RESULTS.md','setup.ps1','train.py','train_entry.py','download.py','fetch_large.py','report.py','evaluate.py','export_adapter.py','package.ps1','.gitignore','results')
if (Test-Path 'artifacts') { $files += 'artifacts' }
if (Test-Path 'logs') { $files += 'logs' }
if (Test-Path 'environment-lock.txt') { $files += 'environment-lock.txt' }
Compress-Archive -LiteralPath $files -DestinationPath 'embodied-finetune-submission.zip' -Force
Write-Output 'Packaged code, evidence, and exported artifacts/adapter when available. Full optimizer checkpoints remain in outputs/.'
