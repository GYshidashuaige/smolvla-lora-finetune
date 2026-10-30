$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$proxySettings = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings'
if ($proxySettings.ProxyEnable -eq 1 -and $proxySettings.ProxyServer -match '^127\.0\.0\.1:\d+$') {
    $env:HTTP_PROXY = 'http://' + $proxySettings.ProxyServer
    $env:HTTPS_PROXY = $env:HTTP_PROXY
}
if (!(Test-Path '.venv/Scripts/python.exe')) { py -3.11 -m venv .venv }
if (!(Test-Path 'vendor/lerobot/pyproject.toml')) {
    git clone --depth 1 --branch v0.4.3 https://github.com/huggingface/lerobot.git vendor/lerobot
    if ($LASTEXITCODE) { throw 'Clone failed' }
}
$torchWheel = 'cache/wheels/torch-2.7.1+cu128-cp311-cp311-win_amd64.whl'
if (Test-Path $torchWheel) {
    & .venv/Scripts/python.exe -m pip install $torchWheel torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
} else {
    & .venv/Scripts/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
}
if ($LASTEXITCODE) { throw 'CUDA PyTorch install failed' }
& .venv/Scripts/python.exe -m pip install -e './vendor/lerobot[smolvla,peft]' 'transformers==4.57.1' 'numpy<2.3'
if ($LASTEXITCODE) { throw 'LeRobot install failed' }
& .venv/Scripts/python.exe -c "import torch; print(torch.__version__,torch.cuda.is_available()); assert torch.cuda.is_available()"
if ($LASTEXITCODE) { throw 'CUDA validation failed' }
