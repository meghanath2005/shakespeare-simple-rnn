$projectRoot = $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Create .venv and install requirements first. See README.md.' }
$basePrefix = & $projectPython -c 'import sys; print(sys.base_prefix)'
$env:JUPYTER_PATH = (Join-Path $projectRoot '.venv\share\jupyter') + ';' + (Join-Path $basePrefix 'share\jupyter')
$env:JUPYTER_RUNTIME_DIR = Join-Path $projectRoot '.jupyter_runtime'
$localLab = Join-Path $projectRoot '.venv\share\jupyter\lab'
$baseLab = Join-Path $basePrefix 'share\jupyter\lab'
if (Test-Path -LiteralPath $localLab) { $env:JUPYTERLAB_DIR = $localLab }
elseif (Test-Path -LiteralPath $baseLab) { $env:JUPYTERLAB_DIR = $baseLab }
Set-Location -LiteralPath $projectRoot
& $projectPython -m notebook 'Shakespeare_SimpleRNN.ipynb'
