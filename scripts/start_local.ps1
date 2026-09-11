$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$vitePath = Join-Path $projectRoot 'frontend\node_modules\vite\bin\vite.js'
$logDirectory = Join-Path $projectRoot '.tmp'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv and install requirements.txt first.' }
if (-not (Test-Path -LiteralPath $vitePath)) { throw 'Run npm ci in frontend first.' }
foreach ($port in @(8000, 5173)) {
    if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. Stop the existing service or start manually on another port."
    }
}
$apiProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory 'backend.log') -RedirectStandardError (Join-Path $logDirectory 'backend-error.log') -PassThru
$nodePath = (Get-Command node).Source
$uiProcess = Start-Process -FilePath $nodePath -ArgumentList @($vitePath,'--host','127.0.0.1','--port','5173') -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory 'frontend.log') -RedirectStandardError (Join-Path $logDirectory 'frontend-error.log') -PassThru
@{backend=$apiProcess.Id; frontend=$uiProcess.Id} | ConvertTo-Json | Set-Content (Join-Path $logDirectory 'local-processes.json')
Write-Output "Started backend PID $($apiProcess.Id), frontend PID $($uiProcess.Id)."
Write-Output 'Frontend: http://localhost:5173 | API: http://localhost:8000/docs'
Write-Output 'Logs and process IDs are in .tmp. Stop these specific processes when finished.'
