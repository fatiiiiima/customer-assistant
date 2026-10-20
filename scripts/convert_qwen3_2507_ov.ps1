# Reproducible conversion of Qwen/Qwen3-4B-Instruct-2507 to OpenVINO int4, using the SAME recipe the OpenVINO org
# documents for OpenVINO/Qwen3-4B-int4-ov (INT4_SYM, group 128, ratio 1.0, AWQ + scale estimation, wikitext2).
# Tools live in an ISOLATED env outside the repo (torch + transformers are large). Every tool is Apache-2.0/BSD-3/MIT.
# Writes timings to logs\convert_timings.txt. If any download fails, it stops and prints the failing step.
$ErrorActionPreference = "Continue"   # native tools print progress to stderr; failures are caught via $LASTEXITCODE below
Set-Location (Split-Path $PSScriptRoot -Parent)
$uv = "$env:APPDATA\Python\Python314\Scripts\uv.exe"
$env_dir = "C:\dev\ov-convert"
$t = "logs\convert_timings.txt"; "" | Set-Content $t
function Step($name, [scriptblock]$b) {
    Write-Host "=== $name $(Get-Date -Format HH:mm:ss)"
    $d = Measure-Command { & $b }
    if ($LASTEXITCODE -ne 0) { "$name : FAILED (exit $LASTEXITCODE)" | Add-Content $t; throw "step failed: $name" }
    "$name : $([math]::Round($d.TotalMinutes,1)) min" | Add-Content $t
}
if (-not (Test-Path "$env_dir\.venv")) { Step "create env"  { & $uv venv "$env_dir\.venv" --python 3.12 } }
Step "install optimum-intel[openvino] (idempotent)" { & $uv pip install --python "$env_dir\.venv\Scripts\python.exe" "optimum-intel[openvino]" }
& $uv pip freeze --python "$env_dir\.venv\Scripts\python.exe" | Select-String '^(optimum|optimum-intel|transformers|torch|nncf|openvino|datasets)==' | Set-Content logs\convert_versions.txt
$cli = "$env_dir\.venv\Scripts\optimum-cli.exe"
Step "export data-free int4 (no calibration)" {
    & $cli export openvino -m Qwen/Qwen3-4B-Instruct-2507 --weight-format int4 --sym --group-size 128 --ratio 1.0 --task text-generation-with-past "models\ov\Qwen3-4B-Instruct-2507-int4-ov-nocal"
}
Step "export AWQ + scale-estimation int4 (org recipe)" {
    & $cli export openvino -m Qwen/Qwen3-4B-Instruct-2507 --weight-format int4 --sym --group-size 128 --ratio 1.0 --awq --scale-estimation --dataset wikitext2 --task text-generation-with-past "models\ov\Qwen3-4B-Instruct-2507-int4-ov"
}
Write-Host "=== convert done $(Get-Date -Format HH:mm:ss)"
