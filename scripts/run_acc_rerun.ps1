param([string]$Tag = "ac2")
Set-Location (Split-Path $PSScriptRoot -Parent)
$uv = "$env:APPDATA\Python\Python314\Scripts\uv.exe"
$m = @("qwen3:0.6b","qwen3:1.7b","qwen3:4b-instruct-2507-q4_K_M","ministral-3:3b")
Write-Host "=== acc state $(Get-Date -Format HH:mm:ss)"
& $uv run python benchmarks/bench_llm.py --models @m --mode state --accuracy --no-speed --cases all --tag "${Tag}_acc_state"
Write-Host "=== acc cached $(Get-Date -Format HH:mm:ss)"
& $uv run python benchmarks/bench_llm.py --models @m --mode cached --accuracy --no-speed --cases all --tag "${Tag}_acc_cached"
Write-Host "=== rerun done $(Get-Date -Format HH:mm:ss)"
