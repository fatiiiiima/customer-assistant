# Phase 0b LLM suite: run ONLY when plugged in (AC) with nothing else heavy running.
# Usage: .\scripts\run_llm_suite.ps1 [-Tag ac_best]
param([string]$Tag = "ac")
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$uv = "$env:APPDATA\Python\Python314\Scripts\uv.exe"
$all = @("qwen3:0.6b","qwen3:1.7b","qwen3:4b-instruct-2507-q4_K_M","qwen3.5:0.8b","qwen3.5:2b",
         "ministral-3:3b","llama3.2:1b","llama3.2:3b","gemma3:4b")
$rev = $all[($all.Count-1)..0]
$accState  = @("qwen3:0.6b","qwen3:1.7b","qwen3:4b-instruct-2507-q4_K_M","qwen3.5:2b","ministral-3:3b","llama3.2:3b","gemma3:4b")
$accCached = @("qwen3:0.6b","qwen3:1.7b","qwen3:4b-instruct-2507-q4_K_M","ministral-3:3b")

function Run($label, $argList) {
    Write-Host "=== $label $(Get-Date -Format HH:mm:ss)"
    & $uv run python benchmarks/bench_llm.py @argList
}

# 1-2) speed, per-state prompts, two passes in opposite model order (spreads any drift evenly)
Run "speed state pass1" (@("--models") + $all + @("--mode","state","--repeat","5","--tag","${Tag}_speed_state_p1"))
Run "speed state pass2" (@("--models") + $rev + @("--mode","state","--repeat","5","--tag","${Tag}_speed_state_p2"))
# 3) speed, one static prompt + grammar-only state (always a cache hit)
Run "speed cached" (@("--models") + $all + @("--mode","cached","--repeat","5","--tag","${Tag}_speed_cached"))
# 4-5) accuracy on dev + heldout + fresh sets, every row kept
Run "accuracy state"  (@("--models") + $accState  + @("--mode","state","--accuracy","--no-speed","--cases","all","--tag","${Tag}_acc_state"))
Run "accuracy cached" (@("--models") + $accCached + @("--mode","cached","--accuracy","--no-speed","--cases","all","--tag","${Tag}_acc_cached"))
Write-Host "=== suite done $(Get-Date -Format HH:mm:ss)"
