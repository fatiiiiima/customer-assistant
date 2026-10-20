# Pull the Ollama models used in the Phase 0b spike. Run: .\scripts\pull_models.ps1
# If a pull fails (network restriction), the script prints the failing model and stops.
# Do not work around it: download manually per README "Manual model download".

param([switch]$IncludeUnverified, [switch]$IncludeBenchOnly)

$ErrorActionPreference = "Stop"

# Production-candidate family (Apache-2.0). Thinking is disabled at call time (think=false).
$apache = @(
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen3:4b-instruct-2507-q4_K_M",   # non-thinking instruct variant
    "ministral-3:3b"                   # Apache-2.0 per its Ollama page
)

# Newer Qwen generation. LICENSE NOT YET VERIFIED: check the HF model card before use.
$unverified = @("qwen3.5:0.8b", "qwen3.5:2b")

# BENCHMARK ONLY: custom (non-MIT/Apache) licenses. Never the production default.
$benchOnly = @("llama3.2:1b", "llama3.2:3b", "gemma3:4b")

$list = @($apache)
if ($IncludeUnverified) { $list += $unverified }
if ($IncludeBenchOnly)  { $list += $benchOnly }

foreach ($m in $list) {
    Write-Host "Pulling $m ..."
    & ollama pull $m
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED to pull '$m'. Network restriction? See README: Manual model download."
    }
}
Write-Host "Done. Installed models:"
& ollama list
