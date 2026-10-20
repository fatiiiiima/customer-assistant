# Interleaved repeats of the sequential-turn benchmark (A B C A B C ...) to expose run-to-run drift. AC only.
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\require_ac.ps1"
$uv = "$env:APPDATA\Python\Python314\Scripts\uv.exe"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$stt = "ovw:whisper-tiny-int8-ov@CPU+hot"
$cfgs = @(
  @{ name = "P_4B_igpu_state";   llm = "ov:Qwen3-4B-int4-ov@GPU";   mode = "state"  },
  @{ name = "F_1p7B_cpu_static"; llm = "ov:Qwen3-1.7B-int4-ov@CPU"; mode = "static" },
  @{ name = "Q_1p7B_igpu_state"; llm = "ov:Qwen3-1.7B-int4-ov@GPU"; mode = "state"  }
)
foreach ($rep in 1..3) {
  foreach ($c in $cfgs) {
    . "$PSScriptRoot\require_ac.ps1"
    Write-Host "=== rep $rep $($c.name) $(Get-Date -Format HH:mm:ss)"
    & $uv run python benchmarks/bench_turns.py --stt $stt --llm $c.llm --prompt-mode $c.mode --tag "reps_$($c.name)_$rep"
  }
}
Write-Host "=== reps done $(Get-Date -Format HH:mm:ss)"
