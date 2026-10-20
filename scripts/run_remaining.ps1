# Remaining Phase 0b measurements that must be taken on AC power. Usage: .\scripts\run_remaining.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
. "$PSScriptRoot\require_ac.ps1"
$uv = "$env:APPDATA\Python\Python314\Scripts\uv.exe"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$tts = "piper:vits-piper-en_US-libritts_r-medium"
$sherpa = "sherpa:sherpa-onnx-streaming-zipformer-en-2023-06-26+hot"
function Conc($name, $stt, $llm) {
    . "$PSScriptRoot\require_ac.ps1"
    Write-Host "=== all-three-at-once $name"
    & $uv run python benchmarks/bench_concurrent.py --stt $stt --llm $llm --tts $tts --stt-threads 3 --llm-threads 10 --tts-threads 2 --seconds 20 --tag "acf_conc_$name"
}
Conc "C1" $sherpa "qwen3:1.7b"
Conc "C2" "fw:tiny.en+hot" "qwen3:1.7b"
Conc "C3" "fw:tiny.en+hot" "qwen3:4b-instruct-2507-q4_K_M"
foreach ($c in @(@("C1", $sherpa), @("C2", "fw:tiny.en+hot"))) {
    . "$PSScriptRoot\require_ac.ps1"
    Write-Host "=== barge-in overlap (STT+TTS, LLM idle) $($c[0])"
    & $uv run python benchmarks/bench_concurrent.py --stt $c[1] --llm qwen3:1.7b --tts $tts --stages stt,tts --stt-threads 4 --tts-threads 4 --seconds 20 --tag "acf_bargein_$($c[0])"
}
. "$PSScriptRoot\require_ac.ps1"
Write-Host "=== OpenVINO CPU / iGPU / NPU"
& $uv run python benchmarks/bench_openvino.py --structured --repeat 5 --tag acf_openvino
. "$PSScriptRoot\require_ac.ps1"
Write-Host "=== throttle check (end)"
& $uv run python benchmarks/bench_throttle.py --seconds 90
Copy-Item benchmarks\results\throttle.json benchmarks\results\acf_throttle_end.json -Force
Write-Host "=== remaining done $(Get-Date -Format HH:mm:ss)"
