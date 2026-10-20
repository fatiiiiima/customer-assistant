# Download STT/TTS model files for the Phase 0b spike into models\<name>\ (gitignored).
# Each archive is extracted into its own directory. Models are data only; nothing is executed.
# faster-whisper and moonshine models download themselves from Hugging Face on first use.
# If any URL fails, the script stops and prints it: see README "Manual model download".
$ErrorActionPreference = "Stop"
$root = Join-Path (Split-Path $PSScriptRoot -Parent) "models"
New-Item -ItemType Directory -Force $root | Out-Null

$gh = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
$archives = @(
    @{ tag = "asr-models"; file = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17" },
    @{ tag = "asr-models"; file = "sherpa-onnx-streaming-zipformer-en-2023-06-26" },
    @{ tag = "tts-models"; file = "vits-piper-en_US-lessac-medium" },
    @{ tag = "tts-models"; file = "vits-piper-en_US-lessac-medium-int8" },
    @{ tag = "tts-models"; file = "vits-piper-en_US-amy-low-int8" }
)
foreach ($a in $archives) {
    $dest = Join-Path $root $a.file
    if (Test-Path $dest) { Write-Host "have $($a.file)"; continue }
    $url = "$gh/$($a.tag)/$($a.file).tar.bz2"
    $tmp = Join-Path $root "$($a.file).tar.bz2"
    Write-Host "GET $url"
    curl.exe -fL --retry 3 -o $tmp $url; if ($LASTEXITCODE -ne 0) { throw "FAILED URL: $url" }
    New-Item -ItemType Directory -Force $dest | Out-Null
    tar -xjf $tmp -C $dest
    Remove-Item $tmp
}

# Kokoro (kokoro-onnx files; model Apache-2.0)
$k = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
$kd = Join-Path $root "kokoro-v1.0"
New-Item -ItemType Directory -Force $kd | Out-Null
foreach ($f in "kokoro-v1.0.int8.onnx", "kokoro-v1.0.onnx", "voices-v1.0.bin") {
    $out = Join-Path $kd $f
    if (Test-Path $out) { Write-Host "have $f"; continue }
    Write-Host "GET $k/$f"
    curl.exe -fL --retry 3 -o $out "$k/$f"; if ($LASTEXITCODE -ne 0) { throw "FAILED URL: $k/$f" }
}
Write-Host "Done."

