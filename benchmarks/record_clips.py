"""Recording helper for the STT test clips (docs/RECORDING_GUIDE.md).

Windows only (uses msvcrt for single-key input). Run from the repo root:

    .venv\\Scripts\\python.exe benchmarks\\record_clips.py --help

Modes:
    normal        every line in script.tsv         -> benchmarks/recordings/<id>.wav
    noisy         car* and num* lines only         -> <id>_noisy.wav
    spk2          every line, second speaker       -> <id>_spk2.wav
    quiet         every line, quiet room           -> <id>_quiet.wav   (scored as its own condition)
    headset       every line, headset mic          -> <id>_headset.wav (scored as its own condition)
    raw           every line, Windows enhancements off / WASAPI exclusive -> <id>_raw.wav (own condition)
    spontaneous   your own sentence + typed text   -> benchmarks/spontaneous/spontNNN.wav/.txt
    convert       *.m4a in recordings/ -> WAV (originals kept)

Output is always WAV, mono, 16 kHz, 16-bit PCM. One take per clip: redo is for real
mistakes only (misread line, interruption), and every redo is logged to redo_log.tsv.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
import wave
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REC_DIR = BENCH_DIR / "recordings"
SPONT_DIR = BENCH_DIR / "spontaneous"
SCRIPT_TSV = REC_DIR / "script.tsv"
REDO_LOG = REC_DIR / "redo_log.tsv"

SR = 16000
PREROLL_S = 0.5          # silence kept before the start key (guide: ~0.5 s each end)
TAIL_S = 0.5             # recorded after the stop key
SILENT_PEAK = 0.01       # < -40 dBFS peak: probably the wrong mic / muted
CLIP_SAMPLES = 5         # this many full-scale samples counts as clipped
SHORT_S = 1.5
NOISY_PREFIXES = ("car", "num")
SUFFIXES = {"normal": "", "noisy": "_noisy", "spk2": "_spk2", "quiet": "_quiet", "headset": "_headset", "raw": "_raw"}
CAPTURE_LOG = REC_DIR / "capture_log.tsv"

SPONT_NOTE = """\
SPONTANEOUS TEST SET - HELD OUT.
Never use these files for tuning: no hotwords, no prompt/grammar changes, no model
or threshold selection based on them. They exist only to measure how the system does
on speech nobody optimised for. Audio (*.wav) is gitignored; transcripts are the
reference text exactly as typed.
"""


# ----------------------------------------------------------------------------- audio io
def save_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype("<i2").tobytes())


def check_quality(pcm: np.ndarray) -> list[str]:
    """Return human-readable warnings for a take (empty list = fine)."""
    warns = []
    if pcm.size == 0:
        return ["EMPTY: nothing was recorded"]
    peak = np.abs(pcm.astype(np.int32)).max() / 32768.0
    clipped = int((np.abs(pcm.astype(np.int32)) >= 32767).sum())
    dur = pcm.size / SR
    if peak < SILENT_PEAK:
        warns.append(f"SILENT: peak {20*np.log10(max(peak,1e-6)):.0f} dBFS. Wrong mic or muted?")
    if clipped >= CLIP_SAMPLES:
        warns.append(f"CLIPPED: {clipped} samples at full scale. Lower the input gain or move back.")
    elif peak > 0.98:
        warns.append("Very hot signal (peak near full scale). Consider lowering the input gain.")
    if dur < SHORT_S:
        warns.append(f"SHORT: only {dur:.1f} s. Did you speak the whole line?")
    return warns


def describe(pcm: np.ndarray) -> str:
    if pcm.size == 0:
        return "0.0 s"
    peak = np.abs(pcm.astype(np.int32)).max() / 32768.0
    return f"{pcm.size / SR:.1f} s, peak {20*np.log10(max(peak, 1e-6)):.0f} dBFS"


# --------------------------------------------------------------------------- terminal io
try:
    import msvcrt
except ImportError:  # pragma: no cover
    sys.exit("This helper needs Windows (msvcrt).")


def getkey(timeout: float | None = None) -> str | None:
    """Single key press (lower-cased); None on timeout. Special keys are swallowed."""
    end = None if timeout is None else time.monotonic() + timeout
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()
                continue
            return ch.lower()
        if end is not None and time.monotonic() >= end:
            return None
        time.sleep(0.02)


def flush_keys() -> None:
    while msvcrt.kbhit():
        msvcrt.getwch()


def clear_line() -> None:
    sys.stdout.write("\r" + " " * 78 + "\r")
    sys.stdout.flush()


# ---------------------------------------------------------------------------- recorder
class Recorder:
    """Mic stream that runs for the whole session (cheap) and keeps a short pre-roll."""

    def __init__(self, device: int | str | None, exclusive: bool = False):
        import sounddevice as sd
        from math import gcd

        self.sd = sd
        self.chunks: list[np.ndarray] = []
        self.recording = False
        self.level = 0.0
        self.exclusive = exclusive
        extra = None
        if exclusive:
            # WASAPI exclusive mode bypasses the Windows audio-effects chain (APOs). The device must run at its
            # native rate, so capture there and resample to 16 kHz when the take is finished.
            wasapi = next(i for i, h in enumerate(sd.query_hostapis()) if h["name"] == "Windows WASAPI")
            if device is None or sd.query_devices(device)["hostapi"] != wasapi:
                device = sd.query_hostapis(wasapi)["default_input_device"]
            extra = sd.WasapiSettings(exclusive=True)
        info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
        self.sr = int(info["default_samplerate"]) if exclusive else SR
        self.device_name = info["name"]
        self.api = sd.query_hostapis(info["hostapi"])["name"]
        self._gcd = gcd(SR, self.sr)
        self._pre = np.zeros(0, dtype=np.int16)
        self.stream = sd.InputStream(
            samplerate=self.sr, channels=1, dtype="int16", device=device, extra_settings=extra,
            blocksize=int(self.sr * 0.05), callback=self._cb,
        )

    def banner(self) -> str:
        mode = "WASAPI EXCLUSIVE (Windows audio effects bypassed)" if self.exclusive else "shared mode (Windows audio effects, if enabled, are applied)"
        return f"Input: {self.device_name} | API: {self.api} | {mode} | capture {self.sr} Hz -> saved as 16000 Hz"

    def to_16k(self, pcm: np.ndarray) -> np.ndarray:
        if self.sr == SR or pcm.size == 0:
            return pcm
        from scipy.signal import resample_poly
        out = resample_poly(pcm.astype(np.float32), SR // self._gcd, self.sr // self._gcd)
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)

    def _cb(self, indata, frames, time_info, status):
        block = indata[:, 0].copy()
        self.level = float(np.abs(block.astype(np.int32)).max()) / 32768.0
        if self.recording:
            self.chunks.append(block)
        else:
            keep = int(self.sr * PREROLL_S)
            self._pre = np.concatenate([self._pre, block])[-keep:]

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    def begin(self) -> None:
        self.chunks = [self._pre.copy()]
        self.recording = True

    def end(self) -> np.ndarray:
        time.sleep(TAIL_S)
        self.recording = False
        return self.to_16k(np.concatenate(self.chunks)) if self.chunks else np.zeros(0, dtype=np.int16)

    def cancel(self) -> None:
        self.recording = False
        self.chunks = []

    def take(self, prompt: str) -> np.ndarray | None:
        """Wait for SPACE, record with a live meter until SPACE. None if cancelled (ESC) or quit."""
        print(f"  {prompt}")
        flush_keys()
        while True:
            self._idle_meter()
            k = getkey(0.05)
            if k == " ":
                break
            if k in ("q", "\x1b"):
                clear_line()
                return None
        self.begin()
        t0 = time.monotonic()
        flush_keys()
        while True:
            self._rec_meter(time.monotonic() - t0)
            k = getkey(0.05)
            if k == " ":
                break
            if k == "\x1b":
                self.cancel()
                clear_line()
                print("  (take cancelled)")
                return np.zeros(0, dtype=np.int16)
        pcm = self.end()
        clear_line()
        return pcm

    @staticmethod
    def _bar(level: float, width: int = 30) -> str:
        db = 20 * np.log10(max(level, 1e-5))
        n = int(np.clip((db + 60) / 60, 0, 1) * width)
        return "#" * n + "." * (width - n)

    def _idle_meter(self):
        sys.stdout.write(f"\r  [ready ] SPACE=start  Q=quit   mic |{self._bar(self.level)}|   ")
        sys.stdout.flush()

    def _rec_meter(self, secs: float):
        blink = "*" if int(secs * 2) % 2 == 0 else " "
        sys.stdout.write(
            f"\r  [{blink}REC {secs:4.1f}s] SPACE=stop  ESC=cancel   |{self._bar(self.level)}|   "
        )
        sys.stdout.flush()


def review(rec: Recorder, pcm: np.ndarray, allow_redo: bool = True) -> str:
    """After a take: returns 'keep', 'redo' or 'skip'."""
    print(f"  Recorded {describe(pcm)}")
    for w in check_quality(pcm):
        print(f"  !! {w}")
    opts = "ENTER=keep  P=play back  S=skip clip"
    if allow_redo:
        opts += "  R=redo (real mistake only)"
    print(f"  {opts}")
    while True:
        k = getkey()
        if k in ("\r", "\n"):
            return "keep"
        if k == "p":
            rec.sd.play(pcm, SR, blocking=True)
        elif k == "s":
            return "skip"
        elif k == "r" and allow_redo:
            return "redo"


# ------------------------------------------------------------------------------ modes
def load_script() -> list[tuple[str, str]]:
    rows = []
    with SCRIPT_TSV.open(encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for r in rd:
            if len(r) >= 2 and r[0].strip():
                rows.append((r[0].strip(), r[1].strip()))
    return rows


def log_capture(rec: "Recorder", path: Path, mode: str) -> None:
    """One provenance row per saved clip: which device/API/mode produced it."""
    new = not CAPTURE_LOG.exists()
    with CAPTURE_LOG.open("a", encoding="utf-8") as f:
        if new:
            f.write("timestamp\tfile\tmode\tdevice\tapi\texclusive\tcapture_hz\n")
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{path.name}\t{mode}\t{rec.device_name}\t{rec.api}\t"
                f"{rec.exclusive}\t{rec.sr}\n")


def log_redo(name: str) -> None:
    new = not REDO_LOG.exists()
    with REDO_LOG.open("a", encoding="utf-8") as f:
        if new:
            f.write("timestamp\tfile\n")
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{name}\n")


def run_clips(args) -> None:
    suffix = SUFFIXES[args.mode]
    rows = load_script()
    if args.mode == "noisy":
        rows = [r for r in rows if r[0].startswith(NOISY_PREFIXES)]
    if args.ids:
        want = {i.strip() for i in args.ids.split(",")}
        rows = [r for r in rows if r[0] in want]
    if not rows:
        sys.exit("No lines selected.")

    todo = []
    for cid, text in rows:
        if (REC_DIR / f"{cid}{suffix}.wav").exists() and args.resume:
            continue
        todo.append((cid, text))
    done = len(rows) - len(todo)
    print(f"\nMode: {args.mode}  | {len(rows)} lines, {done} already recorded, {len(todo)} to go.")
    if args.mode == "noisy":
        print("Keep a fan or a TV playing at low volume the whole time.")
    if args.mode == "spk2":
        print("This session is for the SECOND speaker.")
    print("Read each line exactly as written, in your normal voice. One take each.\n")
    if not todo:
        print("Nothing to do.")
        return

    with Recorder(args.device, args.exclusive) as rec:
        print(rec.banner() + "\n")
        for i, (cid, text) in enumerate(todo, 1):
            path = REC_DIR / f"{cid}{suffix}.wav"
            print(f"[{i}/{len(todo)}] {path.name}")
            print(f'      "{text}"\n')
            if path.exists():
                print(f"  {path.name} already exists. O=overwrite  any other key=skip")
                if getkey() != "o":
                    print()
                    continue
            while True:
                pcm = rec.take("Press SPACE, then read the line.")
                if pcm is None:
                    print("\nStopped. Re-run with --resume to continue where you left off.")
                    return
                if pcm.size == 0:
                    continue
                verdict = review(rec, pcm)
                if verdict == "redo":
                    log_redo(path.name)
                    continue
                if verdict == "keep":
                    save_wav(path, pcm)
                    log_capture(rec, path, args.mode)
                    print(f"  saved {path.relative_to(BENCH_DIR.parent)}\n")
                else:
                    print("  skipped\n")
                break
    print("All selected clips done.")


def run_spontaneous(args) -> None:
    SPONT_DIR.mkdir(parents=True, exist_ok=True)
    (SPONT_DIR / "README.txt").write_text(SPONT_NOTE, encoding="utf-8")
    gi = SPONT_DIR / ".gitignore"
    if not gi.exists():
        gi.write_text("*.wav\n", encoding="utf-8")

    def next_id() -> str:
        n = 1
        while (SPONT_DIR / f"spont{n:03d}.wav").exists() or (SPONT_DIR / f"spont{n:03d}.txt").exists():
            n += 1
        return f"spont{n:03d}"

    print("\nSpontaneous mode: say a sentence of your own, then type exactly what you said.")
    print("These are a held-out set: never use them for tuning.\n")
    with Recorder(args.device, args.exclusive) as rec:
        print(rec.banner() + "\n")
        while True:
            sid = next_id()
            print(f"[{sid}]")
            pcm = rec.take("Press SPACE, say your sentence, SPACE to stop. (Q to finish)")
            if pcm is None:
                break
            if pcm.size == 0:
                continue
            verdict = review(rec, pcm)
            if verdict == "redo":
                log_redo(f"spontaneous/{sid}.wav")
                continue
            if verdict == "skip":
                continue
            flush_keys()
            text = ""
            while not text:
                text = input("  Type what you said (or /discard): ").strip()
            if text == "/discard":
                print("  discarded\n")
                continue
            save_wav(SPONT_DIR / f"{sid}.wav", pcm)
            log_capture(rec, SPONT_DIR / f"{sid}.wav", "spontaneous")
            (SPONT_DIR / f"{sid}.txt").write_text(text + "\n", encoding="utf-8")
            print(f"  saved {sid}.wav + {sid}.txt\n")
    print("Done.")


def run_convert(args) -> None:
    import av

    known = {cid for cid, _ in load_script()}
    files = sorted(REC_DIR.glob("*.m4a"))
    if not files:
        print(f"No .m4a files in {REC_DIR}. Drop them there, named by id (car01.m4a).")
        return
    n_ok = 0
    for src in files:
        base = src.stem
        for suf in ("_noisy", "_spk2", "_quiet", "_headset", "_raw"):
            if base.endswith(suf):
                base = base[: -len(suf)]
        if base not in known:
            print(f"  ?? {src.name}: '{base}' is not an id in script.tsv (converting anyway)")
        dst = src.with_suffix(".wav")
        if dst.exists() and not args.force:
            print(f"  skip {src.name}: {dst.name} exists (use --force to overwrite)")
            continue
        try:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
            parts = []
            with av.open(str(src)) as container:
                stream = container.streams.audio[0]
                for frame in container.decode(stream):
                    for out in resampler.resample(frame):
                        parts.append(out.to_ndarray().reshape(-1))
                for out in resampler.resample(None):
                    parts.append(out.to_ndarray().reshape(-1))
            pcm = np.concatenate(parts).astype(np.int16) if parts else np.zeros(0, np.int16)
        except Exception as e:  # noqa: BLE001 - report and continue with the next file
            print(f"  FAILED {src.name}: {e}")
            continue
        save_wav(dst, pcm)
        n_ok += 1
        print(f"  {src.name} -> {dst.name}  ({describe(pcm)})")
        for w in check_quality(pcm):
            print(f"     !! {w}")
    print(f"Converted {n_ok} file(s). Originals kept.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="normal",
                    choices=["normal", "noisy", "spk2", "quiet", "headset", "raw", "spontaneous", "convert"])
    ap.add_argument("--resume", action="store_true",
                    help="skip clips whose WAV already exists (continue where you stopped)")
    ap.add_argument("--ids", help="comma-separated ids to record, e.g. car03,num05")
    ap.add_argument("--device", help="input device index or name (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--exclusive", action="store_true",
                    help="WASAPI exclusive mode: bypasses Windows audio effects (default for the raw mode)")
    ap.add_argument("--shared", action="store_true", help="raw mode only: use shared mode (you turned enhancements off in Windows)")
    ap.add_argument("--force", action="store_true", help="convert: overwrite existing WAVs")
    args = ap.parse_args()

    args.exclusive = (args.exclusive or args.mode == "raw") and not args.shared
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.mode == "convert":
        run_convert(args)
    elif args.mode == "spontaneous":
        run_spontaneous(args)
    else:
        run_clips(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Saved clips are kept; use --resume to continue.")
