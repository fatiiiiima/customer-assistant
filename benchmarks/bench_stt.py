"""STT benchmark: latency on a ~3 s clip and accuracy on car names / numbers / plates.

Latency definitions (what the caller waits for after they stop talking):
  streaming engines (sherpa-onnx): 'tail' = time to process the LAST audio chunk plus
      end-of-input flush. Earlier chunks are decoded while the caller is still speaking.
  offline engines (Moonshine, Whisper): 'decode' = whole-clip decode time after endpointing.
Accuracy clips: real recordings in benchmarks/recordings/*.wav if present, otherwise the
SYNTHETIC clips from make_synth_clips.py (optimistic; flagged in the output).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import HOTWORDS, MODELS, RECORDINGS, RESULTS, hotword_terms, load_wav_16k, now, save_result, summarize  # noqa: E402
from stt_eval import canon_numbers, term_recall, wer  # noqa: E402

CHUNK = 1600  # 100 ms at 16 kHz


class Engine:
    name: str
    streaming = False

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Return (text, latency_ms) where latency is as defined in the module docstring."""
        raise NotImplementedError


class SherpaStreaming(Engine):
    streaming = True

    def __init__(self, dirname: str, hotwords: bool, threads: int) -> None:
        import sherpa_onnx

        d = next(p for p in (MODELS / dirname).iterdir() if p.is_dir())
        self.name = f"sherpa:{dirname.replace('sherpa-onnx-streaming-zipformer-', '')}" + ("+hotwords" if hotwords else "")
        enc = sorted(d.glob("encoder*int8.onnx")) or sorted(d.glob("encoder*.onnx"))
        dec = sorted(d.glob("decoder*int8.onnx")) or sorted(d.glob("decoder*.onnx"))
        joi = sorted(d.glob("joiner*int8.onnx")) or sorted(d.glob("joiner*.onnx"))
        kw: dict = {}
        if hotwords:
            bpe = d / "bpe.model"
            if not bpe.exists():
                # Passing an empty bpe_vocab aborts natively (uncatchable), so refuse up front.
                raise ValueError("this model has no bpe.model; sherpa hotword biasing needs a BPE model")
            vocab = d / "bpe.vocab"
            if not vocab.exists():  # sherpa wants "piece<TAB>score" lines derived from bpe.model
                import sentencepiece as spm

                sp = spm.SentencePieceProcessor(model_file=str(bpe))
                vocab.write_text("\n".join(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}"
                                           for i in range(sp.get_piece_size())) + "\n", encoding="utf-8")
            hw = d / "hotwords_cars.txt"
            hw.write_text("\n".join(t.upper() for t in hotword_terms()) + "\n", encoding="utf-8")
            kw = dict(decoding_method="modified_beam_search", hotwords_file=str(hw), hotwords_score=2.0,
                      modeling_unit="bpe", bpe_vocab=str(vocab))
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=str(enc[0]), decoder=str(dec[0]), joiner=str(joi[0]),
            num_threads=threads, sample_rate=16000, feature_dim=80, **kw)
        self.files = (enc[0].name, dec[0].name, joi[0].name)

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        s = self.rec.create_stream()
        padded = np.concatenate([audio, np.zeros(int(0.66 * 16000), dtype=np.float32)])  # recommended tail padding
        n = len(audio)
        pos = 0
        while pos + CHUNK < n:  # all but the final real chunk: decoded "while speaking", not timed
            s.accept_waveform(16000, audio[pos:pos + CHUNK])
            while self.rec.is_ready(s):
                self.rec.decode_stream(s)
            pos += CHUNK
        t0 = now()
        s.accept_waveform(16000, padded[pos:])
        s.input_finished()
        while self.rec.is_ready(s):
            self.rec.decode_stream(s)
        text = self.rec.get_result(s)
        text = text if isinstance(text, str) else text.text
        return text.strip(), (now() - t0) * 1000


class FasterWhisper(Engine):
    def __init__(self, size: str, threads: int, hotwords: bool) -> None:
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper:{size}-int8" + ("+hotwords" if hotwords else "")
        self.m = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads,
                              download_root=str(MODELS / "whisper"))
        self.hot = ", ".join(hotword_terms()[:60]) if hotwords else None

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = now()
        segs, _ = self.m.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False,
                                    without_timestamps=True, hotwords=self.hot)
        text = " ".join(s.text for s in segs).strip()
        return text, (now() - t0) * 1000


class Moonshine(Engine):
    def __init__(self, size: str, threads: int) -> None:
        import moonshine_onnx as ms

        self.name = f"moonshine:{size}"
        self.ms = ms
        self.model = ms.MoonshineOnnxModel(model_name=f"moonshine/{size}")
        self.tok = ms.load_tokenizer()

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = now()
        tokens = self.model.generate(audio[None, :].astype(np.float32))
        text = self.tok.decode_batch(tokens)[0]
        return text.strip(), (now() - t0) * 1000


def build(spec: str, threads: int) -> Engine:
    kind, _, arg = spec.partition(":")
    if kind == "sherpa":
        name, _, hw = arg.partition("+")
        return SherpaStreaming(name, hw == "hot", threads)
    if kind == "fw":
        size, _, hw = arg.partition("+")
        return FasterWhisper(size, threads, hw == "hot")
    if kind == "moonshine":
        return Moonshine(arg, threads)
    raise SystemExit(f"unknown engine spec {spec}")


def pick_clips() -> tuple[list[tuple[str, Path]], str]:
    real = sorted(RECORDINGS.glob("*.wav"))
    if real:
        return [(p.stem, p) for p in real], "real"
    synth = sorted((RESULTS / "synth").glob("*.wav"))
    return [(p.stem, p) for p in synth], "SYNTHETIC (optimistic)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", required=True,
                    help="sherpa:<dir>[+hot] | fw:<tiny.en|base.en>[+hot] | moonshine:<tiny|base>")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--tag", default="stt")
    a = ap.parse_args()

    refs = {}
    for line in (RECORDINGS / "script.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        cid, text = line.split("\t", 1)
        refs[cid] = text
    clips, kind = pick_clips()
    terms = hotword_terms()
    # ~3 s clip for the speed test
    speed_id = "num01" if any(c[0] == "num01" for c in clips) else clips[0][0]
    speed_audio = load_wav_16k(dict(clips)[speed_id])
    results = []
    for spec in a.engines:
        print(f"== {spec} threads={a.threads}", flush=True)
        try:
            t0 = now()
            eng = build(spec, a.threads)
            load_s = now() - t0
            eng.transcribe(speed_audio)  # warm up
            lats = [eng.transcribe(speed_audio)[1] for _ in range(a.repeat)]
            r: dict = {"engine": eng.name, "clip_seconds": round(len(speed_audio) / 16000, 2),
                       "load_s": round(load_s, 1), "latency_ms": summarize(lats),
                       "latency_kind": "tail after last chunk" if eng.streaming else "decode after endpoint"}
            wers, num_ok, num_n, tf, te = [], 0, 0, 0, 0
            per_group: dict[str, list[float]] = {}
            samples = []
            for cid, path in clips:
                base = cid.split("_")[0]
                if base not in refs:
                    continue
                text, _ = eng.transcribe(load_wav_16k(path))
                w = wer(refs[base], text)
                wers.append(w)
                per_group.setdefault(base.rstrip("0123456789"), []).append(w)
                if base.rstrip("0123456789") in ("num", "plate", "phone", "date"):
                    num_n += 1
                    num_ok += int(canon_numbers(refs[base]) == canon_numbers(text))
                f, e = term_recall(refs[base], text, terms)
                tf, te = tf + f, te + e
                if w > 0:
                    samples.append({"id": cid, "ref": refs[base], "hyp": text})
            r["accuracy"] = {
                "clips": len(wers), "clip_source": kind,
                "wer_mean": round(float(np.mean(wers)), 3) if wers else None,
                "wer_by_group": {k: round(float(np.mean(v)), 3) for k, v in per_group.items()},
                "number_exact": f"{num_ok}/{num_n}",
                "term_recall": f"{tf}/{te}",
                "errors_sample": samples[:14],
            }
            print(f"   {r['latency_kind']}: median {r['latency_ms']['median']}ms p95 {r['latency_ms']['p95']}ms | "
                  f"WER {r['accuracy']['wer_mean']} numbers {num_ok}/{num_n} terms {tf}/{te} ({kind})")
        except Exception as e:
            r = {"engine": spec, "error": f"{type(e).__name__}: {e}"}
            print("   ERROR", r["error"])
        results.append(r)
        save_result(a.tag, {"threads": a.threads, "results": results})


if __name__ == "__main__":
    main()
