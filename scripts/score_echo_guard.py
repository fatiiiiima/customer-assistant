r"""Which recorded turns would an echo test throw away? Scores `echoes_agent` variants on the words logs (logs/audio_dump/*_in.txt: "stt" = what the recogniser wrote for a
segment, "said" = each sentence the agent spoke) without recognising anything again, so it is free and repeatable.

  uv run python scripts/score_echo_guard.py                # every log; prints each turn a variant would flag, with what the agent had just said and what it did next
  uv run python scripts/score_echo_guard.py --quiet        # only the counts

For each "stt" line the agent's line is every sentence it said since the previous "stt" line (the hint the recogniser was given). A flagged turn is judged by reading it:
an echo (the recogniser wrote the agent's question: good to catch) or a real answer that happens to share words with the question (bad to throw away).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def turns(path: Path) -> list[dict]:
    """[{stt, agent_line, next_said}] in order for one words log."""
    out, said = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        kind, text = parts[1], parts[2]
        if kind == "said":
            if out and out[-1]["next_said"] is None:
                out[-1]["next_said"] = text
            said.append(text)
        elif kind == "stt":
            out.append({"stt": text, "agent_line": " ".join(said), "next_said": None})
            said = []
    return out


def main() -> int:
    from dealer_core.voice import services

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=Path, default=ROOT / "logs" / "audio_dump")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    def plan_overlap(heard: str, agent_line: str, threshold: float = 0.6, min_words: int = 4) -> bool:
        """The rule ARCH_PLAN.md C5 proposed (not in the code): 60% of the heard words in ANY one sentence of the agent's line, 4 or more words, no matching start."""
        h = services._words(heard)
        if len(h) < min_words:
            return False
        return any(len(w) >= 3 and sum(x in w for x in h) / len(h) >= threshold for w in map(services._words, re.split(r"(?<=[.!?])\s+", agent_line)))

    variants = {"echoes_agent (in the code)": services.echoes_agent, "plan_overlap (ARCH_PLAN C5, not adopted)": plan_overlap}
    total = 0
    flagged = {n: [] for n in variants}
    for path in sorted(a.folder.glob("*_in.txt")):
        for t in turns(path):
            if not t["stt"].strip():
                continue
            total += 1
            for n, fn in variants.items():
                if t["agent_line"] and fn(t["stt"], t["agent_line"]):
                    flagged[n].append((path.name[:15], t))
    print(f"{total} recognised segments in {len(list(a.folder.glob('*_in.txt')))} logs")
    for n, rows in flagged.items():
        print(f"\n{n}: {len(rows)} flagged")
        if a.quiet:
            continue
        for stamp, t in rows:
            print(f"  [{stamp}] heard : {t['stt']!r}\n             agent : {t['agent_line'][-110:]!r}\n             then  : {t['next_said']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
