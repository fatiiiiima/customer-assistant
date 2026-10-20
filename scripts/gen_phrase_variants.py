r"""Development-time helper: ask the LOCAL model for alternative wordings of the low-risk reply templates and save them, for a
human to review, in config/phrase_variants.yaml.

  .venv\Scripts\python.exe scripts\gen_phrase_variants.py [--per 8] [--only no_match,opening] [--model qwen3:4b-instruct-2507-q4_K_M]

Rules it follows
  * Every candidate goes through the same validator the engine uses (placeholders, required phrases, no new commitment words,
    negation parity, voice rules). Candidates that fail are dropped, never repaired.
  * New entries are saved as `status: proposed`. NOTHING is used at run time until a person changes it to `approved`.
  * Existing entries (approved, rejected, proposed) are never changed or removed; duplicates are skipped.
  * No runtime use: this script is not imported by the agent.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dealer_core.dialogue.phrasebook import SITUATIONS, check_variant  # noqa: E402

OUT = ROOT / "config" / "phrase_variants.yaml"
HEADER = """# Wording variants for low-risk replies. Written offline by the local model (scripts/gen_phrase_variants.py).
#
# REVIEW: for each entry set  status: approved  (use it), leave  proposed  (not used yet) or  rejected  (never use / do not regenerate).
# You may edit the text of an entry, but keep its {placeholders} exactly as in the base wording. Edited entries are re-checked
# when the agent starts; an invalid approved entry stops start-up with a message naming it.
# Only entries marked approved are used, and only when  dialogue.variants: on  in config/default.yaml (currently off).
"""

PROMPT = """You write short spoken lines for a polite used-car dealership phone agent in the UAE. Rewrite the sentence below in {n} different
ways. Keep the same meaning exactly: do not add promises, prices, offers, discounts or new information, and do not change a
question into a statement or the other way round. Use 1 or 2 short sentences, plain spoken English, no emojis.

Situation: {note}
Original: {base}
{rules}
Reply with only the {n} rewrites, one per line, no numbering, no quotes."""


def build_rules(sit) -> str:
    out = []
    if sit.fields:
        names = ", ".join("{" + f + "}" for f in sorted(sit.fields))
        out.append(f"Keep these placeholders exactly as written, each one once: {names}")
    for r in sit.required:
        out.append(f"The phrase '{r}' must appear word for word.")
    return "\n".join(out)


def clean(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
    return line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=8, help="valid new variants wanted per situation")
    ap.add_argument("--only", default="", help="comma-separated situation ids")
    ap.add_argument("--model", default="qwen3:4b-instruct-2507-q4_K_M")
    ap.add_argument("--rounds", type=int, default=4, help="model calls per situation at most")
    args = ap.parse_args()

    import ollama

    client = ollama.Client(host="http://127.0.0.1:11434", timeout=120)
    data = yaml.safe_load(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data = data or {}
    wanted = [s for s in args.only.split(",") if s] or list(SITUATIONS)
    for sid in wanted:
        sit = SITUATIONS[sid]
        entries = data.setdefault(sid, [])
        seen = {e["text"].strip().lower() for e in entries} | {sit.base.lower()}
        added = dropped = 0
        for rnd in range(args.rounds):
            if added >= args.per:
                break
            resp = client.chat(model=args.model, messages=[{"role": "user", "content": PROMPT.format(
                n=args.per, note=sit.note, base=sit.base, rules=build_rules(sit))}],
                options={"temperature": 0.9, "top_p": 0.95, "seed": 100 + rnd, "num_predict": 60 * args.per})
            for line in resp["message"]["content"].splitlines():
                text = clean(line)
                if not text or text.lower() in seen:
                    continue
                seen.add(text.lower())
                if check_variant(sit, text):
                    dropped += 1
                    continue
                if added < args.per:
                    entries.append({"text": text, "status": "proposed"})
                    added += 1
        print(f"{sid:22s} +{added} proposed, {dropped} dropped by the validator", flush=True)
    OUT.write_text(HEADER + "\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=130), encoding="utf-8")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
