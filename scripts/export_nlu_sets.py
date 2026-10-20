"""One-off: export the Phase 0b labelled sets (benchmarks/nlu_cases.py) to data/nlu_sets/*.json for dealer_core."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "benchmarks"))
from nlu_cases import CASES, FRESH, HELDOUT  # noqa: E402

out = root / "data" / "nlu_sets"
out.mkdir(parents=True, exist_ok=True)
for name, cases in (("dev", CASES), ("heldout", HELDOUT), ("fresh", FRESH)):
    rows = [{"state": s, "car": car, "utterance": u, "intent": g["intent"], "slots": g["slots"]} for s, car, u, g in cases]
    (out / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(name, len(rows))
