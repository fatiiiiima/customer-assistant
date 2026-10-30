"""Load test for the stock database (ARCH_PLAN step 0): 5,000 generated cars, imported through the same importer as a real stock list,
then the reads a voice turn and a page view make, timed on SQLite (in memory) and on PostgreSQL (local, started by pgserver).

  uv run python scripts/inventory_load_test.py [--cars 5000] [--reps 300] [--no-pg]

Prints p50 / p99 / max per operation and writes benchmarks/results/inventory_load_<date>.json. Local only: nothing paid, no GPU.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dealer_core.inventory.columns import Branches  # noqa: E402
from dealer_core.inventory.sync import import_file  # noqa: E402
from dealer_core.models import BodyType, SearchFilters  # noqa: E402

MAKES = {"Toyota": ["Land Cruiser", "Camry", "Corolla", "RAV4", "Prado", "Hilux"], "Nissan": ["Patrol", "Altima", "X-Trail", "Sunny"],
         "BMW": ["X5", "320i", "530i", "X3"], "Mercedes-Benz": ["C 300", "GLE 450", "E 200"], "Kia": ["Sorento", "Sportage", "Rio"],
         "Hyundai": ["Tucson", "Elantra", "Santa Fe"], "Lexus": ["RX 350", "ES 350", "LX 600"], "Ford": ["Ranger", "Explorer", "Mustang"],
         "Audi": ["A4", "Q7", "Q5"], "Porsche": ["Macan", "Cayenne", "911"]}
BODIES = ["SUV", "Sedan", "Hatchback", "Coupe", "Pickup", "Van"]
COLOURS = ["Pearl White", "Black", "Silver", "Grey", "Red", "Blue", "Bronze"]


def fleet(n: int, path: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    head = "Stock No.,Make,Model,Year,Price (AED),Mileage,Body,Fuel,Colour,Seats,HP,Engine,Drive,Owners,Warranty,Accident,Features,Description\n"
    rows = []
    for i in range(n):
        make = rng.choice(list(MAKES))
        rows.append(",".join([f"L-{i:05d}", make, rng.choice(MAKES[make]), str(rng.randint(2014, 2025)), str(rng.randrange(30000, 600000, 500)),
                              str(rng.randrange(5000, 220000, 1000)), rng.choice(BODIES), rng.choice(["Petrol", "Petrol", "Diesel", "Hybrid", "Electric"]),
                              rng.choice(COLOURS), str(rng.choice([2, 4, 5, 5, 5, 7, 8])) if rng.random() > 0.1 else "", str(rng.randint(110, 620)),
                              f"{rng.choice([1.6, 2.0, 2.5, 3.5, 4.0, 5.7])}L", rng.choice(["FWD", "RWD", "AWD", "4WD"]), str(rng.randint(1, 4)),
                              rng.choice(["12 months / 20,000 km", "6 months / 10,000 km", ""]), rng.choice(["Accident-free", ""]),
                              '"Reverse camera; Cruise control; Apple CarPlay"', '"A well-kept car with a full set of keys."']))
    path.write_text(head + "\n".join(rows) + "\n", encoding="utf-8")


def timed(fn, reps: int) -> dict:
    fn()                                                              # warm
    ms = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t) * 1000)
    ms.sort()
    return {"p50_ms": round(statistics.median(ms), 3), "p99_ms": round(ms[min(len(ms) - 1, int(len(ms) * 0.99))], 3), "max_ms": round(ms[-1], 3), "reps": reps}


def run(inv, csv: Path, reps: int, label: str) -> dict:
    out: dict = {}
    br = Branches(default="AGMC-HQ", by_name={"agmc-hq": "AGMC-HQ"})
    t = time.perf_counter()
    rep = import_file(inv, csv, br)
    out["first_import_s"] = round(time.perf_counter() - t, 2)
    out["first_import"] = {"added": len(rep.added), "skipped": len(rep.errors)}
    t = time.perf_counter()
    rep = import_file(inv, csv, br)
    out["reimport_unchanged_s"] = round(time.perf_counter() - t, 2)
    rng = random.Random(1)
    ids = [f"L-{i:05d}" for i in range(0, 5000, 7)]
    from dealer_core.inventory.api import create_api
    from fastapi.testclient import TestClient

    client = TestClient(create_api(inv))
    ops = {
        "search make+budget (voice)": lambda: inv.search(SearchFilters(make=rng.choice(list(MAKES)).replace("-Benz", ""), max_price=rng.randrange(80000, 400000, 10000)), 3),
        "count make+body+budget (voice)": lambda: inv.count(SearchFilters(make="Toyota", body_type=BodyType.SUV, max_price=200000)),
        "search 7+ seats, by power (voice)": lambda: inv.search(SearchFilters(min_seats=7, sort="power", max_price=rng.randrange(100000, 500000, 10000)), 3),
        "search fuel=hybrid (voice)": lambda: inv.search(SearchFilters(fuel="hybrid", body_type=BodyType.SEDAN), 3),
        "get car (voice)": lambda: inv.get(rng.choice(ids)),
        "spec sheet of one car (voice)": lambda: inv.details(rng.choice(ids)),
        "spec sheets of 5 cars (voice)": lambda: inv.details_many(rng.sample(ids, 5)),
        "showroom (voice)": inv.showroom,
        "vocabulary (call start)": inv.vocabulary,
        "GET /api/cars page of 24 (site)": lambda: client.get("/api/cars", params={"make": "Toyota", "limit": 24, "offset": rng.randrange(0, 200, 24)}),
        "GET /api/cars/{id} (site)": lambda: client.get(f"/api/cars/{rng.choice(ids)}"),
        "GET /api/site-cars.js cached (site)": lambda: client.get("/api/site-cars.js"),
    }
    out["ops"] = {name: timed(fn, reps) for name, fn in ops.items()}
    inv.mark_status(ids[0], __import__("dealer_core.models", fromlist=["CarStatus"]).CarStatus.RESERVED)
    t = time.perf_counter()
    body = client.get("/api/site-cars.js").content
    out["site_cars_js_rebuild_ms"] = round((time.perf_counter() - t) * 1000, 1)
    out["site_cars_js_bytes"] = len(body)
    print(f"\n{label}: first import {out['first_import_s']} s ({out['first_import']}), unchanged re-import {out['reimport_unchanged_s']} s")
    for name, r in out["ops"].items():
        print(f"  {name:<38} p50 {r['p50_ms']:>8.3f} ms   p99 {r['p99_ms']:>8.3f} ms   max {r['max_ms']:>8.3f} ms")
    print(f"  site-cars.js rebuilt after a change: {out['site_cars_js_rebuild_ms']} ms, {out['site_cars_js_bytes'] / 1e6:.2f} MB")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--no-pg", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="stock_load_"))
    csv = tmp / "stock.csv"
    fleet(a.cars, csv)
    import platform

    result = {"date": date.today().isoformat(), "machine": platform.machine(), "cpu": platform.processor(), "cars": a.cars, "reps": a.reps}
    from dealer_core.inventory import open_db

    result["sqlite_memory"] = run(open_db("")[0], csv, a.reps, "SQLite (in memory)")
    if not a.no_pg:
        import pgserver

        srv = pgserver.get_server(str(tmp / "pg"), cleanup_mode="stop")
        result["postgresql_local"] = run(open_db(srv.get_uri())[0], csv, a.reps, "PostgreSQL (local, pgserver, unix socket)")
    out = ROOT / "benchmarks" / "results" / f"inventory_load_{date.today().isoformat()}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nWritten {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
