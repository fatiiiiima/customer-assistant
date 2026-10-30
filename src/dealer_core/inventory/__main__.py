"""The stock from the command line.

  python -m dealer_core.inventory headings stock.xlsx                 how each column of a stock list would be read (nothing written)
  python -m dealer_core.inventory import stock.csv --dry-run          what an import would change (nothing written)
  python -m dealer_core.inventory import stock.csv                    bring the database in line with the list
  python -m dealer_core.inventory import stock.json --map map.json    with {"Their heading": "price", ...} for headings not recognised
  python -m dealer_core.inventory status                              how many cars, by status; the last import
  python -m dealer_core.inventory serve --port 8070                   the website's car API (inventory/api.py)

The database: --db, else DATABASE_URL, else config app.database_url (postgresql://... or sqlite:///stock.db).
Exit status: 0 done, 1 refused or failed (nothing written), so a scheduled import can raise the alarm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..app import ROOT
from ..config import load_config
from ..models import CarStatus, SearchFilters
from . import branches_of, open_db
from .columns import REQUIRED, SYNONYMS, map_headings
from .sources import FORMATS, guess_format, read_rows
from .sync import import_file


def _db_url(a, cfg) -> str:
    return a.db or os.environ.get("DATABASE_URL", "") or cfg.app.database_url


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dealer_core.inventory", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("headings", "import", "status", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--db", default="", help="database address (default: DATABASE_URL, then config app.database_url)")
        p.add_argument("--config", type=Path, default=ROOT / "config" / "portal.yaml", help="config overlay with the branches")
        if name in ("headings", "import"):
            p.add_argument("file", type=Path)
            p.add_argument("--format", choices=FORMATS, help="default: from the file's extension")
            p.add_argument("--map", type=Path, help='JSON file {"Their heading": "field"} for headings not recognised (fields: ' + ", ".join(SYNONYMS) + ")")
        if name == "import":
            p.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
            p.add_argument("--keep-missing", action="store_true", help="leave cars that are not on the list as they are (default: mark them sold)")
            p.add_argument("--force", action="store_true", help="write even if the list would take more than half the cars on sale off sale")
            p.add_argument("--json", action="store_true", help="print the report as JSON")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8070)
            p.add_argument("--origin", action="append", help="a page allowed to call the API (repeat); default the local site on 8080")
    a = ap.parse_args(argv)
    cfg = load_config(overlay=a.config)
    mapping = json.loads(a.map.read_text(encoding="utf-8")) if getattr(a, "map", None) else None

    if a.cmd == "headings":
        if (a.format or guess_format(a.file)) == "datajs":
            print(f"{a.file.name}: the website's data.js, read by its fixed layout (car(id, make, model, ...)); no headings to map.")
            return 0
        headings, rows = read_rows(a.file, a.format)
        heads, unknown = map_headings(headings, mapping)
        print(f"{a.file.name}: {len(rows)} row(s)")
        for h in headings:
            print(f"  {h!s:<30} -> {heads.get(h, '(not recognised: not imported)')}")
        absent = [f for f in REQUIRED if f not in heads.values()]
        print("Missing required: " + ", ".join(absent) if absent else "Every required column is there.")
        return 1 if absent else 0

    url = _db_url(a, cfg)
    if not url and not (a.cmd == "import" and a.dry_run):
        print("No database: give --db, or set DATABASE_URL (an in-memory database would lose the stock at exit).", file=sys.stderr)
        return 1
    inv, _ = open_db(url, cfg.app.db_pool_size)

    if a.cmd == "import":
        rep = import_file(inv, a.file, branches_of(cfg), fmt=a.format, mapping=mapping, missing="keep" if a.keep_missing else "sold",
                          force=a.force, dry_run=a.dry_run)
        print(json.dumps(rep.plain(), indent=1) if a.json else rep.summary())
        return 1 if rep.refused else 0

    if a.cmd == "status":
        every = SearchFilters(include_unavailable=True)
        counts = {s.value: 0 for s in CarStatus}
        for c in inv.search(every, inv.count(every) or 1):
            counts[c.status.value] += 1
        print(f"{sum(counts.values())} car(s): " + ", ".join(f"{n} {s}" for s, n in counts.items()) + f"; stock version {inv.version()}")
        print(f"Last import: {inv.meta('synced_at') or 'never'}")
        last = inv.meta("last_import")
        if last:
            r = json.loads(last)
            print(f"  {r['source']}: {len(r['added'])} added, {len(r['updated'])} updated, {len(r['gone'])} marked sold, {len(r['errors'])} skipped")
        return 0

    import uvicorn

    from .api import create_api

    print(f"Stock API on http://{a.host}:{a.port}/api/cars (pages allowed: {', '.join(a.origin or ['http://localhost:8080'])})", flush=True)
    uvicorn.run(create_api(inv, a.origin), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
