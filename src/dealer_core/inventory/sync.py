"""Bringing the database in line with the dealership's stock list: the one way cars get into the database.

A sync compares the new list with what the database holds and writes only the difference, in one transaction:
  * a car not seen before is added; a car whose price, mileage, spec sheet... changed is updated; the rest is left alone;
  * a car that was sold and is listed again is back in stock;
  * a car no longer listed is marked sold (its bookings and history stay), unless `missing="keep"`;
  * a reserved car stays reserved unless the list itself has a status column.
Safety: a list that would take more than `max_gone` of the cars on sale off sale at once (a half-exported file, the wrong sheet) is refused
unless forced, and a list with no readable row changes nothing. A dry run reports what would happen and writes nothing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..models import Car, CarStatus
from ..site.details import CarDetails, SiteInfo
from .columns import REQUIRED, Branches, RowError, map_headings, to_record
from .sources import guess_format, read_data_js, read_rows


@dataclass
class Report:
    source: str
    rows: int = 0
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    back_in_stock: list[str] = field(default_factory=list)
    gone: list[str] = field(default_factory=list)                   # marked sold: no longer on the list
    unchanged: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)     # (row, why it was skipped)
    unknown_headings: list[str] = field(default_factory=list)       # columns not recognised (not imported; name them with a mapping)
    dry_run: bool = False
    refused: str = ""                                               # why nothing was written

    @property
    def written(self) -> bool:
        return not (self.dry_run or self.refused)

    def plain(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        head = ("DRY RUN, nothing written. " if self.dry_run else "") + (f"REFUSED, nothing written: {self.refused}. " if self.refused else "")
        lines = [f"{head}{self.source}: {self.rows} row(s) read; {len(self.added)} added, {len(self.updated)} updated, "
                 f"{len(self.back_in_stock)} back in stock, {len(self.gone)} no longer listed (marked sold), {self.unchanged} unchanged, "
                 f"{len(self.errors)} skipped."]
        lines += [f"  skipped {row}: {why}" for row, why in self.errors[:50]]
        if len(self.errors) > 50:
            lines.append(f"  ... and {len(self.errors) - 50} more")
        if self.unknown_headings:
            lines.append("  columns not imported (not recognised; name them with --map): " + ", ".join(self.unknown_headings))
        return "\n".join(lines)


def sync(repo, records: list[tuple[Car, CarDetails]], *, source: str, showroom: SiteInfo | None = None, status_given: bool = False,
         missing: str = "sold", max_gone: float = 0.5, force: bool = False, dry_run: bool = False, report: Report | None = None) -> Report:
    rep = report or Report(source)
    rep.dry_run = dry_run
    rep.rows = rep.rows or len(records)
    if not records:
        rep.refused = "no car could be read from the list"
        return rep
    have = repo.snapshot()
    cars: list[Car] = []
    details: list[CarDetails] = []
    seen: set[str] = set()
    for car, d in records:
        if car.stock_id in seen:
            rep.errors.append((car.stock_id, "the same stock id appears twice; the first row was used"))
            continue
        seen.add(car.stock_id)
        old_car, old_d = have.get(car.stock_id, (None, None))
        if not status_given and old_car is not None and old_car.status is CarStatus.RESERVED:
            car = car.model_copy(update={"status": CarStatus.RESERVED})       # reserved here (a deposit): the list does not know
        if old_car is None:
            rep.added.append(car.stock_id)
        elif old_car == car and old_d == d:
            rep.unchanged += 1
            continue
        elif old_car.status is CarStatus.SOLD and car.status is CarStatus.AVAILABLE:
            rep.back_in_stock.append(car.stock_id)
        else:
            rep.updated.append(car.stock_id)
        cars.append(car)
        details.append(d)
    statuses: dict[str, CarStatus] = {}
    if missing == "sold":
        on_sale = [sid for sid, (c, _) in have.items() if c.status is not CarStatus.SOLD]
        rep.gone = sorted(sid for sid in on_sale if sid not in seen)
        statuses = {sid: CarStatus.SOLD for sid in rep.gone}
        if rep.gone and not force and len(on_sale) >= 10 and len(rep.gone) > max_gone * len(on_sale):
            rep.refused = (f"{len(rep.gone)} of the {len(on_sale)} cars on sale are not on this list; if they really are all gone, run it again with --force")
    if showroom is not None and repo.showroom().plain() == showroom.plain():
        showroom = None                                                    # the same as before: not written again
    if rep.written and (cars or statuses or showroom is not None):          # nothing changed: nothing written (every worker may import the same list)
        meta = {"synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "last_import": json.dumps(rep.plain())}
        repo.apply_sync(cars, details, statuses, showroom, meta)
    return rep


def import_file(repo, path: Path, branches: Branches, *, fmt: str | None = None, mapping: dict[str, str] | None = None, missing: str = "sold",
                max_gone: float = 0.5, force: bool = False, dry_run: bool = False) -> Report:
    """Read a stock list (CSV, TSV, Excel, JSON or the website's data.js) and sync the database with it."""
    fmt = fmt or guess_format(path)
    if fmt == "datajs":
        records, site = read_data_js(path.read_text(encoding="utf-8"), branches.default)
        return sync(repo, records, source=path.name, showroom=site, missing=missing, max_gone=max_gone, force=force, dry_run=dry_run)
    headings, rows = read_rows(path, fmt)
    heads, unknown = map_headings(headings, mapping)
    rep = Report(path.name, rows=len(rows), unknown_headings=unknown)
    absent = [f for f in REQUIRED if f not in heads.values()]
    if absent:
        rep.refused = ("no column for " + ", ".join(a.replace("_", " ") for a in absent) + f" (the columns are: {', '.join(map(str, headings))}); "
                       "name them with --map")
        return rep
    records = []
    for i, row in enumerate(rows, start=2):                              # row 1 is the headings, as a spreadsheet numbers them
        try:
            records.append(to_record(row, heads, branches))
        except RowError as e:
            rep.errors.append((f"row {i}", str(e)))
    return sync(repo, records, source=path.name, status_given="status" in heads.values(), missing=missing, max_gone=max_gone, force=force,
                dry_run=dry_run, report=rep)
