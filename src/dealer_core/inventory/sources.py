"""Reading a stock list, whatever it is kept in: a spreadsheet saved as CSV or Excel (.xlsx), a JSON export from a dealer system, or the
website's data.js (the demo stock).

Every reader gives (headings, rows); data.js is the exception and gives Cars and spec sheets directly (its layout is fixed). Excel is read with
the standard library (an .xlsx is a zip of XML files), so nothing has to be installed; the first sheet is read, its first non-empty row is the headings.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from ..models import Car
from ..site.details import CarDetails, SiteInfo, parse_site
from ..site.portal import parse_cars

FORMATS = ("csv", "tsv", "xlsx", "json", "datajs")


def guess_format(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if path.name.lower() == "data.js" or ext == "js":
        return "datajs"
    if ext in ("xlsx", "xlsm"):
        return "xlsx"
    if ext in ("csv", "tsv", "json"):
        return ext
    if ext == "txt":
        return "tsv" if "\t" in path.read_text(encoding="utf-8-sig", errors="replace").split("\n", 1)[0] else "csv"
    raise ValueError(f"{path.name}: say which format it is (one of {', '.join(FORMATS)})")


def read_csv(text: str, delimiter: str | None = None) -> tuple[list[str], list[dict]]:
    text = text.lstrip("﻿")
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = [r for r in reader if any((x or "").strip() for x in r.values() if isinstance(x, str))]
    return list(reader.fieldnames or []), rows


def read_json(text: str) -> tuple[list[str], list[dict]]:
    """A list of objects, or an object holding one ({"cars": [...]}, {"data": {"vehicles": [...]}}...)."""
    def lists_in(x) -> list[list]:
        if isinstance(x, list):
            return [x] if x and all(isinstance(r, dict) for r in x) else []
        return [found for v in x.values() for found in lists_in(v)] if isinstance(x, dict) else []

    found = lists_in(json.loads(text))
    if not found:
        raise ValueError("no list of cars found in the JSON")
    data = max(found, key=len)                                          # the longest list of objects anywhere in it
    headings: list[str] = []
    for row in data:
        for k in row:
            if k not in headings:
                headings.append(k)
    return headings, [r for r in data if isinstance(r, dict)]


_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _col_index(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(data: bytes) -> tuple[list[str], list[dict]]:
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    sheet = "xl/worksheets/sheet1.xml"
    if "xl/workbook.xml" in names and "xl/_rels/workbook.xml.rels" in names:        # the first sheet in the workbook's order
        first = ElementTree.fromstring(z.read("xl/workbook.xml")).find("m:sheets/m:sheet", _NS)
        rels = ElementTree.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next((r.get("Target") for r in rels if first is not None and r.get("Id") == first.get(_REL)), None)
        if target:
            sheet = target.lstrip("/") if target.startswith("/") else "xl/" + target
    table: list[list] = []
    for row in ElementTree.fromstring(z.read(sheet)).iter(f"{{{_NS['m']}}}row"):
        cells: list = []
        for c in row.findall("m:c", _NS):
            i = _col_index(c.get("r", "A"))
            kind, v = c.get("t"), c.find("m:v", _NS)
            if kind == "s" and v is not None:
                val = shared[int(v.text)]
            elif kind == "inlineStr":
                val = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
            elif kind == "b" and v is not None:
                val = "yes" if v.text == "1" else "no"
            elif v is not None and v.text is not None:
                num = float(v.text) if kind in (None, "n") and re.fullmatch(r"-?\d+(\.\d+)?(E-?\d+)?", v.text, re.I) else None
                val = (int(num) if num is not None and num.is_integer() else num) if num is not None else v.text
            else:
                val = None
            cells.extend([None] * (i - len(cells) + 1))
            cells[i] = val
        table.append(cells)
    table = [r for r in table if any(x not in (None, "") for x in r)]
    if not table:
        return [], []
    headings = [str(h).strip() if h is not None else "" for h in table[0]]
    return headings, [{h: (r[i] if i < len(r) else None) for i, h in enumerate(headings) if h} for r in table[1:]]


def read_rows(path: Path, fmt: str | None = None) -> tuple[list[str], list[dict]]:
    fmt = fmt or guess_format(path)
    if fmt == "xlsx":
        return read_xlsx(path.read_bytes())
    text = path.read_text(encoding="utf-8-sig")
    if fmt == "json":
        return read_json(text)
    if fmt in ("csv", "tsv"):
        return read_csv(text, "\t" if fmt == "tsv" else None)
    raise ValueError(f"unknown format {fmt!r}")


def read_data_js(text: str, branch_id: str) -> tuple[list[tuple[Car, CarDetails]], SiteInfo]:
    """The demo stock: the website's data.js, its cars with their spec sheets as the car page shows them, and its showroom details."""
    site = parse_site(text)
    cars = parse_cars(text, branch_id)
    if not cars:
        raise ValueError("no cars found in data.js: has it changed shape? (expected car(\"AG-101\", \"Make\", \"Model\", year, price, km, ...))")
    return [(c, site.cars.get(c.stock_id) or CarDetails(c.stock_id)) for c in cars], site
