"""Everything else the website knows, read from the same `site/data.js`: the full spec sheet of each car, and the showroom details.

A car on the site is built with
  car(id, make, model, year, price, km, body, fuel, seats, colourName, colourHex, hp, nm, engine, drivetrain, accel0to100, topSpeed,
      consumption, bootLitres, interior, owners, features[], blurb)
and data.js then derives, for every car: condition "Used", transmission (single-speed if electric, else automatic), GCC specs, accident-free,
the service history from the owner count, a warranty from age and mileage, and an inspection score out of 150. Those rules are copied here
(`derived`) so the voice agent says exactly what the car page shows; tests compare the two.

Showroom details (hours, locations, phone, WhatsApp, email) come from window.CONFIG in the same file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CarDetails:
    stock_id: str
    full_model: str                 # with the trim, as the page shows it ("Land Cruiser GXR")
    colour_name: str                # the marketing colour ("Pearl White")
    seats: int
    power_hp: int
    torque_nm: int
    engine: str
    drive: str
    accel_0_100: float
    top_speed: int
    consumption: str
    boot_litres: int
    interior: str
    owners: int
    features: tuple[str, ...]
    blurb: str
    transmission: str
    doors: int
    spec: str
    accident: str
    service: str
    warranty: str
    inspection: int                 # checks passed, out of 150


@dataclass(frozen=True)
class Location:
    name: str
    address: str
    hours: str


@dataclass(frozen=True)
class SiteInfo:
    open_hour: int = 9
    close_hour: int = 20
    location: str = ""
    phone: str = ""
    whatsapp: str = ""
    email: str = ""
    locations: tuple[Location, ...] = ()
    deposit: int = 0
    cars: dict[str, CarDetails] = field(default_factory=dict)


def _js_args(call: str) -> list[str]:
    """Top-level arguments of a JS call body, respecting quotes and brackets."""
    args, cur, depth, quote = [], "", 0, ""
    i = 0
    while i < len(call):
        ch = call[i]
        if quote:
            cur += ch
            if ch == "\\" and i + 1 < len(call):
                cur += call[i + 1]
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote, cur = ch, cur + ch
        elif ch in "([{":
            depth, cur = depth + 1, cur + ch
        elif ch in ")]}":
            depth, cur = depth - 1, cur + ch
        elif ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
        i += 1
    if cur.strip():
        args.append(cur.strip())
    return args


def _str(x: str) -> str:
    x = x.strip()
    return x[1:-1].replace('\\"', '"').replace("\\'", "'") if len(x) >= 2 and x[0] in "\"'" and x[-1] == x[0] else x


def _num(x: str) -> float:
    return float(x.strip())


def _call_bodies(js: str) -> list[str]:
    """The text inside each car( ... ) call in window.CARS (balanced parentheses)."""
    start = js.find("window.CARS")
    out = []
    for m in re.finditer(r"\bcar\(", js[start:] if start >= 0 else js):
        i, depth, j = m.end(), 1, m.end()
        text = js[start:] if start >= 0 else js
        quote = ""
        while j < len(text) and depth:
            ch = text[j]
            if quote:
                if ch == "\\":
                    j += 1
                elif ch == quote:
                    quote = ""
            elif ch in "\"'":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        out.append(text[i:j - 1])
    return out


def derived(index: int, year: int, km: int, fuel: str, body: str, owners: int, site_year: int = 2026) -> dict:
    """The fields data.js computes for every car (keep in step with the .map() and car() in data.js)."""
    age = site_year - year
    return {"transmission": "single-speed" if fuel.lower() == "electric" else "automatic",
            "doors": 2 if body == "Coupe" else 5 if body in ("Hatchback", "Van") else 4,
            "spec": "GCC specs", "accident": "accident-free",
            "service": "full dealer service history" if owners == 1 else "full service history",
            "warranty": "12 months or 20,000 kilometres" if age <= 5 and km <= 80000 else "6 months or 10,000 kilometres",
            "inspection": 146 + (index % 5)}


def parse_site(js: str) -> SiteInfo:
    cars: dict[str, CarDetails] = {}
    for i, body in enumerate(_call_bodies(js)):
        a = _js_args(body)
        if len(a) < 23 or not _str(a[0]).startswith("AG-"):
            continue
        sid, year, km, body_type, fuel, owners = _str(a[0]), int(_num(a[3])), int(_num(a[5])), _str(a[6]), _str(a[7]), int(_num(a[20]))
        feats = tuple(_str(x) for x in _js_args(a[21].strip()[1:-1])) if a[21].strip().startswith("[") else ()
        d = derived(i, year, km, fuel, body_type, owners)
        cars[sid] = CarDetails(stock_id=sid, full_model=_str(a[2]), colour_name=_str(a[9]), seats=int(_num(a[8])), power_hp=int(_num(a[11])),
                               torque_nm=int(_num(a[12])), engine=_str(a[13]), drive=_str(a[14]), accel_0_100=_num(a[15]), top_speed=int(_num(a[16])),
                               consumption=_str(a[17]), boot_litres=int(_num(a[18])), interior=_str(a[19]), owners=owners, features=feats, blurb=_str(a[22]),
                               **d)

    def grab(pattern: str, default: str = "") -> str:
        m = re.search(pattern, js)
        return m.group(1) if m else default

    locs = tuple(Location(_str(n), _str(ad), _str(h)) for n, ad, h in
                 re.findall(r'\{\s*name:\s*("[^"]*")\s*,\s*addr:\s*("[^"]*")\s*,\s*hours:\s*("[^"]*")\s*\}', js))
    return SiteInfo(open_hour=int(grab(r"hours:\s*\{\s*open:\s*(\d+)", "9")), close_hour=int(grab(r"close:\s*(\d+)", "20")),
                    location=grab(r'\blocation:\s*"([^"]*)"'), phone=grab(r'\bphone:\s*"([^"]*)"'), whatsapp=grab(r'\bwhatsapp:\s*"([^"]*)"'),
                    email=grab(r'\bemail:\s*"([^"]*)"'), locations=locs, deposit=int(grab(r"depositAmount:\s*(\d+)", "0")), cars=cars)
