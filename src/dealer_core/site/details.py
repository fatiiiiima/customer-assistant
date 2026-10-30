"""The spec sheet of a car and the showroom details, and the demo reader of the website's `site/data.js`.

The spec sheets and the showroom details live in the database (tables car_details and showroom; inventory/ imports them from whatever the
dealership keeps). Every field but the stock id may be None: "not recorded", said as such, never guessed (owner, 30 Oct).

data.js (the demo stock) builds a car with
  car(id, make, model, year, price, km, body, fuel, seats, colourName, colourHex, hp, nm, engine, drivetrain, accel0to100, topSpeed,
      consumption, bootLitres, interior, owners, features[], blurb)
and then derives, for every car: condition "Used", transmission (single-speed if electric, else automatic), GCC specs, accident-free,
the service history from the owner count, a market price, a warranty from age and mileage, and an inspection score out of 150. Those rules are
copied here (`derived`), worded as the car page shows them, so the demo import holds what the page says; tests compare the two.

Showroom details (hours, locations, phone, WhatsApp, email) come from window.CONFIG in the same file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CarDetails:
    stock_id: str
    full_model: str | None = None           # with the trim, as the page shows it ("Land Cruiser GXR")
    colour_name: str | None = None          # the marketing colour ("Pearl White")
    seats: int | None = None
    power_hp: int | None = None
    torque_nm: int | None = None
    engine: str | None = None
    drive: str | None = None
    accel_0_100: float | None = None
    top_speed: int | None = None
    consumption: str | None = None
    boot_litres: int | None = None
    interior: str | None = None
    owners: int | None = None
    features: tuple[str, ...] = ()
    blurb: str | None = None
    transmission: str | None = None
    doors: int | None = None
    spec: str | None = None                 # "GCC specs"
    accident: str | None = None             # "Accident-free"
    service: str | None = None              # "Full dealer service history"
    warranty: str | None = None             # "12 months / 20,000 km"
    inspection: int | None = None           # checks passed, out of 150
    colour_hex: str | None = None           # for the website's colour swatch
    condition: str | None = None            # "Used"
    market_price: int | None = None         # a market reference price (the site's price rating)


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
    cars: dict[str, CarDetails] = field(default_factory=dict)      # only as read from a data.js; the voice agent reads spec sheets from the database

    def plain(self) -> dict:
        """The showroom details as plain JSON (the database row and the website API), without the cars."""
        return {"open_hour": self.open_hour, "close_hour": self.close_hour, "location": self.location, "phone": self.phone, "whatsapp": self.whatsapp,
                "email": self.email, "deposit": self.deposit, "locations": [{"name": x.name, "address": x.address, "hours": x.hours} for x in self.locations]}

    @classmethod
    def from_plain(cls, d: dict) -> "SiteInfo":
        return cls(open_hour=int(d.get("open_hour", 9)), close_hour=int(d.get("close_hour", 20)), location=d.get("location", ""), phone=d.get("phone", ""),
                   whatsapp=d.get("whatsapp", ""), email=d.get("email", ""), deposit=int(d.get("deposit", 0)),
                   locations=tuple(Location(x.get("name", ""), x.get("address", ""), x.get("hours", "")) for x in d.get("locations", ())))


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


MARKET_FACTORS = (1.06, 1.02, 1.09, 1.04, 0.99, 1.07, 1.03, 1.05, 1.08, 1.01, 1.06, 1.04, 1.1, 1.02, 1.05, 1.07, 0.98, 1.06, 1.03, 1.05, 1.09, 1.02, 1.04, 1.07)


def _js_round(x: float) -> int:
    """JavaScript's Math.round (halves go up)."""
    import math

    return math.floor(x + 0.5)


def derived(index: int, year: int, km: int, fuel: str, body: str, owners: int, price: int = 0, site_year: int = 2026) -> dict:
    """The fields data.js computes for every car, worded as the page shows them (keep in step with the .map() and car() in data.js)."""
    age = site_year - year
    return {"condition": "Used", "transmission": "Single-speed" if fuel.lower() == "electric" else "Automatic",
            "doors": 2 if body == "Coupe" else 5 if body in ("Hatchback", "Van") else 4,
            "spec": "GCC specs", "accident": "Accident-free",
            "service": "Full dealer service history" if owners == 1 else "Full service history",
            "market_price": _js_round(price * MARKET_FACTORS[index % 24] / 500) * 500,
            "warranty": "12 months / 20,000 km" if age <= 5 and km <= 80000 else "6 months / 10,000 km",
            "inspection": 146 + (index % 5)}


def parse_site(js: str) -> SiteInfo:
    cars: dict[str, CarDetails] = {}
    for i, body in enumerate(_call_bodies(js)):
        a = _js_args(body)
        if len(a) < 23 or not _str(a[0]).startswith("AG-"):
            continue
        sid, year, km, body_type, fuel, owners = _str(a[0]), int(_num(a[3])), int(_num(a[5])), _str(a[6]), _str(a[7]), int(_num(a[20]))
        feats = tuple(_str(x) for x in _js_args(a[21].strip()[1:-1])) if a[21].strip().startswith("[") else ()
        d = derived(i, year, km, fuel, body_type, owners, int(_num(a[4])))
        cars[sid] = CarDetails(stock_id=sid, full_model=_str(a[2]), colour_name=_str(a[9]), colour_hex=_str(a[10]), seats=int(_num(a[8])), power_hp=int(_num(a[11])),
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
