"""From one row of somebody's stock list to a Car and its spec sheet.

Nobody knows yet where the dealership keeps its stock (30 Oct), so the importer does not expect one layout. A column is recognised by its
heading, whatever the spelling: "Stock No.", "stock_id" and "Ref" are all the stock id; "Price (AED)", "Selling price" and "Asking price" are
all the price. A heading that is not recognised is reported, never guessed at; `mapping` names it ({"Their heading": "price"}).

Values are read as people type them: "AED 189,000", "92,000 km", "Yes", "4WD", "Leather; Sunroof; Camera". What the row does not say is
None, "not recorded", and the agent says so; nothing is made up. Without a floor price the floor is the list price (prices are fixed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import BodyType, Car, CarStatus
from ..site.details import CarDetails
from ..site.portal import COLOUR_WORDS, MAKE, base_model, plain_colour

# canonical field -> the headings that mean it (compared after lower-casing and dropping everything but letters and digits)
SYNONYMS: dict[str, tuple[str, ...]] = {
    "stock_id": ("stockid", "stockno", "stocknumber", "stocknum", "stock", "id", "ref", "reference", "refno", "listingid", "carid", "sku", "unitid"),
    "vin": ("vin", "chassis", "chassisno", "chassisnumber", "vinnumber"),
    "make": ("make", "brand", "manufacturer", "marque"),
    "model": ("model", "modelname"),
    "trim": ("trim", "variant", "version", "grade", "derivative"),
    "year": ("year", "modelyear", "my", "yearofmanufacture", "regyear"),
    "price": ("price", "priceaed", "sellingprice", "askingprice", "listprice", "retailprice", "saleprice", "aed"),
    "floor_price": ("floorprice", "minprice", "minimumprice", "lowestprice", "bottomprice"),
    "market_price": ("market", "marketprice", "marketvalue"),
    "km": ("km", "kms", "mileage", "mileagekm", "odometer", "odometerkm", "kilometres", "kilometers", "odo"),
    "body_type": ("body", "bodytype", "bodystyle", "type", "category", "vehicletype"),
    "fuel": ("fuel", "fueltype"),
    "transmission": ("transmission", "gearbox", "gear"),
    "colour_name": ("colour", "color", "exteriorcolour", "exteriorcolor", "colourname", "colorname", "paint"),
    "colour_hex": ("colourhex", "colorhex", "hex"),
    "seats": ("seats", "seating", "seatingcapacity", "noofseats", "numberofseats", "passengers"),
    "power_hp": ("hp", "power", "powerhp", "horsepower", "bhp"),
    "torque_nm": ("torque", "torquenm", "nm"),
    "engine": ("engine", "enginesize", "enginedescription"),
    "drive": ("drive", "drivetrain", "drivetype", "driveline", "wheeldrive"),
    "accel_0_100": ("accel", "accel0to100", "acceleration", "0100", "0to100", "zeroto100", "0100kmh"),
    "top_speed": ("top", "topspeed", "topspeedkmh", "vmax", "maxspeed"),
    "consumption": ("cons", "consumption", "fuelconsumption", "economy", "fueleconomy"),
    "boot_litres": ("boot", "bootlitres", "bootspace", "trunk", "cargo", "luggage"),
    "interior": ("interior", "upholstery", "interiorcolour", "interiorcolor", "trimcolour"),
    "owners": ("owners", "previousowners", "noofowners", "ownercount"),
    "features": ("features", "options", "equipment", "highlights", "extras"),
    "blurb": ("blurb", "description", "notes", "comments", "summary"),
    "doors": ("doors", "noofdoors"),
    "spec": ("spec", "specs", "regionalspec", "specification", "gccspec"),
    "accident": ("accident", "accidents", "accidenthistory", "accidentfree"),
    "service": ("service", "servicehistory", "servicerecords"),
    "warranty": ("warranty", "warrantycover"),
    "inspection": ("inspection", "inspectionscore", "inspectionresult", "checkspassed"),
    "condition": ("condition", "newused"),
    "status": ("status", "availability", "state", "stockstatus"),
    "branch": ("branch", "branchid", "showroom", "location", "site", "yard"),
    "images": ("images", "imageurls", "photos", "photo", "image", "pictures", "imageurl"),
}
REQUIRED = ("stock_id", "make", "model", "year", "price", "km", "body_type")
_LOOKUP = {syn: canon for canon, syns in SYNONYMS.items() for syn in syns}

BODY_WORDS = {"suv": BodyType.SUV, "crossover": BodyType.SUV, "4x4": BodyType.SUV, "jeep": BodyType.SUV, "sedan": BodyType.SEDAN, "saloon": BodyType.SEDAN,
              "hatchback": BodyType.HATCHBACK, "hatch": BodyType.HATCHBACK, "coupe": BodyType.COUPE, "coupé": BodyType.COUPE,
              "convertible": BodyType.COUPE, "cabriolet": BodyType.COUPE, "roadster": BodyType.COUPE, "pickup": BodyType.PICKUP,
              "pick-up": BodyType.PICKUP, "pick up": BodyType.PICKUP, "truck": BodyType.PICKUP, "van": BodyType.VAN, "minivan": BodyType.VAN,
              "mpv": BodyType.VAN, "people carrier": BodyType.VAN}
STATUS_WORDS = {"available": CarStatus.AVAILABLE, "in stock": CarStatus.AVAILABLE, "for sale": CarStatus.AVAILABLE, "active": CarStatus.AVAILABLE,
                "yes": CarStatus.AVAILABLE, "reserved": CarStatus.RESERVED, "on hold": CarStatus.RESERVED, "hold": CarStatus.RESERVED,
                "deposit taken": CarStatus.RESERVED, "booked": CarStatus.RESERVED, "sold": CarStatus.SOLD, "no": CarStatus.SOLD,
                "delivered": CarStatus.SOLD, "unavailable": CarStatus.SOLD}
YES = {"yes", "y", "true", "1", "full", "available", "fsh", "full service history"}
NO = {"no", "n", "false", "0", "none", "partial"}


def heading_key(h: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(h).lower())


def map_headings(headings: list[str], mapping: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """({their heading: canonical field}, the headings not recognised). `mapping` wins over the synonyms."""
    given = {heading_key(k): v for k, v in (mapping or {}).items()}
    out, unknown, taken = {}, [], set()
    for h in headings:
        canon = given.get(heading_key(h)) or _LOOKUP.get(heading_key(h))
        if canon and canon not in taken:
            out[h] = canon
            taken.add(canon)
        elif str(h).strip():
            unknown.append(str(h))
    return out, unknown


class RowError(ValueError):
    """This row cannot be imported (the message says why, in the dealership's words)."""


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "-", "n/a", "na", "null", "none", "unknown", "?"))


def _int(v, what: str) -> int | None:
    if _blank(v):
        return None
    if isinstance(v, (int, float)):
        return int(round(v))
    s = str(v).lower().replace(",", "").replace("aed", "").replace("dhs", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?\s*k?\b", s)
    if not m:
        raise RowError(f"{what} {v!r} is not a number")
    num = m.group().strip()
    return int(round(float(num[:-1]) * 1000)) if num.endswith("k") else int(round(float(num)))


def _float(v, what: str) -> float | None:
    if _blank(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"\d+(?:\.\d+)?", str(v).replace(",", "."))
    if not m:
        raise RowError(f"{what} {v!r} is not a number")
    return float(m.group())


def _text(v) -> str | None:
    return None if _blank(v) else str(v).strip()


def _list(v) -> tuple[str, ...]:
    if _blank(v):
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(str(x).strip() for x in v if str(x).strip())
    return tuple(x.strip() for x in re.split(r"[;|\n]|,(?=\s*[A-Z])", str(v)) if x.strip())


def _yes_no(v) -> bool | None:
    if _blank(v):
        return None
    s = str(v).strip().lower()
    return True if s in YES or s.startswith("full") else False if s in NO else None


def _plain_colour(name: str | None, hex_code: str | None) -> str:
    """The plain colour a caller would say ("white" for "Pearl White"). Without a colour word or a swatch, the name as it is, never a guess."""
    if not name:
        return ""
    if hex_code or any(w in COLOUR_WORDS for w in re.findall(r"[a-z]+", name.lower())):
        return plain_colour(name, hex_code or "#808080")
    return name.lower()


@dataclass
class Branches:
    """Which branch a row's car is at: the row's "branch" by id or by name; otherwise the default one."""
    default: str
    by_name: dict[str, str] = field(default_factory=dict)          # lower-cased name or id -> id

    def find(self, v) -> str:
        if _blank(v):
            return self.default
        key = str(v).strip().lower()
        hit = self.by_name.get(key) or next((bid for name, bid in self.by_name.items() if key in name or name in key), None)
        if hit is None:
            raise RowError(f"branch {v!r} is not one of {sorted(set(self.by_name.values()))}")
        return hit


def to_record(row: dict, headings: dict[str, str], branches: Branches) -> tuple[Car, CarDetails]:
    """One stock-list row (their headings) as a Car and its spec sheet. Raises RowError when a required value is missing or unreadable."""
    v = {canon: row.get(h) for h, canon in headings.items()}
    missing = [f for f in REQUIRED if _blank(v.get(f))]
    if missing:
        raise RowError("no " + ", ".join(m.replace("_", " ") for m in missing))
    sid = str(v["stock_id"]).strip()
    body_text = str(v["body_type"]).strip().lower()
    body = BODY_WORDS.get(body_text) or next((b for w, b in BODY_WORDS.items() if w in body_text), None)
    if body is None:
        raise RowError(f"body type {v['body_type']!r} is not one of SUV, sedan, hatchback, coupe, pickup, van")
    year, price, km = _int(v["year"], "year"), _int(v["price"], "price"), _int(v["km"], "mileage")
    if not 1950 <= year <= 2100:
        raise RowError(f"year {v['year']!r} is not a year")
    if price <= 0:
        raise RowError(f"price {v['price']!r} is not a price")
    floor = _int(v.get("floor_price"), "floor price") or price
    if not 0 < floor <= price:
        raise RowError(f"floor price {v.get('floor_price')!r} is above the price")
    make = str(v["make"]).strip()
    make = MAKE.get(make.lower(), make)
    model, trim = str(v["model"]).strip(), _text(v.get("trim"))
    full = f"{model} {trim}" if trim else model
    status_text = _text(v.get("status"))
    status = STATUS_WORDS.get(status_text.lower()) if status_text else CarStatus.AVAILABLE
    if status is None:
        raise RowError(f"status {status_text!r} is not available, reserved or sold")
    colour, hex_code = _text(v.get("colour_name")), _text(v.get("colour_hex"))
    fuel = (_text(v.get("fuel")) or "").lower()
    gearbox = _text(v.get("transmission"))
    service = _text(v.get("service"))
    car = Car(stock_id=sid, vin=_text(v.get("vin")) or "", make=make, model=model if trim else base_model(model), year=year, km=km, price=price, floor_price=floor,
              status=status, branch_id=branches.find(v.get("branch")), body_type=body, fuel=fuel, transmission=(gearbox or "").lower(),
              color=_plain_colour(colour, hex_code), regional_spec=_text(v.get("spec")),
              service_history=_yes_no(service), image_urls=_list(v.get("images")))
    details = CarDetails(stock_id=sid, full_model=full, colour_name=colour, colour_hex=hex_code, seats=_int(v.get("seats"), "seats"),
                         power_hp=_int(v.get("power_hp"), "power"), torque_nm=_int(v.get("torque_nm"), "torque"), engine=_text(v.get("engine")),
                         drive=_text(v.get("drive")), accel_0_100=_float(v.get("accel_0_100"), "acceleration"), top_speed=_int(v.get("top_speed"), "top speed"),
                         consumption=_text(v.get("consumption")), boot_litres=_int(v.get("boot_litres"), "boot"), interior=_text(v.get("interior")),
                         owners=_int(v.get("owners"), "owners"), features=_list(v.get("features")), blurb=_text(v.get("blurb")), transmission=gearbox,
                         doors=_int(v.get("doors"), "doors"), spec=_text(v.get("spec")), accident=_text(v.get("accident")), service=service,
                         warranty=_text(v.get("warranty")), inspection=_int(v.get("inspection"), "inspection"), condition=_text(v.get("condition")),
                         market_price=_int(v.get("market_price"), "market price"))
    return car, details
