"""The website's cars, read straight from its `site/data.js`, so the voice agent talks about the SAME cars the page shows.

data.js builds each car with  car("AG-101", "Toyota", "Land Cruiser GXR", 2018, 189000, 92000, "SUV", "Petrol", 8, "Pearl White", "#ecebe6", ...).
Only the first eleven arguments are needed here. There is no JavaScript engine on this machine, so the call is read with a regular expression
(and the test fails loudly if the site's data changes shape).

Differences between the site's cars and ours, and how they are bridged:
  * the site's model names carry trims ("Land Cruiser GXR", "Camry Hybrid"): a caller says "Land Cruiser", so the model is the base name
  * "Mercedes-Benz" is said "Mercedes"
  * colours are marketing names ("Pearl White", "Hella Yella"): reduced to a plain colour, by name and otherwise by the nearest of a few base colours
  * no floor price: the site has no negotiation, so floor = list price (the policy engine is switched off in the portal profile)
"""
from __future__ import annotations

import re
from pathlib import Path

from ..models import BodyType, Car, CarStatus
from ..repos.sqlite_store import SqliteInventoryRepo

CAR_CALL = re.compile(
    r'\bcar\(\s*"(?P<id>[^"]+)"\s*,\s*"(?P<make>[^"]+)"\s*,\s*"(?P<model>[^"]+)"\s*,\s*(?P<year>\d{4})\s*,\s*(?P<price>\d+)\s*,\s*(?P<km>\d+)\s*,'
    r'\s*"(?P<body>[^"]+)"\s*,\s*"(?P<fuel>[^"]+)"\s*,\s*(?P<seats>\d+)\s*,\s*"(?P<color>[^"]+)"\s*,\s*"(?P<hex>#[0-9a-fA-F]{3,8})"')

BODY = {"suv": BodyType.SUV, "sedan": BodyType.SEDAN, "hatchback": BodyType.HATCHBACK, "coupe": BodyType.COUPE, "van": BodyType.VAN,
        "pickup": BodyType.PICKUP}
MAKE = {"mercedes-benz": "Mercedes"}

# Words after which a model name is only trim / package ("Land Cruiser GXR", "Camry Hybrid", "C 300 AMG Line"). Base model = the words before the first one.
TRIM_WORDS = {"gxr", "hybrid", "desert", "edition", "rs", "gt", "limited", "sxl", "amg", "xse", "f", "s", "svr", "long", "m", "competition", "gti",
              "rubicon", "4matic", "tfsi", "xdrive40i", "sport", "40", "55", "gt", "line"}
# Exceptions where "sport" / "range" are part of the base name.
BASE_OVERRIDES = {"range rover sport svr": "Range Rover Sport", "model 3 long range": "Model 3", "x5 xdrive40i": "X5", "c 300 amg line": "C 300",
                  "gle 450 4matic": "GLE 450", "rx 350 f sport": "RX 350", "320i m sport": "320i", "530i m sport": "530i", "a4 40 tfsi": "A4",
                  "q7 55 tfsi": "Q7", "m3 competition": "M3", "rav4 hybrid xse": "RAV4"}

BASE_COLOURS = {"white": (240, 240, 238), "black": (24, 26, 30), "grey": (128, 134, 140), "silver": (190, 194, 198), "red": (170, 30, 40),
                "blue": (30, 70, 160), "green": (60, 130, 60), "yellow": (230, 205, 50), "brown": (100, 70, 50), "beige": (210, 195, 160),
                "orange": (230, 120, 30), "gold": (200, 165, 80)}
COLOUR_WORDS = {"white": "white", "black": "black", "grey": "grey", "gray": "grey", "silver": "silver", "red": "red", "blue": "blue", "green": "green",
                "yellow": "yellow", "yella": "yellow", "brown": "brown", "bronze": "brown", "beige": "beige", "orange": "orange", "gold": "gold",
                "sapphire": "black", "ebony": "black", "iridium": "silver", "florett": "silver", "celestite": "grey", "mocha": "brown"}


def base_model(full: str) -> str:
    key = full.strip().lower()
    if key in BASE_OVERRIDES:
        return BASE_OVERRIDES[key]
    words = full.split()
    keep: list[str] = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in TRIM_WORDS:
            break
        keep.append(w)
    return " ".join(keep) or full


def plain_colour(name: str, hex_code: str) -> str:
    lowered = name.lower()
    if "pearl white" in lowered or "polar white" in lowered or "white" in lowered:
        return "white"                                            # "Pearl White / Red roof" is a white car
    for word in re.findall(r"[a-z]+", lowered):
        if word in COLOUR_WORDS:
            return COLOUR_WORDS[word]
    h = hex_code.lstrip("#")
    h = "".join(c * 2 for c in h) if len(h) in (3, 4) else h
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return min(BASE_COLOURS, key=lambda k: sum((a - b) ** 2 for a, b in zip(rgb, BASE_COLOURS[k])))


def parse_cars(js_text: str, branch_id: str) -> list[Car]:
    cars: list[Car] = []
    for m in CAR_CALL.finditer(js_text):
        body = BODY.get(m["body"].strip().lower())
        if body is None:
            raise ValueError(f"unknown body type {m['body']!r} for {m['id']}: add it to site/portal.py BODY")
        make = MAKE.get(m["make"].lower(), m["make"])
        cars.append(Car(stock_id=m["id"], vin=f"SITE{m['id'].replace('-', '')}0000", make=make, model=base_model(m["model"]), year=int(m["year"]),
                        km=int(m["km"]), price=int(m["price"]), floor_price=int(m["price"]), status=CarStatus.AVAILABLE, branch_id=branch_id,
                        body_type=body, fuel=m["fuel"].strip().lower(), transmission="single-speed" if m["fuel"].lower() == "electric" else "automatic",
                        color=plain_colour(m["color"], m["hex"]), regional_spec="GCC", service_history=True))
    return cars


def load_portal_inventory(data_js: Path, repo: SqliteInventoryRepo, branch_id: str) -> list[Car]:
    cars = parse_cars(data_js.read_text(encoding="utf-8"), branch_id)
    if not cars:
        raise ValueError(f"no cars found in {data_js}: has the site's data.js changed shape? (expected car(\"AG-101\", \"Make\", \"Model\", year, price, km, ...))")
    for c in cars:
        repo.add(c)
    return cars
