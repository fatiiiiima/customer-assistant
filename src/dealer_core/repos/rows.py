"""What the SQLite and PostgreSQL stores share about the spec sheets: the columns of car_details and the conversion of a row.

Every column but stock_id may be NULL: the stock list the dealership keeps did not have it, and the agent says it is not recorded."""
from __future__ import annotations

import json

from ..site.details import CarDetails

DETAIL_COLS = ("full_model", "colour_name", "colour_hex", "seats", "power_hp", "torque_nm", "engine", "drive", "accel_0_100", "top_speed",
               "consumption", "boot_litres", "interior", "owners", "features", "blurb", "transmission", "doors", "spec", "accident", "service",
               "warranty", "inspection", "condition", "market_price")
INT_DETAILS = {"seats", "power_hp", "torque_nm", "top_speed", "boot_litres", "owners", "doors", "inspection", "market_price"}


def detail_columns_sql(integer: str, real: str) -> str:
    """The column list of CREATE TABLE car_details, in each database's own type names."""
    return ", ".join(f"{c} {integer if c in INT_DETAILS else real if c == 'accel_0_100' else 'TEXT'}" for c in DETAIL_COLS)


def detail_values(d: CarDetails) -> list:
    """The values of DETAIL_COLS for one car, in order (the features as a JSON list)."""
    return [json.dumps(list(d.features)) if c == "features" else getattr(d, c) for c in DETAIL_COLS]


def details_from_row(r) -> CarDetails:
    raw = {c: r[c] for c in DETAIL_COLS}
    raw["features"] = tuple(json.loads(raw["features"] or "[]"))
    if raw["accel_0_100"] is not None:
        raw["accel_0_100"] = float(raw["accel_0_100"])
    return CarDetails(stock_id=r["stock_id"], **raw)
