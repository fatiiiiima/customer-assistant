"""A small, hand-written FAKE inventory for demos and scripted scenarios (deterministic, easy to talk about).

Every value is invented. `S1042` is the Land Cruiser the test sets and scenarios refer to. Floors are PLACEHOLDERS
(TO BE SET BY THE BUSINESS). Use `seed_demo_inventory(repo)` instead of the random 60-car seed when you want the same
cars every time.
"""
from __future__ import annotations

from ..models import BodyType as B
from ..models import Car, CarStatus as S
from .sqlite_store import SqliteInventoryRepo

# stock, make, model, year, km, price, floor, branch, body, color, status
_ROWS = [
    ("S1042", "Toyota", "Land Cruiser", 2021, 62_000, 148_000, 141_000, "DXB-01", B.SUV, "white", S.AVAILABLE),
    ("S1050", "Nissan", "Patrol", 2020, 87_500, 138_500, 131_500, "DXB-01", B.SUV, "black", S.AVAILABLE),
    ("S1051", "Hyundai", "Santa Fe", 2021, 54_000, 112_000, 106_500, "SHJ-01", B.SUV, "grey", S.AVAILABLE),
    ("S1052", "Kia", "Sorento", 2019, 98_000, 89_500, 85_000, "DXB-01", B.SUV, "silver", S.AVAILABLE),
    ("S1053", "Toyota", "Prado", 2018, 121_000, 105_000, 99_500, "SHJ-01", B.SUV, "blue", S.AVAILABLE),
    ("S1054", "Mitsubishi", "Pajero", 2017, 143_000, 72_000, 68_500, "DXB-01", B.SUV, "brown", S.AVAILABLE),
    ("S1055", "Range Rover", "Sport", 2022, 41_000, 318_000, 303_000, "DXB-01", B.SUV, "black", S.AVAILABLE),
    ("S1056", "Lexus", "RX", 2020, 66_000, 164_000, 156_000, "DXB-01", B.SUV, "white", S.AVAILABLE),
    ("S1060", "Toyota", "Camry", 2021, 58_000, 78_500, 74_500, "DXB-01", B.SEDAN, "white", S.AVAILABLE),
    ("S1061", "Honda", "Accord", 2020, 71_000, 69_000, 65_500, "SHJ-01", B.SEDAN, "grey", S.AVAILABLE),
    ("S1062", "Nissan", "Altima", 2019, 92_000, 52_500, 49_500, "DXB-01", B.SEDAN, "red", S.AVAILABLE),
    ("S1063", "Mercedes", "C Class", 2019, 77_000, 118_000, 112_000, "DXB-01", B.SEDAN, "black", S.AVAILABLE),
    ("S1070", "Ford", "Mustang", 2018, 95_000, 120_000, 114_000, "DXB-01", B.COUPE, "red", S.AVAILABLE),
    ("S1071", "Toyota", "Hilux", 2020, 99_000, 85_000, 80_500, "DXB-01", B.PICKUP, "silver", S.AVAILABLE),
    ("S1080", "Nissan", "Patrol", 2019, 101_000, 129_000, 122_500, "SHJ-01", B.SUV, "white", S.RESERVED),
    ("S1081", "Kia", "Sportage", 2021, 47_000, 96_000, 91_000, "DXB-01", B.SUV, "grey", S.SOLD),
]


def demo_cars() -> list[Car]:
    out = []
    for stock, make, model, year, km, price, floor, branch, body, color, status in _ROWS:
        out.append(Car(stock_id=stock, vin=f"DEMO{stock}0000000", make=make, model=model, year=year, km=km, price=price,
                       floor_price=floor, status=status, branch_id=branch, body_type=body, color=color,
                       fuel="diesel" if body is B.PICKUP else "petrol", service_history=stock not in ("S1054", "S1062"),
                       image_urls=tuple(f"https://cars.example.invalid/{stock}/{k}.jpg" for k in (1, 2, 3))))
    return out


def seed_demo_inventory(repo: SqliteInventoryRepo) -> list[Car]:
    cars = demo_cars()
    for c in cars:
        repo.add(c)
    return cars
