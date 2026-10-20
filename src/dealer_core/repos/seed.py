"""FAKE inventory generator (deterministic). Every value is invented: prices, floors, VINs, URLs.

Real stock replaces this by implementing the repository Protocols; nothing here is used outside tests and the demo CLI.
"""
from __future__ import annotations

import random

from ..models import BodyType as B
from ..models import Car, CarStatus
from .sqlite_store import SqliteInventoryRepo

# (make, model, body type, approximate new price in AED). A generic UAE-market list, not tied to any dealer.
TEMPLATES: list[tuple[str, str, B, int]] = [
    ("Toyota", "Land Cruiser", B.SUV, 320_000), ("Toyota", "Camry", B.SEDAN, 115_000), ("Toyota", "Corolla", B.SEDAN, 85_000),
    ("Toyota", "Prado", B.SUV, 230_000), ("Toyota", "Hilux", B.PICKUP, 125_000), ("Nissan", "Patrol", B.SUV, 290_000),
    ("Nissan", "Altima", B.SEDAN, 95_000), ("Nissan", "Sunny", B.SEDAN, 55_000), ("Honda", "Accord", B.SEDAN, 120_000),
    ("Honda", "Civic", B.SEDAN, 90_000), ("Honda", "CR-V", B.SUV, 130_000), ("Hyundai", "Elantra", B.SEDAN, 80_000),
    ("Hyundai", "Tucson", B.SUV, 110_000), ("Hyundai", "Santa Fe", B.SUV, 150_000), ("Kia", "Sportage", B.SUV, 105_000),
    ("Kia", "Sorento", B.SUV, 140_000), ("Kia", "Cerato", B.SEDAN, 75_000), ("Mitsubishi", "Pajero", B.SUV, 150_000),
    ("Mazda", "CX-5", B.SUV, 115_000), ("Lexus", "RX", B.SUV, 260_000), ("Mercedes", "C Class", B.SEDAN, 210_000),
    ("Mercedes", "E Class", B.SEDAN, 290_000), ("BMW", "X5", B.SUV, 320_000), ("Ford", "Mustang", B.COUPE, 190_000),
    ("Jeep", "Wrangler", B.SUV, 180_000), ("Chevrolet", "Tahoe", B.SUV, 250_000), ("Porsche", "Cayenne", B.SUV, 380_000),
    ("Range Rover", "Sport", B.SUV, 420_000),
]
COLORS = ["white", "black", "silver", "grey", "blue", "red", "brown"]
VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"      # real VINs exclude I, O and Q


def generate_cars(branch_ids: list[str], n: int = 60, seed: int = 7, current_year: int = 2026) -> list[Car]:
    rng = random.Random(seed)
    cars: list[Car] = []
    for i in range(n):
        make, model, body, new_price = TEMPLATES[i % len(TEMPLATES)]
        year = rng.randint(2015, 2025)
        age = current_year - year
        km = int(max(3_000, age * rng.randint(9_000, 22_000)))
        # Simple depreciation: ~11%/year plus a small kilometre penalty, with noise. Rounded to 500 AED.
        price = new_price * (0.89 ** age) * (1 - min(0.25, km / 900_000)) * rng.uniform(0.92, 1.08)
        price = max(12_000, int(round(price / 500) * 500))
        floor = int(round(price * rng.uniform(0.93, 0.96) / 100) * 100)      # PLACEHOLDER floors (TO BE SET BY THE BUSINESS)
        roll = rng.random()
        status = CarStatus.AVAILABLE if roll < 0.80 else CarStatus.RESERVED if roll < 0.90 else CarStatus.SOLD
        stock = f"S{1000 + i}"
        cars.append(Car(
            stock_id=stock, vin="".join(rng.choice(VIN_CHARS) for _ in range(17)), make=make, model=model, year=year, km=km,
            price=price, floor_price=min(floor, price), status=status, branch_id=rng.choice(branch_ids), body_type=body,
            fuel="petrol" if body is not B.PICKUP else "diesel", color=rng.choice(COLORS), service_history=rng.random() < 0.8,
            image_urls=tuple(f"https://cars.example.invalid/{stock}/{k}.jpg" for k in range(1, 4))))
    # The labelled test sets and scenarios refer to S1042: make sure it exists and can be sold.
    cars = [c.model_copy(update={"status": CarStatus.AVAILABLE}) if c.stock_id == "S1042" else c for c in cars]
    return cars


def seed_inventory(repo: SqliteInventoryRepo, branch_ids: list[str], n: int = 60, seed: int = 7) -> list[Car]:
    cars = generate_cars(branch_ids, n, seed)
    for c in cars:
        repo.add(c)
    return cars
