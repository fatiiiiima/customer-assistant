"""The website's car API: read-only, straight from the database the voice agent reads, so the page and the agent always show the same cars.

  GET /api/cars?make=&model=&body=&fuel=&color=&max_price=&min_price=&year_min=&max_km=&min_seats=&sort=price|power&limit=&offset=
      {"total": n, "cars": [...]}: a page of matching cars on sale, each in the shape the pages use today (window.CARS)
  GET /api/cars/{id}      one car (any status) with its spec sheet; 404 when there is none
  GET /api/showroom       hours, locations, phone, WhatsApp, email
  GET /api/site-cars.js   `window.CARS = [...]` (every car on sale) and the showroom details into window.CONFIG: the pages as they are
                          today load this one script after data.js and change nothing else (ETag: the stock's version)
  GET /api/health         how many cars, the stock's version, when it was last imported

Never in a reply: the floor price and the VIN. A field the spec sheet does not record is null; the page shows "Not recorded".
"""
from __future__ import annotations

import json
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from ..models import BodyType, Car, CarStatus, SearchFilters
from ..site.details import CarDetails

BODY_LABEL = {BodyType.SUV: "SUV", BodyType.SEDAN: "Sedan", BodyType.HATCHBACK: "Hatchback", BodyType.COUPE: "Coupe", BodyType.PICKUP: "Pickup",
              BodyType.VAN: "Van"}
SITE_MAKE = {"Mercedes": "Mercedes-Benz"}          # the voice agent says "Mercedes" (site/portal.py MAKE); the page shows the badge
MAX_PAGE = 200


def site_car(car: Car, d: CarDetails | None) -> dict:
    """One car as the website's pages use it (the keys of data.js's car()); never the floor price or the VIN."""
    d = d or CarDetails(car.stock_id)
    return {"id": car.stock_id, "make": SITE_MAKE.get(car.make, car.make), "model": d.full_model or car.model, "year": car.year, "price": car.price,
            "mileage": car.km, "body": BODY_LABEL[car.body_type], "fuel": car.fuel.title() if car.fuel else None, "seats": d.seats,
            "colorName": d.colour_name or (car.color.title() if car.color else None), "color": d.colour_hex, "power": d.power_hp, "torque": d.torque_nm,
            "engine": d.engine, "drive": d.drive, "accel": d.accel_0_100, "top": d.top_speed, "cons": d.consumption, "boot": d.boot_litres,
            "interior": d.interior, "owners": d.owners, "features": list(d.features), "blurb": d.blurb, "condition": d.condition,
            "transmission": d.transmission or (car.transmission.title() if car.transmission else None), "doors": d.doors, "spec": d.spec,
            "accident": d.accident, "service": d.service, "market": d.market_price, "warranty": d.warranty, "inspection": d.inspection,
            "status": car.status.value, "branch": car.branch_id, "images": list(car.image_urls)}


def create_api(repo, origins: list[str] | None = None) -> FastAPI:
    api = FastAPI(title="AGMC stock", docs_url=None, redoc_url=None)
    api.add_middleware(CORSMiddleware, allow_origins=origins or ["http://localhost:8080", "http://127.0.0.1:8080"], allow_methods=["GET"],
                       allow_headers=["*"], expose_headers=["ETag"])
    cache: dict = {"version": None, "body": b""}
    lock = threading.Lock()

    def every_car_on_sale() -> list[Car]:
        f = SearchFilters(include_unavailable=True)
        cars = [c for c in repo.search(f, repo.count(f) or 1) if c.status is not CarStatus.SOLD]
        return sorted(cars, key=lambda c: c.stock_id)

    @api.get("/api/cars")
    def cars(make: str | None = None, model: str | None = None, body: str | None = None, fuel: str | None = None, color: str | None = None,
             max_price: int | None = None, min_price: int | None = None, year_min: int | None = None, max_km: int | None = None,
             min_seats: int | None = None, sort: str = "price", limit: int = 50, offset: int = 0) -> dict:
        try:
            body_type = BodyType(body.lower()) if body else None
        except ValueError:
            raise HTTPException(400, f"body must be one of {[b.value for b in BodyType]}")
        if sort not in ("price", "power"):
            raise HTTPException(400, "sort must be price or power")
        f = SearchFilters(make=make, model=model, body_type=body_type, fuel=fuel, color=color, max_price=max_price, min_price=min_price,
                          year_min=year_min, max_km=max_km, min_seats=min_seats, sort=sort)
        found = repo.search(f, max(1, min(limit, MAX_PAGE)), max(0, offset))
        details = repo.details_many([c.stock_id for c in found])
        return {"total": repo.count(f), "cars": [site_car(c, details.get(c.stock_id)) for c in found]}

    @api.get("/api/cars/{stock_id}")
    def one_car(stock_id: str) -> dict:
        car = repo.get(stock_id)
        if car is None:
            raise HTTPException(404, f"no car {stock_id}")
        return site_car(car, repo.details(stock_id))

    @api.get("/api/showroom")
    def showroom() -> dict:
        return repo.showroom().plain()

    @api.get("/api/health")
    def health() -> dict:
        n = repo.count(SearchFilters())
        return {"cars_on_sale": n, "version": repo.version(), "synced_at": repo.meta("synced_at")}

    @api.get("/api/site-cars.js")
    def site_cars(request: Request) -> Response:
        version = repo.version()
        etag = f'"stock-{version}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        with lock:                                                      # built once per version of the stock, not once per page view
            if cache["version"] != version:
                cars = every_car_on_sale()
                details = repo.details_many([c.stock_id for c in cars])
                room = repo.showroom()
                config = {"hours": {"open": room.open_hour, "close": room.close_hour}, "location": room.location,
                          "locations": [{"name": x.name, "addr": x.address, "hours": x.hours} for x in room.locations]}
                if room.deposit:
                    config["depositAmount"] = room.deposit
                body = ("// AGMC stock from the database (version %d). Generated: do not edit.\n"
                        "window.CARS = %s;\n"
                        "window.CONFIG = Object.assign(window.CONFIG || {}, %s);\n"
                        "window.CONFIG.contact = Object.assign(window.CONFIG.contact || {}, %s);\n") % (
                    version, json.dumps([site_car(c, details.get(c.stock_id)) for c in cars], ensure_ascii=False),
                    json.dumps(config, ensure_ascii=False),
                    json.dumps({k: v for k, v in (("phone", room.phone), ("whatsapp", room.whatsapp), ("email", room.email)) if v}, ensure_ascii=False))
                cache.update(version=version, body=body.encode("utf-8"))
            data = cache["body"]
        return Response(data, media_type="application/javascript; charset=utf-8", headers={"ETag": etag, "Cache-Control": "no-cache"})

    return api
