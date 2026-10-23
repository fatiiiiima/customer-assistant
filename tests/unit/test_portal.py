"""The website's inventory (read from its data.js), the portal config overlay, and fixed prices. Skips the real-file tests if the site is not on this machine."""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.app import build_app
from dealer_core.app import ROOT
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent
from dealer_core.nlu.backend import IntentResult
from dealer_core.site.portal import base_model, parse_cars, plain_colour

DATA_JS = Path("C:/dev/bmw-portal/site/data.js")
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=ZoneInfo("Asia/Dubai"))


class NoLlm:
    name = "none"

    def classify(self, state, car, text, last_agent=""):
        return IntentResult(Intent.MAKE_OFFER if "250000" in text else Intent.SEARCH_CARS, 1.0, self.name, "{}")


def test_base_model_drops_trims():
    assert base_model("Land Cruiser GXR") == "Land Cruiser"
    assert base_model("Camry Hybrid") == "Camry"
    assert base_model("Range Rover Sport SVR") == "Range Rover Sport"
    assert base_model("Mustang GT") == "Mustang"
    assert base_model("Macan S") == "Macan"


def test_plain_colour():
    assert plain_colour("Pearl White / Red roof", "#f00") == "white"
    assert plain_colour("Hella Yella", "#e9d12a") == "yellow"
    assert plain_colour("Black Sapphire", "#1b1d20") == "black"
    assert plain_colour("Something Odd", "#1e2a44") in {"blue", "black"}


def test_parse_line():
    js = 'car("AG-1", "Mercedes-Benz", "C 300 AMG Line", 2022, 178000, 28000, "Sedan", "Petrol", 5, "Polar White", "#f1f1ee", 255, 400)'
    (c,) = parse_cars(js, "X")
    assert (c.make, c.model, c.price, c.floor_price, c.color, c.body_type.value) == ("Mercedes", "C 300", 178000, 178000, "white", "sedan")


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_every_site_car_loads():
    assert len(parse_cars(DATA_JS.read_text(encoding="utf-8"), "X")) == 24


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_portal_app_fixed_price_and_booking_hours():
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    assert [b.id for b in cfg.branches] == ["AGMC-HQ"] and not cfg.negotiation.enabled
    app = build_app(cfg, inventory="portal", portal_data=DATA_JS, classifier=NoLlm(), reply_mode="template", clock=lambda: NOW.astimezone(timezone.utc))
    assert len(app.inventory.vocabulary()) >= 15
    s = app.engine.new_session()
    r = app.engine.handle(s, "do you have a toyota")
    assert "Toyota" in r.reply or "toyota" in r.reply.lower()
    s2 = app.engine.new_session()
    app.engine.handle(s2, "tell me about the bmw m3")
    r = app.engine.handle(s2, "can you do 250000 for it")
    assert "fixed" in r.reply and "289,000" in r.reply


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_partial_model_names_and_misheard_bmw():
    from dealer_core.nlu import extractors as ex

    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=DATA_JS, classifier=NoLlm(), reply_mode="template")
    v = app.inventory.vocabulary()
    assert ex.search_filters("Mercedes GLE", v) == {"make": "mercedes", "model": "gle 450"}
    assert ex.search_filters("the GLE please", v)["model"] == "gle 450"
    assert ex.search_filters("a range rover", v).get("make") == "land rover"
    for heard in ("B&W, X5.", "BNW X5", "b and w x5"):                     # Whisper's mishearings of "BMW X5"
        assert ex.search_filters(heard, v) == {"make": "bmw", "model": "x5"}, heard
    assert "model" not in ex.search_filters("a toyota", v)                  # a make alone stays a make
    assert ex.search_filters("NEDMW Model", v) == {}                         # "model" alone is not the Tesla Model 3
    assert ex.search_filters("in that price range", v) == {}                 # "range" alone is not a Range Rover
