"""The user-journey load script (scripts/user_journeys.py): what a browsing agent fetches, and how the results are summed."""
import asyncio
import importlib.util
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("user_journeys", ROOT / "scripts" / "user_journeys.py")
uj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uj)

HOME = ('<link href="data:image/svg+xml,%3Csvg%3E"><link href="https://fonts.googleapis.com"><link href="styles.css"><a href="cars.html">'
        '<img src="img/AG-122.jpg"><img src="img/AG-114.jpg"><script src="data.js"></script><script src="//cdn.example.com/x.js"></script><a href="#top">')


def test_a_page_asks_only_for_its_own_files():
    assert uj.local_assets(HOME) == ["data.js", "img/AG-114.jpg", "img/AG-122.jpg", "styles.css"]          # no other site, no data: address, no page links, no anchors


def test_the_stock_numbers_come_from_the_pictures():
    assert uj.car_ids(HOME) == ["AG-114", "AG-122"]


def test_a_first_visit_loads_the_files_and_a_second_only_the_page():
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return httpx.Response(200, text=HOME, headers={"content-type": "text/html"}) if request.url.path in ("/", "/cars.html", "/brand.html", "/car.html") else httpx.Response(200, content=b"x")

    async def go():
        log, seen = uj.Log(), set()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://site") as c:
            uj.random.seed(1)
            await uj.browse_once(c, log, "http://site", seen, ["AG-122"])
            first = len(asked)
            await uj.browse_once(c, log, "http://site", seen, ["AG-122"])
            return log, first, len(asked) - first
    log, first, second = asyncio.run(go())
    assert first == 5 and second == 1                                                   # the page and its four files; then just the page
    assert [r["kind"] for r in log.rows].count("asset") == 4 and all(r["ok"] for r in log.rows)


def test_a_failed_page_is_a_result_not_a_crash():
    def handler(request):
        raise httpx.ConnectError("down")

    async def go():
        log = uj.Log()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            text = await uj.get(c, log, "http://site/", "page")
        return text, log.rows
    text, rows = asyncio.run(go())
    assert text == "" and rows[0]["ok"] is False and rows[0]["error"] == "ConnectError"


def test_the_summary_counts_calls_waits_and_bookings_by_kind():
    rows = [{"kind": "page", "what": "/", "s": 0.01, "ok": True}, {"kind": "page", "what": "/", "s": 0.03, "ok": False},
            {"kind": "talk", "what": "call", "s": 40.0, "ok": True, "status": "ok", "waits": [0.4, 0.6, 2.5], "booked": False, "turns": 3},
            {"kind": "talk", "what": "call", "s": 1.0, "ok": False, "status": "busy", "waits": [], "booked": False, "turns": 0},
            {"kind": "book", "what": "call", "s": 50.0, "ok": True, "status": "ok", "waits": [0.5, 0.7], "booked": True, "turns": 2},
            {"kind": "book", "what": "assign", "s": 0.01, "ok": False, "error": "HTTPError"}]
    t = uj.table(rows)
    assert t["page"]["requests"] == 2 and t["page"]["failed"] == 1
    assert t["talk"]["calls"] == 2 and t["talk"]["completed"] == 1 and t["talk"]["busy"] == 1 and t["talk"]["failed"] == 0
    assert t["talk"]["wait_median_s"] == 0.6 and round(t["talk"]["over_2s"], 2) == 0.33
    assert t["book"]["booked"] == 1 and t["book"]["failed"] == 1                         # the refused assignment counts as a failure


def test_every_booking_conversation_is_different_and_carries_the_test_number():
    import random

    rng = random.Random(7)
    calls = [uj.unique_booking_lines(rng) for _ in range(30)]
    assert len({tuple(c) for c in calls}) == 30                                          # no two alike
    for c in calls:
        assert c[0].startswith("Hello") and c[2] == "Can I book a test drive for it?" and c[-1] == "Yes that is right"
        words = c[5].replace("My number is ", "").split()
        digits = "".join(str(uj.DIGITS.index(w)) for w in words)
        assert len(digits) == 10 and digits.startswith(uj.TEST_PHONE_PREFIX)             # 05599 and five more
