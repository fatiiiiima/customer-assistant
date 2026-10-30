"""The database is the only source of cars (docs/ARCH_PLAN.md step 0, ARCH_REVIEW section 51).

Nobody knows yet what the dealership keeps its stock in, so: any stock list (CSV, Excel, JSON, the site's data.js) is imported by its headings,
whatever their spelling; what it does not say is "not recorded" and said so; an import writes only the difference, in one transaction, and
refuses a list that would take most cars off sale at once. The website reads the same database through a small API."""
import io
import json
import zipfile
from pathlib import Path

import pytest

from dealer_core.agent import answers as A
from dealer_core.config import load_config
from dealer_core.inventory import branches_of, open_db
from dealer_core.inventory.columns import Branches, map_headings
from dealer_core.inventory.sources import read_json, read_xlsx
from dealer_core.inventory.sync import import_file
from dealer_core.models import BodyType, CarStatus, SearchFilters
from dealer_core.site.details import CarDetails, parse_site

ROOT = Path(__file__).resolve().parents[2]
DATA_JS = ROOT.parent / "bmw-portal" / "site" / "data.js"
BR = Branches(default="AGMC-HQ", by_name={"agmc-hq": "AGMC-HQ", "sheikh zayed road": "AGMC-HQ", "sharjah": "AGMC-SHJ"})

SHEET = """Stock No.,Brand,Model,Trim,Year,Price (AED),Mileage,Body Style,Colour,Seats,HP,Warranty,Fuel Type,Status,Dealer notes
X-1,Toyota,Land Cruiser,GXR,2021,"AED 245,000","38,000 km",SUV,Pearl White,8,305,12 months / 20000 km,Petrol,,first owner
X-2,Nissan,Patrol,,2019,189k,61000,Station wagon,Black,7,,,Petrol,,
X-3,Mazda,CX-5,,2022,"89,900",22000,crossover,Soul Red,5,187,,Petrol,Reserved,
X-4,Mercedes-Benz,C 300,AMG Line,2020,"139,000",45000,Saloon,Obsidian Black,5,255,,petrol,,
"""


def repo():
    return open_db("")[0]


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def fleet_csv(n: int, skip: set[int] = frozenset(), price: dict[int, int] | None = None) -> str:
    rows = ["stock_id,make,model,year,price,km,body"]
    rows += [f"C-{i},Toyota,Camry,2020,{(price or {}).get(i, 70000 + i)},{30000 + i},Sedan" for i in range(n) if i not in skip]
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------------------------------------ reading any stock list
def test_headings_are_recognised_whatever_their_spelling_and_unknown_ones_are_reported():
    heads, unknown = map_headings(SHEET.splitlines()[0].split(","))
    assert heads["Stock No."] == "stock_id" and heads["Price (AED)"] == "price" and heads["Mileage"] == "km" and heads["Body Style"] == "body_type"
    assert unknown == ["Dealer notes"]
    heads, unknown = map_headings(["Ref", "Asking", "Dealer notes"], {"Asking": "price", "dealer notes": "blurb"})
    assert heads == {"Ref": "stock_id", "Asking": "price", "Dealer notes": "blurb"} and unknown == []


def test_a_csv_with_messy_values_is_imported_and_a_bad_row_is_skipped_with_its_row_number(tmp_path):
    inv = repo()
    rep = import_file(inv, write(tmp_path, "stock.csv", SHEET), BR)
    assert rep.added == ["X-1", "X-3", "X-4"] and rep.errors == [("row 3", "body type 'Station wagon' is not one of SUV, sedan, hatchback, coupe, pickup, van")]
    assert rep.unknown_headings == ["Dealer notes"]
    lc = inv.get("X-1")
    assert (lc.make, lc.model, lc.price, lc.km, lc.body_type, lc.color, lc.floor_price) == ("Toyota", "Land Cruiser", 245000, 38000, BodyType.SUV, "white", 245000)
    assert lc.regional_spec is None and lc.service_history is None             # not in the list: not recorded, not "GCC" / "yes"
    d = inv.details("X-1")
    assert (d.full_model, d.seats, d.power_hp, d.warranty, d.accident, d.engine) == ("Land Cruiser GXR", 8, 305, "12 months / 20000 km", None, None)
    assert inv.get("X-3").status is CarStatus.RESERVED                      # "Reserved" in the status column
    benz = inv.get("X-4")
    assert (benz.make, benz.model, benz.body_type) == ("Mercedes", "C 300", BodyType.SEDAN)


def test_with_a_trim_column_the_model_column_is_kept_whole(tmp_path):
    """"Patrol" + "LE Platinum" is a Patrol (the caller says "Patrol"); without a trim column "Land Cruiser GXR" is cut to "Land Cruiser"."""
    inv = repo()
    import_file(inv, write(tmp_path, "s.csv", "Ref,Make,Model,Trim,Year,Price,KM,Body\nA,Nissan,Patrol,LE Platinum,2020,1,1,SUV\nB,Lexus,LX 600,Signature,2023,1,1,SUV\n"
                                              "C,Toyota,Land Cruiser GXR,,2020,1,1,SUV\n"), BR)
    assert [inv.get(x).model for x in "ABC"] == ["Patrol", "LX 600", "Land Cruiser"]
    assert inv.details("A").full_model == "Patrol LE Platinum" and inv.details("C").full_model == "Land Cruiser GXR"


def test_a_list_without_a_required_column_writes_nothing(tmp_path):
    inv = repo()
    rep = import_file(inv, write(tmp_path, "stock.csv", "Ref,Brand,Model,Year,Price\nA,Toyota,Camry,2020,70000\n"), BR)
    assert rep.refused.startswith("no column for km, body type") and inv.count(SearchFilters(include_unavailable=True)) == 0


def _xlsx(rows: list[list]) -> bytes:
    """A minimal workbook, as Excel saves one: shared strings for text, numbers as numbers."""
    strings: list[str] = []

    def cell(ref, v):
        if isinstance(v, (int, float)):
            return f'<c r="{ref}"><v>{v}</v></c>'
        strings.append(str(v))
        return f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>'
    sheet = "".join(f'<row r="{i + 1}">' + "".join(cell(f"{chr(65 + j)}{i + 1}", v) for j, v in enumerate(r) if v is not None) + "</row>"
                    for i, r in enumerate(rows))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook {ns} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                                      '<sheets><sheet name="Stock" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                                 '<Relationship Id="rId1" Target="worksheets/stock.xml"/></Relationships>')
        z.writestr("xl/worksheets/stock.xml", f"<worksheet {ns}><sheetData>{sheet}</sheetData></worksheet>")
        z.writestr("xl/sharedStrings.xml", f"<sst {ns}>" + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>")
    return buf.getvalue()


def test_an_excel_workbook_is_read_without_installing_anything(tmp_path):
    data = _xlsx([["Stock", "Make", "Model", "Year", "Price", "KM", "Type", "Seats"], ["E-1", "Kia", "Sorento", 2022, 105000, 18000.0, "SUV", 7],
                  [None, None, None, None, None, None, None, None], ["E-2", "Kia", "Rio", 2021, 52000, 40000, "Hatchback"]])
    headings, rows = read_xlsx(data)
    assert headings[:3] == ["Stock", "Make", "Model"] and rows[0]["KM"] == 18000 and rows[1]["Seats"] is None and len(rows) == 2
    (tmp_path / "stock.xlsx").write_bytes(data)
    inv = repo()
    assert import_file(inv, tmp_path / "stock.xlsx", BR).added == ["E-1", "E-2"] and inv.details("E-1").seats == 7


def test_a_json_export_is_found_wherever_the_list_of_cars_sits():
    heads, rows = read_json(json.dumps({"meta": {"n": 1}, "data": {"vehicles": [{"id": "J-1", "make": "Kia"}, {"id": "J-2", "trim": "LX"}]}}))
    assert heads == ["id", "make", "trim"] and [r["id"] for r in rows] == ["J-1", "J-2"]


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website's data.js is not next to this repository")
def test_the_demo_data_js_imports_all_its_cars_with_the_spec_sheets_the_car_page_shows(tmp_path):
    inv = repo()
    rep = import_file(inv, DATA_JS, BR)
    site = parse_site(DATA_JS.read_text(encoding="utf-8"))
    assert len(rep.added) == 24 and inv.details_many(site.cars) == site.cars
    d = inv.details("AG-101")                                           # 189,000 x 1.06 / 500 = 400.68 -> 401 x 500
    assert (d.market_price, d.accident, d.spec, d.condition, d.colour_hex) == (200500, "Accident-free", "GCC specs", "Used", "#ecebe6")
    assert inv.showroom().phone == "800AGMC" and len(inv.showroom().locations) == 3


# ------------------------------------------------------------------------------------------------ syncing
def test_a_second_import_writes_only_the_difference(tmp_path):
    inv = repo()
    import_file(inv, write(tmp_path, "a.csv", fleet_csv(12)), BR)
    v = inv.version()
    rep = import_file(inv, write(tmp_path, "a.csv", fleet_csv(12)), BR)
    assert rep.unchanged == 12 and inv.version() == v                   # nothing changed: nothing written
    inv.mark_status("C-5", CarStatus.RESERVED)                         # a deposit taken here: the list does not know
    rep = import_file(inv, write(tmp_path, "b.csv", fleet_csv(12, skip={3}, price={7: 65000})), BR)
    assert (rep.updated, rep.gone, rep.unchanged) == (["C-7"], ["C-3"], 10)
    assert inv.get("C-7").price == 65000 and inv.get("C-3").status is CarStatus.SOLD and inv.get("C-5").status is CarStatus.RESERVED
    rep = import_file(inv, write(tmp_path, "c.csv", fleet_csv(12, price={7: 65000})), BR)
    assert rep.back_in_stock == ["C-3"] and inv.get("C-3").status is CarStatus.AVAILABLE


def test_a_list_that_would_take_most_cars_off_sale_is_refused_unless_forced(tmp_path):
    inv = repo()
    import_file(inv, write(tmp_path, "a.csv", fleet_csv(12)), BR)
    half = write(tmp_path, "half.csv", fleet_csv(12, skip=set(range(7))))
    rep = import_file(inv, half, BR)
    assert "7 of the 12 cars on sale" in rep.refused and inv.count(SearchFilters()) == 12
    assert import_file(inv, half, BR, missing="keep").gone == [] and inv.count(SearchFilters()) == 12
    assert import_file(inv, half, BR, force=True).written and inv.count(SearchFilters()) == 5


def test_a_dry_run_reports_and_writes_nothing(tmp_path):
    inv = repo()
    rep = import_file(inv, write(tmp_path, "a.csv", fleet_csv(3)), BR, dry_run=True)
    assert rep.added == ["C-0", "C-1", "C-2"] and rep.summary().startswith("DRY RUN") and inv.count(SearchFilters()) == 0 and inv.version() == 0


def test_a_sqlite_file_keeps_the_stock_between_runs(tmp_path):
    url = f"sqlite:///{tmp_path / 'stock.db'}"
    import_file(open_db(url)[0], write(tmp_path, "a.csv", fleet_csv(3)), BR)
    assert open_db(url)[0].count(SearchFilters()) == 3
    with pytest.raises(ValueError):
        open_db("mysql://somewhere/db")


def test_the_branch_is_found_by_its_name():
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    assert branches_of(cfg).find("AGMC showroom on Sheikh Zayed Road") == cfg.branches[0].id == branches_of(cfg).find(None)


# ------------------------------------------------------------------------------------------------ searching on the spec sheet, in the database
def seeded(inv=None):
    inv = inv or repo()
    from dealer_core.models import Car

    for sid, price, seats, hp, fuel in (("A", 90000, 7, 200, "petrol"), ("B", 120000, 5, 400, "petrol"), ("C", 60000, None, 150, "hybrid"), ("D", 80000, 8, None, "diesel")):
        inv.apply_sync([Car(stock_id=sid, vin="", make="Toyota", model="X", year=2020, km=1, price=price, floor_price=price, status=CarStatus.AVAILABLE,
                            branch_id="AGMC-HQ", body_type=BodyType.SUV, fuel=fuel)], [CarDetails(sid, seats=seats, power_hp=hp)], {}, None, {})
    return inv


def test_seats_power_and_fuel_are_searched_in_the_database():
    inv = seeded()
    ids = lambda **kw: [c.stock_id for c in inv.search(SearchFilters(**kw), 10)]
    assert ids(min_seats=7) == ["D", "A"]                                 # C's seats are not recorded: it does not match
    assert ids(min_power=180, sort="power") == ["B", "A"] and ids(sort="power") == ["B", "A", "C", "D"]   # not recorded: last
    assert ids(fuel="Hybrid") == ["C"] and inv.count(SearchFilters(min_seats=7, fuel="petrol")) == 1
    assert ids(sort="price")[1:3] == ["D", "A"] and [c.stock_id for c in inv.search(SearchFilters(), 2, offset=2)] == ["A", "B"]


# ------------------------------------------------------------------------------------------------ "not recorded" is said, never guessed
def view(fuel="petrol"):
    from dealer_core.models import CarView

    return CarView(stock_id="A", make="Toyota", model="Land Cruiser", year=2021, km=1, price=1, status=CarStatus.AVAILABLE, branch_id="b",
                   body_type=BodyType.SUV, fuel=fuel, transmission="", color="white", regional_spec=None, service_history=None, image_urls=())


def test_a_fact_the_spec_sheet_does_not_record_is_said_to_be_not_on_record():
    d = CarDetails("A", seats=8, accident="Accident-free", warranty="12 months / 20,000 km")
    assert A.detail_answer(view(), d, "does it have a warranty") == ["It comes with an AGMC warranty of 12 months or 20,000 kilometres."]
    assert A.detail_answer(view(), d, "what's the engine like") == ["I don't have its engine details on record for this car."]
    assert A.detail_answer(view(), d, "has it had any accidents") == ["It's accident-free."]
    assert A.detail_answer(view(), d, "does it have a sunroof") == ["I don't have its feature list on record, so I can't say whether it has a sunroof."]
    assert A.detail_answer(view(), CarDetails("A", accident="Front bumper repaired"), "any accidents") == ["Its accident record says: front bumper repaired."]


def test_tell_me_more_offers_only_what_is_recorded():
    d = CarDetails("A", seats=8, accident="Accident-free", owners=1)
    told: list[str] = []
    assert A.next_facts(d, told, limit=8) == ["It's accident-free.", "It seats 8.", "It's had 1 previous owner."]
    assert A.next_facts(d, told) == [] and A.next_facts(CarDetails("A"), []) == []


def test_comparisons_and_superlatives_leave_out_what_is_not_recorded():
    a, b = view(), view().model_copy(update={"stock_id": "B", "model": "Patrol", "price": 2})
    assert len(A.compare(a, CarDetails("A", seats=8), b, CarDetails("B"))) == 2              # the basics and the price only
    best = A.best_of([a, b], {"A": CarDetails("A"), "B": CarDetails("B", power_hp=400)}, "which one is the most powerful")
    assert best[0].stock_id == "B"


def test_the_legacy_wording_of_spec_and_service_history():
    c = view()
    assert (c.spec_said(), c.service_said()) == ("regional spec not recorded", "service history not recorded")
    c = c.model_copy(update={"regional_spec": "GCC specs", "service_history": False})
    assert (c.spec_name, c.spec_said(), c.service_said()) == ("GCC", "GCC spec", "no service history on file")


def test_the_agent_says_a_missing_warranty_is_not_on_record():
    from .test_live_call_fixes import said
    from .test_scale import agent

    e, b = agent(None)
    e.svc.inventory.put_details(CarDetails("S1042", full_model="Land Cruiser", seats=8))
    s = e.new_session()
    said(e, s, "tell me about the Land Cruiser")
    r = said(e, s, "does it have a warranty")
    assert "don't have the warranty on record" in r.reply, r.reply


# ------------------------------------------------------------------------------------------------ the website's API
@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from dealer_core.inventory.api import create_api

    inv = repo()
    import_file(inv, write(tmp_path, "stock.csv", SHEET), BR)
    inv.mark_status("X-4", CarStatus.SOLD)
    return TestClient(create_api(inv)), inv


def test_the_api_lists_and_pages_cars_on_sale_without_floor_price_or_vin(client):
    c, _ = client
    page = c.get("/api/cars", params={"limit": 1}).json()
    assert page["total"] == 1 and [x["id"] for x in page["cars"]] == ["X-1"]          # X-3 is reserved, X-4 sold
    assert c.get("/api/cars", params={"min_seats": 9}).json() == {"total": 0, "cars": []}
    one = c.get("/api/cars/X-1").json()
    assert (one["model"], one["mileage"], one["body"], one["seats"], one["accident"]) == ("Land Cruiser GXR", 38000, "SUV", 8, None)
    assert "floor_price" not in json.dumps(one) and "vin" not in one
    assert c.get("/api/cars/NOPE").status_code == 404 and c.get("/api/cars", params={"body": "boat"}).status_code == 400


def test_site_cars_js_gives_the_pages_window_cars_and_follows_the_stock(client):
    c, inv = client
    r = c.get("/api/site-cars.js")
    assert r.headers["content-type"].startswith("application/javascript") and r.text.count('"id"') == 2   # the sold car is not shown
    body = r.text.split("window.CARS = ", 1)[1].split(";\n", 1)[0]
    assert [x["id"] for x in json.loads(body)] == ["X-1", "X-3"]
    assert c.get("/api/site-cars.js", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    inv.mark_status("X-3", CarStatus.SOLD)                               # sold on the phone: the page's next load no longer shows it
    r2 = c.get("/api/site-cars.js", headers={"If-None-Match": r.headers["etag"]})
    assert r2.status_code == 200 and r2.text.count('"id"') == 1


def test_a_sqlite_stock_database_is_not_used_to_share_calls_between_workers():
    from dealer_core.server.__main__ import shared_state_url

    assert shared_state_url("", {"DATABASE_URL": "sqlite:///data/stock.db"}, "") == ""
    assert shared_state_url("", {"DATABASE_URL": "postgresql://u@h/db"}, "") == "postgresql://u@h/db"
    assert shared_state_url("", {"STATE_URL": "redis://r", "DATABASE_URL": "sqlite:///x.db"}, "") == "redis://r"
    assert shared_state_url("redis://a", {}, "postgresql://u@h/db") == "redis://a"


def test_does_it_come_with_a_warranty_is_a_question_about_the_car_not_a_visit():
    """30 Oct, a call to a worker on the stock database: "Does it come with a warranty?" got "Sure! What day would you like to come in?"."""
    from .test_live_call_fixes import said
    from .test_scale import agent

    for asked, expect in (("does it come with a warranty", "warranty of 12 months or 20,000 kilometres"), ("does it come with a sunroof", "sunroof"),
                          ("does it come with a reverse camera", "camera")):
        e, b = agent(None)
        e.svc.inventory.put_details(CarDetails("S1042", features=("Sunroof", "Reverse camera"), warranty="12 months / 20,000 km"))
        s = e.new_session()
        said(e, s, "tell me about the Land Cruiser")
        r = said(e, s, asked)
        assert r.debug.intent == "get_car" and expect in r.reply.lower(), (asked, r.reply)
    e, b = agent(None)
    s = e.new_session()
    said(e, s, "tell me about the Land Cruiser")
    assert said(e, s, "can I come and see it tomorrow").debug.intent == "prepare_booking"            # a visit is still a visit


def test_yes_to_would_you_like_to_hear_more_gives_the_car_whatever_the_model_would_pick():
    """30 Oct, a live call on Nemotron (Bedrock): "Yes." after "... the Tahoe ... Would you like to hear more about it?" got "I don't have a booking to confirm yet"."""
    from .test_live_call_fixes import said
    from .test_scale import Scripted, agent

    for answer in ("Yes.", "yeah sure", "okay", "sounds good"):
        e, _ = agent(None, backend=Scripted(["confirm_booking"] * 3))
        s = e.new_session()
        said(e, s, "tell me about the Land Cruiser")
        s.history.append(("agent", "The best for a family we have is the Land Cruiser, with 8 seats. Would you like to hear more about it?"))
        r = said(e, s, answer)
        assert r.debug.intent == "get_car" and r.debug.via == "agent:rules" and "booking to confirm" not in r.reply, (answer, r.reply)
