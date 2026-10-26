"""The PostgreSQL booking store passes the same tests as the SQLite one, against a real PostgreSQL (started locally by pgserver).

Every storage test from test_repos.py runs here too (except the two that poke SQLite directly); on top, a race across separate
connection pools (as two voice workers would be) must still give exactly one booking."""
from __future__ import annotations

import importlib.util
import tempfile
import threading
from pathlib import Path

import pytest

pgserver = pytest.importorskip("pgserver")
pytest.importorskip("pg8000")

from dealer_core.errors import SlotTaken                       # noqa: E402
from dealer_core.repos.pg_store import PgBookingRepo, PgDb, PgInventoryRepo   # noqa: E402
from dealer_core.repos.seed import seed_inventory              # noqa: E402

_spec = importlib.util.spec_from_file_location("repo_tests", Path(__file__).with_name("test_repos.py"))
repo_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repo_tests)


@pytest.fixture(scope="module")
def pg_url():
    srv = pgserver.get_server(tempfile.mkdtemp(prefix="pg_test_"), cleanup_mode="stop")
    yield srv.get_uri()


@pytest.fixture
def pgdb(pg_url):
    db = PgDb(pg_url)
    with db.tx() as c:                                           # a clean database for every test
        c.run("DROP TABLE IF EXISTS bookings")
        c.run("DROP TABLE IF EXISTS cars")
    return PgDb(pg_url)


@pytest.fixture
def inventory(pgdb, cfg):
    repo = PgInventoryRepo(pgdb)
    seed_inventory(repo, [b.id for b in cfg.branches])
    return repo


@pytest.fixture
def bookings(pgdb, inventory):
    return PgBookingRepo(pgdb)


SKIP = {"test_db_rejects_floor_above_price", "test_database_guards_hold_even_if_python_checks_are_bypassed"}   # SQLite-specific
for _name in dir(repo_tests):
    if _name.startswith("test_") and _name not in SKIP:
        globals()[_name] = getattr(repo_tests, _name)


def test_two_workers_racing_for_one_slot_produce_exactly_one_booking(pg_url, inventory):
    """Separate pools = separate voice workers. 16 callers, one slot: one booking, 15 told it has gone."""
    workers = [PgBookingRepo(PgDb(pg_url, size=4)) for _ in range(4)]
    results, barrier = [], threading.Barrier(16)

    def call(i):
        barrier.wait()
        try:
            workers[i % 4].create(repo_tests.new(phone=f"05000000{i:02d}"), repo_tests.SLOT)
            results.append("ok")
        except SlotTaken:
            results.append("taken")

    threads = [threading.Thread(target=call, args=(i,)) for i in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("ok") == 1 and results.count("taken") == 15


def test_loading_the_same_stock_twice_is_safe(inventory, cfg):
    before = inventory.count(repo_tests.SearchFilters(include_unavailable=True))
    seed_inventory(inventory, [b.id for b in cfg.branches])         # a second worker starting up
    assert inventory.count(repo_tests.SearchFilters(include_unavailable=True)) == before


def test_a_whole_booking_call_through_the_app_lands_in_postgres(pg_url):
    """The agent, the service and the website's stock on PostgreSQL: a test drive booked in one 'worker' is seen by another."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_agent import DATA_JS, NOW, Scripted
    from datetime import timezone

    if not DATA_JS.exists():
        pytest.skip("the website is not on this machine")
    from dealer_core.agent.engine import AgentEngine
    from dealer_core.app import ROOT, build_app
    from dealer_core.config import load_config

    clean = PgDb(pg_url)
    with clean.tx() as c:                                                    # only this site's stock in the database
        c.run("DROP TABLE IF EXISTS bookings")
        c.run("DROP TABLE IF EXISTS cars")

    def worker():
        cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
        cfg.app.database_url = pg_url
        app = build_app(cfg, inventory="portal", portal_data=DATA_JS, classifier=Scripted([]), reply_mode="conversational",
                        clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Scripted([]))
        return AgentEngine(app.service, Scripted([]), app.engine.reply_writer), app

    e1, app1 = worker()
    s = e1.new_session()
    for line in ("Toyota", "the Corolla", "yes", "tomorrow", "9", "Sara 050 111 2222", "yes"):
        e1.handle(s, line)
    assert s.booking is not None
    e2, app2 = worker()                                                      # another worker, its own connections
    assert app2.service.bookings.find_active("+971501112222", s.booking.stock_id) is not None


def test_workers_starting_at_the_same_moment_all_get_their_tables(pg_url):
    """12 Oct-style start-up race: three workers creating the schema at once must not fail (it did, before the advisory lock)."""
    clean = PgDb(pg_url)
    with clean.tx() as c:
        c.run("DROP TABLE IF EXISTS bookings")
        c.run("DROP TABLE IF EXISTS cars")
    errors, barrier = [], threading.Barrier(6)

    def start():
        barrier.wait()
        try:
            PgDb(pg_url, size=1)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=start) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
