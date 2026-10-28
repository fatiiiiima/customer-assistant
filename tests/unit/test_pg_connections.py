"""B6 (docs/ARCH_REVIEW.md section 20): a worker used to open two connection pools to PostgreSQL (8 for bookings and stock, 4 for call state and the registry),
and the router's registry read was one query for the key list, a DELETE, and one query per worker, every 0.5 s."""
import json
import tempfile
import threading
import time

import pytest

from dealer_core.config import AppCfg
from dealer_core.repos.pg_store import PgDb, shared_db
from dealer_core.shared.state import MemoryStore, PgStore, WorkerRegistry, open_store


@pytest.fixture(scope="module")
def pg_url():
    pgserver = pytest.importorskip("pgserver")
    return pgserver.get_server(tempfile.mkdtemp(prefix="pg_conn_"), cleanup_mode="stop").get_uri()


def count_transactions(db):
    """Wraps db.tx so every transaction (one per statement group) is counted."""
    n = {"tx": 0}
    real = db.tx

    def tx():
        n["tx"] += 1
        return real()

    db.tx = tx
    return n, lambda: setattr(db, "tx", real)


def test_one_pool_per_process_per_database(pg_url):
    a = shared_db(pg_url, 8)
    assert shared_db(pg_url, 4) is a and a._size == 8        # a smaller ask does not shrink it
    assert shared_db(pg_url, 12)._size == 12                 # a larger one raises the limit
    a._size = 8
    assert PgStore(pg_url).db is a                           # the registry and call snapshots use the same pool as the bookings
    assert open_store(pg_url, 8).db is a


def test_the_pool_never_opens_more_connections_than_its_size(pg_url):
    db = PgDb(pg_url, size=3)
    gate, errors = threading.Barrier(12), []

    def work():
        try:
            gate.wait()
            for _ in range(5):
                db.rows("SELECT 1 AS one")
        except Exception as e:                               # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and db._made <= 3


def test_the_pool_size_is_a_setting_and_defaults_to_one_pool_of_eight():
    assert AppCfg().db_pool_size == 8                        # was 8 + 4 per worker; PgStore's 4 is now part of the 8


def test_reading_the_registry_is_one_query_however_many_workers(pg_url):
    store = PgStore(pg_url)
    with store.db.tx() as c:
        c.run("DELETE FROM kv")
    reg = WorkerRegistry(store)
    for i in range(40):
        reg.beat(f"w{i}", f"ws://10.0.0.{i}/voice", active=i % 16, max_calls=16)
    n, restore = count_transactions(store.db)
    try:
        for _ in range(5):
            assert len(reg.workers()) == 40
    finally:
        restore()
    # first read: the clean-up DELETE and the read; the next four: the read alone. It was 2 + 40 on every read (5 x 42 = 210).
    assert n["tx"] == 6


def test_expired_rows_are_cleaned_up_but_not_on_every_read(pg_url):
    store = PgStore(pg_url)
    with store.db.tx() as c:
        c.run("DELETE FROM kv")
        c.run("INSERT INTO kv VALUES ('worker:old', '{}', now() - interval '5 minutes')")
        c.run("INSERT INTO kv VALUES ('worker:fresh', '{\"id\": \"fresh\"}', now() + interval '1 minute')")
    store._tidied = 0.0
    assert [w["id"] for w in WorkerRegistry(store).workers()] == ["fresh"]
    with store.db.tx() as c:
        assert c.run("SELECT count(*) FROM kv WHERE key = 'worker:old'")[0][0] == 0           # tidied on the first read
        c.run("INSERT INTO kv VALUES ('worker:old2', '{}', now() - interval '5 minutes')")
    WorkerRegistry(store).workers()
    with store.db.tx() as c:
        assert c.run("SELECT count(*) FROM kv WHERE key = 'worker:old2'")[0][0] == 1          # not again within TIDY_EVERY_S


def test_items_agree_with_keys_and_get_on_every_store(pg_url):
    for store in (MemoryStore(), PgStore(pg_url)):
        for k in [k for k in store.keys("zz:")]:
            store.delete(k)
        store.set("zz:a", "1")
        store.set("zz:b", json.dumps({"x": 2}), ttl_s=60)
        store.set("yy:c", "3")
        store.set("zz:gone", "4", ttl_s=0.001)
        time.sleep(0.01)
        assert store.items("zz:") == {"zz:a": "1", "zz:b": '{"x": 2}'}
        assert sorted(store.keys("zz:")) == sorted(store.items("zz:"))
