"""Shared fixtures: a seeded in-memory mock database and the default config."""
import pytest

from dealer_core.config import load_config
from dealer_core.repos.seed import seed_inventory
from dealer_core.repos.sqlite_store import SqliteBookingRepo, SqliteDb, SqliteInventoryRepo


@pytest.fixture
def cfg():
    return load_config(environ={})


@pytest.fixture
def db():
    return SqliteDb(":memory:")


@pytest.fixture
def inventory(db, cfg):
    repo = SqliteInventoryRepo(db)
    seed_inventory(repo, [b.id for b in cfg.branches])
    return repo


@pytest.fixture
def bookings(db, inventory):
    return SqliteBookingRepo(db)


@pytest.fixture
def demo_inventory(db):
    from dealer_core.repos.demo_data import seed_demo_inventory

    repo = SqliteInventoryRepo(db)
    seed_demo_inventory(repo)
    return repo
