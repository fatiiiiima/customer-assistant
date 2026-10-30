"""The stock: the database is the only place the voice agent and the website read cars from (docs/ARCH_PLAN.md step 0, ARCH_REVIEW section 51).

  columns.py  a stock-list row, whatever its headings, as a Car and its spec sheet
  sources.py  reading CSV, TSV, Excel, JSON, and the website's data.js (the demo stock)
  sync.py     the database brought in line with a stock list, in one transaction, with a report
  api.py      the website's read-only car API (and window.CARS for the pages as they are today)
  python -m dealer_core.inventory import|status|serve   the command line
"""
from __future__ import annotations

from ..config import Config
from .columns import Branches


def open_db(url: str = "", pool_size: int = 8):
    """(inventory repository, booking repository) for a database address:
    ""                          -> an in-memory SQLite (one process; gone when it stops: the demo)
    "sqlite:///stock.db", a .db -> a SQLite file (one machine)
    "postgresql://..."          -> PostgreSQL, shared by every worker (production)"""
    if url.startswith(("postgresql://", "postgres://")):
        from ..repos.pg_store import PgBookingRepo, PgInventoryRepo, shared_db

        db = shared_db(url, pool_size)
        return PgInventoryRepo(db), PgBookingRepo(db)
    from ..repos.sqlite_store import SqliteBookingRepo, SqliteDb, SqliteInventoryRepo

    path = url[len("sqlite:///"):] if url.startswith("sqlite:///") else url[len("sqlite://"):] if url.startswith("sqlite://") else url
    if url and not url.startswith("sqlite") and not url.endswith((".db", ".sqlite", ".sqlite3")):
        raise ValueError(f"not a database address: {url!r} (postgresql://..., sqlite:///file.db, or a .db file)")
    db = SqliteDb(path or ":memory:")
    return SqliteInventoryRepo(db), SqliteBookingRepo(db)


def branches_of(cfg: Config) -> Branches:
    """The branches a stock list may name, by id or by name; a row without one is at the first branch."""
    names = {}
    for b in cfg.branches:
        names[b.id.lower()] = b.id
        names[b.name.lower()] = b.id
    return Branches(default=cfg.branches[0].id, by_name=names)
