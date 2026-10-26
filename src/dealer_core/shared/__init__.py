"""State shared between voice workers: call snapshots (so a call survives a worker restart) and the worker registry (so new calls go to
the least busy worker). Redis in production, an in-process store on a single machine and in tests."""
