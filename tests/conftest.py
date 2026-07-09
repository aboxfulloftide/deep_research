import os

import pytest_asyncio

from deep_research.kb.db import KBDatabase

# Deliberately a separate database from the real KB (deep_research_kb) --
# tests truncate tables between runs, so this must never point at real data.
TEST_DSN = os.environ.get(
    "DEEP_RESEARCH_TEST_POSTGRES_DSN",
    "postgresql://deep_research:deep_research@localhost:5432/deep_research_kb_test",
)

# Static lookup/reference data seeded by KBDatabase.init() itself, not a test
# subject -- left alone across truncates instead of being re-seeded per test.
_KEEP_SEEDED = {"source_types", "trust_tiers"}


@pytest_asyncio.fixture
async def kb_db():
    """A KBDatabase against the dedicated test database. Schema is created
    (idempotently) on init(); every content table is truncated after the test
    so state never leaks between tests. Skips (not fails) if the test
    database isn't reachable, so pure-logic tests in the same run still work
    without Postgres up."""
    db = KBDatabase(TEST_DSN)
    try:
        await db.init()
    except Exception as e:
        import pytest
        pytest.skip(f"Test database not reachable ({e}); skipping DB-backed test")
        return

    try:
        yield db
    finally:
        async with db.pool.acquire() as conn:
            tables = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            names = [t["tablename"] for t in tables if t["tablename"] not in _KEEP_SEEDED]
            if names:
                quoted = ", ".join(f'"{n}"' for n in names)
                await conn.execute(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE")
        await db.close()
