import os

import pytest

# config.py raises at import if OPENROUTER_API_KEY is unset.
# Set a dummy so test collection can import modules that transitively import config.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point config.DB_PATH at a per-test temporary file.

    db.py resolves DB_PATH at call time, so this gives every test a fresh database
    and prevents tests from touching the developer's real .lemon.db.
    """
    import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "lemon.db")
    yield
