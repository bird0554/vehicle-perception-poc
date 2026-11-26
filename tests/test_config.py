from pathlib import Path

import pytest

from app.config import Settings


def test_settings_reject_invalid_parking_zone() -> None:
    with pytest.raises(ValueError, match="minimum coordinates"):
        Settings(database_path=Path("unused.sqlite3"), parking_zone_mm=(10, 0, 0, 10))


def test_settings_reject_unsupported_required_view() -> None:
    with pytest.raises(ValueError, match="supported view names"):
        Settings(database_path=Path("unused.sqlite3"), required_views=("diagonal",))
