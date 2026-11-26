from pathlib import Path

import pytest

from tools.replay import parse_image_name


def test_parse_replay_filename() -> None:
    assert parse_image_name(Path("overhead__camera-5__frame-001.jpg")) == ("overhead", "camera-5")


def test_replay_filename_requires_known_view() -> None:
    with pytest.raises(ValueError, match="Expected"):
        parse_image_name(Path("unknown__cam__frame.jpg"))
