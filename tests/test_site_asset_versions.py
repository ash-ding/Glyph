"""Every asset the project page loads carries a stamp matching its content.

Without it, a browser can pair a fresh index.html with a stylesheet cached
under the same URL minutes earlier, and render new markup with old styles.
Fix a failure here with ``python tools/stamp_site_assets.py``.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from stamp_site_assets import REF, SITE, stale  # noqa: E402

PAGES = sorted(SITE.glob("*.html"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_asset_stamps_match_content(page):
    text = page.read_text()
    assert REF.search(text), f"{page.name} references no local assets"
    off = stale(text)
    assert not off, "\n".join(f"{p}: stamped {h!r}, content is {w!r}" for p, h, w in off)
