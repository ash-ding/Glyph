#!/usr/bin/env python3
"""Stamp every local asset the project page loads with a content hash.

GitHub Pages serves everything with ``Cache-Control: max-age=600``. A browser
that fetches a fresh ``index.html`` can still apply a stylesheet or script it
cached minutes earlier, and the page then renders new markup with old styles.
Pointing each reference at ``asset?v=<hash of its bytes>`` makes a changed
file a different URL, so no stale copy can be used:

    python tools/stamp_site_assets.py          # rewrite site/*.html in place
    python tools/stamp_site_assets.py --check  # exit 1 if any stamp is stale

``tests/test_site_asset_versions.py`` runs the check.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
PREFIX = "/Glyph/"          # 404.html uses absolute project-page paths
REF = re.compile(r'(?P<attr>href|src)="(?P<path>(?:/Glyph/)?assets/[^"?#]+)(?:\?v=(?P<v>[0-9a-f]*))?"')


def digest(rel: str) -> str:
    local = rel[len(PREFIX):] if rel.startswith(PREFIX) else rel
    return hashlib.sha256((SITE / local).read_bytes()).hexdigest()[:10]


def stale(html: str) -> list[tuple[str, str | None, str]]:
    """(path, current stamp, wanted stamp) for every reference that is off."""
    out = []
    for m in REF.finditer(html):
        want = digest(m["path"])
        if m["v"] != want:
            out.append((m["path"], m["v"], want))
    return out


def stamp(html: str) -> str:
    return REF.sub(lambda m: f'{m["attr"]}="{m["path"]}?v={digest(m["path"])}"', html)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report stale stamps, change nothing")
    args = ap.parse_args(argv)
    bad = 0
    for page in sorted(SITE.glob("*.html")):
        text = page.read_text()
        if args.check:
            for path, have, want in stale(text):
                print(f"{page.name}: {path} stamped {have!r}, content is {want!r}")
                bad += 1
        else:
            new = stamp(text)
            if new != text:
                page.write_text(new)
                print(f"stamped {page.name}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
