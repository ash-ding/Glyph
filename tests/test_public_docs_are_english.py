"""The public face of the repository is English.

README.md and the project page are what someone outside the project reads
first; the internal record in docs/ is not, and is deliberately not covered
here. Chinese has leaked into the site twice now while syncing it to a
change, and neither time did anything catch it.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def _public_files():
    files = [ROOT / "README.md"]
    files += sorted(p for p in (ROOT / "site").rglob("*")
                    if p.is_file() and p.suffix in {".html", ".css", ".js", ".md"})
    return files


@pytest.mark.parametrize("path", _public_files(),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_public_facing_text_is_english(path):
    offenders = [(i, line) for i, line in enumerate(path.read_text().splitlines(), 1)
                 if CJK.search(line)]
    assert not offenders, "\n".join(
        f"{path.relative_to(ROOT)}:{i}: {line.strip()[:90]}" for i, line in offenders)
