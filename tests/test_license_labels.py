"""yuv.ts (and resample.ts, which re-exports it) is treated as derived from FFmpeg's libswscale (LGPL-2.1-or-later), so the copy must say so everywhere it labels a licence and
must ship the LGPL text: next to the source, inside the notices that travel with the page, and in the package metadata."""
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LGPL_HEAD = "GNU LESSER GENERAL PUBLIC LICENSE"
LGPL_CLOSE = "That's all there is to it!"  # last line of the "How to Apply" appendix: marks the whole text, not only a pointer to it


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_lgpl_text_ships_in_the_repo():
    text = read("LICENSES/LGPL-2.1.txt")
    assert LGPL_HEAD in text and "Version 2.1, February 1999" in text and LGPL_CLOSE in text


def test_resample_source_carries_its_own_licence_header():
    for rel in ("web/src/browser/resample.ts", "web/src/browser/yuv.ts"):
        head = "\n".join(read(rel).splitlines()[:6])
        assert "SPDX-License-Identifier: LGPL-2.1-or-later" in head, rel
        assert "LICENSES/LGPL-2.1.txt" in head, rel


def test_notice_and_page_notices_ship_the_lgpl_text_and_no_open_review_note():
    notice = read("NOTICE")
    assert "LGPL-2.1-or-later" in notice and "LICENSES/LGPL-2.1.txt" in notice
    assert "has not been reviewed by a lawyer" not in notice
    page = read("web/src/notices.txt")
    entry = next(e for e in page.split("\n---\n") if re.match(r"\s*Name: FFmpeg libswscale", e))
    assert LGPL_HEAD in entry and LGPL_CLOSE in entry
    assert "get that review before the public release" not in entry


def test_built_page_carries_the_lgpl_text():
    shipped = read("src/inframe/web/dist/THIRD_PARTY_LICENSES.txt")
    assert LGPL_HEAD in shipped and LGPL_CLOSE in shipped
    assert "get that review before the public release" not in shipped


def test_package_metadata_names_the_lgpl_file_beside_apache():
    py = tomllib.loads(read("pyproject.toml"))["project"]["license"]
    js = json.loads(read("web/package.json"))["license"]
    for label in (py["text"] if isinstance(py, dict) else py, js):
        assert "Apache-2.0" in label and "LGPL-2.1-or-later" in label


def test_built_page_ships_the_lgpl_source_beside_its_notices():
    """LGPL-2.1 section 4: the page and the wheel carry resample.ts in object form, so its source must come with them."""
    for name in ("resample.ts", "yuv.ts"):
        shipped = ROOT / "src/inframe/web/dist" / name
        assert shipped.is_file(), f"dist has no {name}"
        assert shipped.read_bytes() == (ROOT / "web/src/browser" / name).read_bytes()
        assert "SPDX-License-Identifier: LGPL-2.1-or-later" in shipped.read_text(encoding="utf-8")
    assert "resample.ts" in read("src/inframe/web/dist/THIRD_PARTY_LICENSES.txt").split("Decision (recorded", 1)[1]
    assert "dist/resample.ts" in read("web/src/notices.txt") and "dist/resample.ts" in read("NOTICE")


def test_wheel_metadata_and_notice_name_the_cc_by_songs_and_their_credits():
    """The wheel bundles CC BY 3.0 audio, so its license expression, its license files and NOTICE must say so."""
    proj = tomllib.loads(read("pyproject.toml"))["project"]
    expr = proj["license"]["text"] if isinstance(proj["license"], dict) else proj["license"]
    assert "CC-BY-3.0" in expr and "Apache-2.0" in expr and "LGPL-2.1-or-later" in expr
    assert "src/inframe/music_library/CREDITS.md" in proj.get("license-files", [])
    notice = read("NOTICE")
    assert "CC BY 3.0" in notice and "https://creativecommons.org/licenses/by/3.0/" in notice and "music_library/CREDITS.md" in notice
