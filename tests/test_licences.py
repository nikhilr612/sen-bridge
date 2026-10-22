# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""The licence files, notices and metadata stay consistent with each other."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDER = "Nikhil Raghavendra"


def test_code_licence_is_apache_and_names_the_holder() -> None:
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert f"Copyright 2026 {HOLDER}" in notice
    assert "Contains modified Copernicus Sentinel data" in notice


def test_package_metadata_uses_an_spdx_licence_and_names_the_author() -> None:
    # ``tomllib`` is missing on the oldest supported Python, so match the text.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in text
    assert f'name = "{HOLDER}"' in text


def test_every_third_party_licence_named_in_notice_is_shipped() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    shipped = {p.name for p in (ROOT / "licenses").glob("LICENSE-*.txt")}
    named = {
        w
        for w in notice.split()
        if w.startswith("LICENSE-") and w.endswith(".txt)")
    }
    assert {n.rstrip(")") for n in named} <= shipped


def test_dataset_licence_text_is_cc_by() -> None:
    text = (ROOT / "licenses" / "LICENSE-CC-BY-4.0.txt").read_text(encoding="utf-8")
    assert "Attribution 4.0 International" in text
