# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Enforce the prohibited-dependency blocklist by scanning ``uv.lock``.

``spotforecast2-safe`` maintains a short, version-controlled list of
dependencies whose presence --- direct or transitive --- would inflate the
attack surface, introduce nondeterminism, or break cross-platform
reproducibility. This test grep-scans the project lockfile and fails if any
blocklisted package is resolved.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROHIBITED_DEPENDENCIES: tuple[str, ...] = (
    "plotly",
    "matplotlib",
    "spotoptim",
    "optuna",
    "torch",
    "tensorflow",
)

LOCKFILE = Path(__file__).resolve().parent.parent / "uv.lock"


def _scan_lockfile_for(pkg: str, lockfile_text: str) -> list[int]:
    pattern = re.compile(rf'^\s*name\s*=\s*"{re.escape(pkg)}"\s*$', re.MULTILINE)
    return [
        lockfile_text.count("\n", 0, m.start()) + 1
        for m in pattern.finditer(lockfile_text)
    ]


class TestProhibitedDependencies:
    """Verify that no blocklisted dependency is pinned in ``uv.lock``."""

    def test_lockfile_exists(self) -> None:
        assert LOCKFILE.is_file(), (
            f"Lockfile not found at {LOCKFILE}; run `uv lock` first."
        )

    @pytest.mark.parametrize("pkg", PROHIBITED_DEPENDENCIES)
    def test_prohibited_dependency_absent(self, pkg: str) -> None:
        text = LOCKFILE.read_text(encoding="utf-8")
        hits = _scan_lockfile_for(pkg, text)
        assert not hits, (
            f"Prohibited dependency {pkg!r} is resolved in {LOCKFILE.name} "
            f"at line(s) {hits}. The blocklist exists to keep the attack "
            f"surface, determinism, and cross-platform reproducibility of "
            f"spotforecast2-safe intact; replace the offending dependency "
            f"rather than loosen this test."
        )

    def test_blocklist_is_nonempty_and_unique(self) -> None:
        assert PROHIBITED_DEPENDENCIES
        assert len(set(PROHIBITED_DEPENDENCIES)) == len(PROHIBITED_DEPENDENCIES)
