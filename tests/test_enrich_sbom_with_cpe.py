# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``scripts/enrich_sbom_with_cpe.py``.

The enrichment step injects the project's CPE 2.3 identifier onto the
root component of the CycloneDX 1.5 SBOM before Sigstore signs it. The
test fixtures model the output shape of ``uv export --format
cyclonedx1.5`` (``metadata.component`` present at the top level) as well
as the two malformed shapes that must raise rather than silently skip:
missing ``metadata`` and missing ``metadata.component``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "enrich_sbom_with_cpe.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "enrich_sbom_with_cpe", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["enrich_sbom_with_cpe"] = module
    spec.loader.exec_module(module)
    return module


enrich_module = _load_module()


def _write_sbom(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_enrich_injects_cpe_on_root_component(tmp_path: Path) -> None:
    sbom = tmp_path / "sbom.cdx.json"
    _write_sbom(
        sbom,
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "metadata": {
                "component": {
                    "type": "application",
                    "name": "spotforecast2-safe",
                    "version": "1.2.3",
                }
            },
            "components": [],
        },
    )

    cpe = enrich_module.enrich(sbom, "1.2.3")

    assert cpe == (
        "cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:"
        "1.2.3:*:*:*:*:*:*:*"
    )
    enriched = json.loads(sbom.read_text(encoding="utf-8"))
    assert enriched["metadata"]["component"]["cpe"] == cpe
    # Pre-existing fields must survive the rewrite.
    assert enriched["metadata"]["component"]["name"] == "spotforecast2-safe"
    assert enriched["specVersion"] == "1.5"


def test_enrich_overwrites_stale_cpe(tmp_path: Path) -> None:
    """A rerun with a different version must replace the prior CPE, not
    duplicate or append. This guards the idempotent-rerun property that
    the release workflow depends on."""
    sbom = tmp_path / "sbom.cdx.json"
    _write_sbom(
        sbom,
        {
            "metadata": {
                "component": {
                    "type": "application",
                    "name": "spotforecast2-safe",
                    "cpe": "cpe:2.3:a:old:old:0.0.0:*:*:*:*:*:*:*",
                }
            }
        },
    )

    new_cpe = enrich_module.enrich(sbom, "2.0.0")

    enriched = json.loads(sbom.read_text(encoding="utf-8"))
    assert enriched["metadata"]["component"]["cpe"] == new_cpe
    assert "old" not in enriched["metadata"]["component"]["cpe"]


def test_enrich_raises_on_missing_metadata(tmp_path: Path) -> None:
    sbom = tmp_path / "sbom.cdx.json"
    _write_sbom(sbom, {"bomFormat": "CycloneDX", "specVersion": "1.5"})

    with pytest.raises(RuntimeError, match="'metadata' object missing"):
        enrich_module.enrich(sbom, "1.0.0")


def test_enrich_raises_on_missing_root_component(tmp_path: Path) -> None:
    sbom = tmp_path / "sbom.cdx.json"
    _write_sbom(sbom, {"metadata": {}})

    with pytest.raises(RuntimeError, match="'metadata.component' missing"):
        enrich_module.enrich(sbom, "1.0.0")
