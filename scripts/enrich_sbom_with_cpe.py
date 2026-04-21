#!/usr/bin/env python3
"""Inject the project's CPE 2.3 identifier onto a CycloneDX 1.5 SBOM.

The SBOM emitted by ``uv export --format cyclonedx1.5`` describes the
resolved production-dependency graph of ``spotforecast2-safe`` using PURL
identifiers. Downstream vulnerability tooling that resolves packages via
the NIST NVD (CISA KEV feeds, several vulnerability-management-as-a-service
products) keys on CPE 2.3, not PURL, so the shipped SBOM must carry the
project's own CPE on its root component.

This script performs that single injection: it reads the SBOM, sets
``metadata.component.cpe`` to the string produced by
``spotforecast2_safe.utils.cpe.get_cpe_identifier``, and writes the file
back in place. The enrichment must run before Sigstore signing so the
signature bundle covers the SBOM that actually ships.

Usage:
    python scripts/enrich_sbom_with_cpe.py --sbom dist/sbom.cdx.json
    python scripts/enrich_sbom_with_cpe.py --sbom dist/sbom.cdx.json --version 1.2.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spotforecast2_safe import __version__
from spotforecast2_safe.utils.cpe import get_cpe_identifier


def enrich(sbom_path: Path, version: str) -> str:
    data = json.loads(sbom_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"{sbom_path}: top-level 'metadata' object missing; "
            "CycloneDX 1.5 SBOM is malformed."
        )
    component = metadata.get("component")
    if not isinstance(component, dict):
        raise RuntimeError(
            f"{sbom_path}: 'metadata.component' missing; cannot locate "
            "root component to attach the CPE identifier to."
        )
    cpe = get_cpe_identifier(version)
    component["cpe"] = cpe
    sbom_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return cpe


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inject the project's CPE 2.3 identifier onto the root "
            "component of a CycloneDX 1.5 SBOM."
        )
    )
    parser.add_argument(
        "--sbom", type=Path, required=True, help="Path to the SBOM JSON file."
    )
    parser.add_argument(
        "--version",
        default=__version__,
        help=f"Version string for the CPE (default: {__version__}).",
    )
    args = parser.parse_args()
    cpe = enrich(args.sbom, args.version)
    print(f"Injected CPE on root component of {args.sbom}: {cpe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
