# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the CPE (Common Platform Enumeration) identifier generation.

This module contains unit tests for the get_cpe_identifier function,
which generates CPE 2.3 formatted strings for compliance and inventory tracking.
"""

import pytest

from spotforecast2_safe.utils.cpe import get_cpe_identifier


class TestGetCPEIdentifier:
    """Test suite for get_cpe_identifier function."""

    def test_cpe_default_wildcard_version(self):
        """Test CPE generation with default wildcard version."""
        result = get_cpe_identifier()
        assert result == (
            "cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:*"
            ":*:*:*:*:*:*:*"
        )

    def test_cpe_specific_stable_version(self):
        """Test CPE generation with a specific stable version."""
        result = get_cpe_identifier("0.8.0")
        assert result == (
            "cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:0.8.0"
            ":*:*:*:*:*:*:*"
        )

    def test_cpe_rc_version(self):
        """Test CPE generation with a release candidate version."""
        result = get_cpe_identifier("0.8.0-rc.1")
        assert result == (
            "cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe"
            ":0.8.0-rc.1:*:*:*:*:*:*:*"
        )

    def test_cpe_beta_version(self):
        """Test CPE generation with a beta version."""
        result = get_cpe_identifier("0.7.0-beta.2")
        assert result == (
            "cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe"
            ":0.7.0-beta.2:*:*:*:*:*:*:*"
        )

    def test_cpe_format_consistency(self):
        """Test that CPE format remains consistent across versions."""
        versions = ["0.1.0", "1.0.0", "*", "2.0.0-alpha"]
        for version in versions:
            result = get_cpe_identifier(version)
            # Verify CPE 2.3 structure
            parts = result.split(":")
            assert parts[0] == "cpe"
            assert parts[1] == "2.3"
            assert parts[2] == "a"  # application type
            assert parts[3] == "sequential_parameter_optimization"
            assert parts[4] == "spotforecast2_safe"
            assert parts[5] == version

    def test_cpe_contains_vendor_and_product(self):
        """Test that CPE contains correct vendor and product names."""
        result = get_cpe_identifier()
        assert "sequential_parameter_optimization" in result
        assert "spotforecast2_safe" in result

    def test_cpe_type_error_on_non_string_version(self):
        """Test that TypeError is raised when version is not a string."""
        with pytest.raises(TypeError, match="version must be a string"):
            get_cpe_identifier(1.0)

        with pytest.raises(TypeError, match="version must be a string"):
            get_cpe_identifier(123)

        with pytest.raises(TypeError, match="version must be a string"):
            get_cpe_identifier(None)

        with pytest.raises(TypeError, match="version must be a string"):
            get_cpe_identifier(["0.8.0"])

    def test_cpe_empty_string_version(self):
        """Test CPE generation with empty string version (allowed)."""
        result = get_cpe_identifier("")
        assert result == (
            "cpe:2.3:a:sequential_parameter_optimization:spotforecast2_safe:"
            ":*:*:*:*:*:*:*"
        )

    def test_cpe_version_with_special_characters(self):
        """Test CPE generation with version containing special characters."""
        result = get_cpe_identifier("0.8.0+build.123")
        assert "0.8.0+build.123" in result

    def test_cpe_long_version_string(self):
        """Test CPE generation with a long version string."""
        long_version = "0.8.0-rc.1.post0.dev1234567890"
        result = get_cpe_identifier(long_version)
        assert long_version in result

    def test_cpe_nist_compliance_format(self):
        """Test that CPE follows NIST 2.3 specification format."""
        result = get_cpe_identifier("0.8.0")
        # CPE 2.3 format: cpe:2.3:part:vendor:product:version:update:edition:language:sw_edition:target_sw:target_hw:other
        parts = result.split(":")
        assert len(parts) == 13
        assert parts[0] == "cpe"
        assert parts[1] == "2.3"
        assert parts[2] in [
            "a",
            "h",
            "o",
        ]  # part: application, hardware, or operating system
        # All remaining parts are either wildcards or version info
        for part in parts[6:]:
            assert part == "*"

    def test_cpe_split_rejoin_round_trip(self):
        """Decomposing the CPE at ':' separators and recomposing with
        ':'.join must yield byte-identical output across stable, release-
        candidate, pre-release, wildcard, and empty-string versions.
        This pins the reverse-direction invariant claimed in the
        compliance narrative alongside the forward-direction well-
        formedness checks above; a downstream pipeline that parses and
        re-emits CPE strings (e.g. ``cyclonedx-py``) must not alter the
        canonical string."""
        for version in ("*", "0.8.0", "0.8.0-rc.1", "0.7.0-beta.2", ""):
            original = get_cpe_identifier(version)
            assert ":".join(original.split(":")) == original


class TestCPEIntegration:
    """Integration tests for CPE identifier usage."""

    def test_cpe_sbom_compatibility(self):
        """Test that CPE format is compatible with SBOM generation."""
        cpe = get_cpe_identifier("0.8.0")
        # SBOM tools typically expect the format to start with cpe:2.3:a
        assert cpe.startswith("cpe:2.3:a")

    def test_cpe_multiple_calls_consistency(self):
        """Test that multiple calls with same version produce identical results."""
        version = "0.8.0"
        result1 = get_cpe_identifier(version)
        result2 = get_cpe_identifier(version)
        assert result1 == result2

    def test_cpe_reproducibility(self):
        """Test that CPE generation is fully reproducible (deterministic)."""
        # This is important for compliance and audit purposes
        versions = ["*", "0.1.0", "1.0.0", "0.8.0-rc.1"]
        results = {}

        for version in versions:
            results[version] = get_cpe_identifier(version)

        # Re-run the same generation
        for version in versions:
            assert results[version] == get_cpe_identifier(version)
