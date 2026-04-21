# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate the audit log against its pinned JSON schema.

The schema at ``src/spotforecast2_safe/manager/audit_log_schema.json`` is the
single source of truth for every line the file handler writes under
``~/spotforecast2_safe_models/logs/``. These tests check that:

1. The schema file loads as a well-formed Draft 2020-12 JSON Schema.
2. ``SCHEMA_VERSION`` is read from the schema, not hard-coded separately.
3. ``JsonAuditFormatter`` produces JSON that satisfies the schema's
   ``required`` and ``properties`` constraints for all plausible record
   shapes (bare, with extras, with an exception).
4. An end-to-end ``setup_logging`` → ``logger.info`` → file-read round trip
   yields schema-conformant lines.

The validation is intentionally dependency-free (no ``jsonschema`` dep) to
keep the audit-critical test path inside stdlib.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from spotforecast2_safe.manager.logger import (
    SCHEMA_VERSION,
    JsonAuditFormatter,
    setup_logging,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "spotforecast2_safe"
    / "manager"
    / "audit_log_schema.json"
)
_JSON_TYPE = {
    "string": str,
    "object": dict,
    "array": list,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _validate(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(schema["required"])
    properties = schema["properties"]

    for field in required:
        if field not in record:
            errors.append(f"missing required field {field!r}")

    for field, value in record.items():
        if field not in properties:
            if not schema.get("additionalProperties", True):
                errors.append(f"unexpected field {field!r}")
            continue
        spec = properties[field]
        expected = _JSON_TYPE[spec["type"]]
        if not isinstance(value, expected):
            errors.append(
                f"{field}: expected {spec['type']}, got {type(value).__name__}"
            )
            continue
        if "const" in spec and value != spec["const"]:
            errors.append(f"{field}: expected const {spec['const']!r}, got {value!r}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{field}: {value!r} not in enum {spec['enum']}")
        if "minLength" in spec and isinstance(value, str) and len(value) < spec["minLength"]:
            errors.append(f"{field}: shorter than minLength {spec['minLength']}")
    return errors


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _make_record(**overrides: Any) -> logging.LogRecord:
    kwargs: dict[str, Any] = {
        "name": "t",
        "level": logging.INFO,
        "pathname": "",
        "lineno": 0,
        "msg": "hello",
        "args": (),
        "exc_info": None,
    }
    kwargs.update(overrides)
    return logging.LogRecord(**kwargs)


class TestSchemaFile:
    def test_schema_loads(self, schema: dict[str, Any]) -> None:
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False

    def test_schema_version_is_single_source(self, schema: dict[str, Any]) -> None:
        assert SCHEMA_VERSION == schema["properties"]["schema_version"]["const"]

    def test_required_fields_are_defined(self, schema: dict[str, Any]) -> None:
        required = set(schema["required"])
        properties = set(schema["properties"])
        assert required.issubset(properties)

    def test_schema_id_includes_version(self, schema: dict[str, Any]) -> None:
        assert SCHEMA_VERSION in schema["$id"]


class TestJsonAuditFormatter:
    def test_bare_record_is_schema_conformant(self, schema: dict[str, Any]) -> None:
        parsed = json.loads(JsonAuditFormatter().format(_make_record()))
        assert _validate(parsed, schema) == []

    def test_default_event_is_log(self) -> None:
        parsed = json.loads(JsonAuditFormatter().format(_make_record()))
        assert parsed["event"] == "log"

    def test_extras_propagate(self, schema: dict[str, Any]) -> None:
        record = _make_record()
        record.event = "fit"
        record.task = "task_safe_demo"
        record.context = {"rows": 42, "lags": [1, 7, 24]}
        parsed = json.loads(JsonAuditFormatter().format(record))
        assert parsed["event"] == "fit"
        assert parsed["task"] == "task_safe_demo"
        assert parsed["context"] == {"rows": 42, "lags": [1, 7, 24]}
        assert _validate(parsed, schema) == []

    def test_timestamp_is_utc_with_microseconds(self) -> None:
        parsed = json.loads(JsonAuditFormatter().format(_make_record()))
        ts = parsed["timestamp_utc"]
        assert ts.endswith("Z")
        assert "." in ts and len(ts.split(".")[1]) == 7  # 6 digits + trailing "Z"

    def test_exception_field_emitted(self, schema: dict[str, Any]) -> None:
        try:
            raise ValueError("deterministic audit trail")
        except ValueError:
            import sys

            record = _make_record(exc_info=sys.exc_info())
            parsed = json.loads(JsonAuditFormatter().format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert _validate(parsed, schema) == []

    def test_schema_version_is_pinned_const(self) -> None:
        parsed = json.loads(JsonAuditFormatter().format(_make_record()))
        assert parsed["schema_version"] == SCHEMA_VERSION

    def test_level_is_enum_member(self, schema: dict[str, Any]) -> None:
        for lvl in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            parsed = json.loads(JsonAuditFormatter().format(_make_record(level=lvl)))
            assert parsed["level"] in schema["properties"]["level"]["enum"]


class TestEndToEnd:
    def test_file_handler_writes_json_records(
        self, schema: dict[str, Any], tmp_path: Path
    ) -> None:
        logger = logging.getLogger("task_safe_n_to_1")
        for h in list(logger.handlers):
            logger.removeHandler(h)

        logger, log_path = setup_logging(log_dir=tmp_path)
        assert log_path is not None

        logger.info(
            "fit complete",
            extra={"event": "fit", "task": "unit_test", "context": {"score": 0.87}},
        )
        for h in logger.handlers:
            h.flush()

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines, "audit log file is empty"

        for line in lines:
            parsed = json.loads(line)
            errors = _validate(parsed, schema)
            assert errors == [], f"line {line!r}: {errors}"

        for h in list(logger.handlers):
            logger.removeHandler(h)
