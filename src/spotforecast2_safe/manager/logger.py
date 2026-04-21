# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audit-grade logging for spotforecast2-safe.

This module sets up a dual-handler logger: a console handler for humans and
a file handler that writes one JSON object per line, conforming to the
schema at ``audit_log_schema.json`` next to this file. The file is the
compliance-relevant sink (EU AI Act Article 12, IEC 62443-4-2 SAR 6.1,
IEC 61508-3 §7.4.7); the console handler is plaintext for interactive use.

Schema versioning rule: ``SCHEMA_VERSION`` is loaded directly from
``audit_log_schema.json`` at import time, so there is exactly one source of
truth. Any change to the schema file is guarded by the CI job
``audit-log-schema-gate`` (``.github/workflows/ci.yml``), which rejects a
pull request that touches the schema without a Conventional-Commits
breaking-change marker (``type!:``) on at least one commit. That marker
cascades to a MAJOR semantic-release bump.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

__all__ = [
    "SCHEMA_VERSION",
    "JsonAuditFormatter",
    "setup_logging",
]

_SCHEMA_PATH = Path(__file__).resolve().parent / "audit_log_schema.json"
_AUDIT_SCHEMA: dict[str, Any] = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
SCHEMA_VERSION: str = _AUDIT_SCHEMA["properties"]["schema_version"]["const"]

_BASE_LOGRECORD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {"message", "asctime"}


class JsonAuditFormatter(logging.Formatter):
    """Format ``LogRecord`` instances as single-line JSON per audit_log_schema.json.

    The formatter emits exactly the fields named in the schema's ``properties``
    section, never more. Callers pass optional structured context through the
    standard ``logging`` ``extra=`` mechanism; recognised extras are ``event``,
    ``task``, and ``context``.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": _utc_isoformat(record.created),
            "logger": record.name,
            "level": record.levelname,
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
        }

        task = getattr(record, "task", None)
        if isinstance(task, str) and task:
            payload["task"] = task

        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = context

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _utc_isoformat(created: float) -> str:
    dt = datetime.fromtimestamp(created, tz=timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def setup_logging(
    level: int = logging.INFO, log_dir: Optional[Path] = None
) -> Tuple[logging.Logger, Optional[Path]]:
    """Configure dual-handler logging for safety-critical execution.

    Attaches a stream handler (stdout, human-readable plaintext) and, when
    ``log_dir`` is provided, a file handler that writes JSON records per
    ``audit_log_schema.json``. The console handler honours ``level``; the
    file handler is always at ``INFO`` so the audit trail stays complete
    even when the operator silences the console.

    Args:
        level: Logging level for console output. Default: ``logging.INFO``.
        log_dir: Optional directory for the audit log file. If provided, a
            timestamped ``task_safe_n_to_1_YYYYMMDD_HHMMSS.log`` file is
            created and receives JSON-formatted records.

    Returns:
        Tuple of the configured logger and the audit log file path (or
        ``None`` if ``log_dir`` was omitted).
    """
    logger = logging.getLogger("task_safe_n_to_1")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        existing_path = None
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                existing_path = Path(h.baseFilename)
        return logger, existing_path

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    log_file_path = None
    if log_dir:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = log_dir / f"task_safe_n_to_1_{timestamp}.log"

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(JsonAuditFormatter())
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.info(
                f"Persistent logging initialized at: {log_file_path}",
                extra={"event": "audit_log_init"},
            )
        except Exception as e:
            logger.warning(f"Failed to initialize file logging in {log_dir}: {e}")

    return logger, log_file_path
