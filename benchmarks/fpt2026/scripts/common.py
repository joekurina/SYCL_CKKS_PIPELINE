#!/usr/bin/env python3
"""Shared deterministic I/O and hashing for the FPT 2026 harness."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class ContractError(RuntimeError):
    """Raised when evidence violates a frozen harness contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")


def _reject_nonfinite_constant(value: str) -> None:
    raise ContractError(f"nonstandard nonfinite JSON constant is forbidden: {value}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path | str) -> tuple[str, list[dict[str, Any]]]:
    """Hash a directory from sorted relative paths, sizes, and file hashes."""
    root = require_absolute(path, "tree path")
    if not root.is_dir():
        raise ContractError(f"tree path is not a directory: {root}")
    entries: list[dict[str, Any]] = []
    for candidate in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append({
            "relative_path": candidate.relative_to(root).as_posix(),
            "size_bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        })
    return sha256_bytes(canonical_json_bytes(entries)), entries


def require_absolute(path: Path | str, label: str = "path") -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ContractError(f"{label} must be absolute: {candidate}")
    return candidate


def load_json(path: Path | str) -> Any:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_reject_nonfinite_constant)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ContractError(f"blank JSONL row at {source}:{line_number}")
            try:
                record = json.loads(line, parse_constant=_reject_nonfinite_constant)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ContractError(f"JSONL row is not an object at {source}:{line_number}")
            records.append(record)
    return records


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_bytes(path: Path | str, payload: bytes) -> None:
    """Durably publish bytes without ever replacing an existing evidence file."""
    destination = require_absolute(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"immutable output already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(f"immutable output already exists: {destination}")
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_new_json(path: Path | str, value: Any) -> str:
    payload = canonical_json_bytes(value)
    write_new_bytes(path, payload)
    return sha256_bytes(payload)


def append_jsonl(path: Path | str, record: dict[str, Any]) -> None:
    destination = require_absolute(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(record)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short JSONL append to {destination}: {written}/{len(payload)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def require_unique(records: Iterable[dict[str, Any]], key: str, label: str) -> None:
    seen: set[Any] = set()
    for record in records:
        value = record.get(key)
        if value in seen:
            raise ContractError(f"duplicate {label} {value!r}")
        seen.add(value)
