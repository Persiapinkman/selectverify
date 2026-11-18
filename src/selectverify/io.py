"""Strict JSONL output handling and portable reproducibility metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}, line {number}; preserve and repair before resuming") from exc


def records(paths):
    result = {}
    for path in paths:
        for row in read_jsonl(path):
            result[row["id"]] = row
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare_output(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Refusing to resume with different experiment metadata")
    else:
        if path.exists() and path.stat().st_size:
            raise ValueError("Existing output has no metadata; use a new output path")
        write_json(meta_path, metadata)
    rows = list(read_jsonl(path)) if path.exists() else []
    keys = [(r["id"], r.get("method", "signals")) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (id, method) records in output")
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                raise ValueError("Output is missing its final newline; repair before resuming")
    return set(keys)
