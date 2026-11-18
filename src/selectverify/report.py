"""Recompute paper metrics from per-example scores without data or model access."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from .algorithm import metrics
from .io import read_jsonl


def distribution(values):
    return {"mean": float(np.mean(values)), "p50": float(np.median(values)),
            "p90": float(np.quantile(values, .9))}


def summarize(path, methods=None, expected_ids=None):
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError("No prediction records")
    methods = methods or sorted({r["method"] for r in rows})
    by_method = {name: {} for name in methods}
    for row in rows:
        if row["method"] not in by_method:
            continue
        mapping = by_method[row["method"]]
        if row["id"] in mapping:
            raise ValueError(f"Duplicate record: {row['id']}, {row['method']}")
        mapping[row["id"]] = row
    expected = set(expected_ids) if expected_ids is not None else set().union(*(set(m) for m in by_method.values()))
    if not expected or any(set(m) != expected for m in by_method.values()):
        raise ValueError("Incomplete or mismatched paired population; refusing partial metrics")
    ids = list(expected_ids) if expected_ids is not None else sorted(expected)
    samples = [SimpleNamespace(framework=by_method[methods[0]][sid]["framework"]) for sid in ids]
    report = {"n_paired": len(ids), "methods": {}}
    for name, mapping in by_method.items():
        selected = [mapping[sid] for sid in ids]
        if any(row["framework"] != sample.framework for row, sample in zip(selected, samples)):
            raise ValueError("Framework mismatch across methods")
        result = metrics(samples, [r["score"] for r in selected])
        for key in ("elapsed_s", "total_tokens", "prompt_tokens", "completion_tokens", "peak_allocated_mib"):
            if all(key in r for r in selected):
                result[key] = distribution([r[key] for r in selected])
        report["methods"][name] = result
    return report
