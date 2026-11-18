"""Frozen SelectVerify ranking, fusion and framework-macro metrics."""
import hashlib
import json
import numpy as np

SINGLE = {"smolagents", "alfagent", "mathchat", "pixelcraft", "eva"}


def validate_config(config):
    if config["global"] not in ("lj", "lj_v2") or config["aggregation"] not in ("point", "first"):
        raise ValueError("Unknown global prompt or aggregation")
    if type(config["k"]) is not int or config["k"] < 0:
        raise ValueError("k must be a nonnegative integer; zero means all steps")
    if not 0 <= config["quantile"] <= 1:
        raise ValueError("quantile must be between zero and one")
    for key in ("cf", "agent", "temperature", "local_temperature", "local_weight", "quantile"):
        if not np.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if config["temperature"] <= 0 or config["local_temperature"] <= 0:
        raise ValueError("Temperatures must be positive")
    if min(config["cf"], config["agent"], config["local_weight"]) < 0:
        raise ValueError("Weights must be nonnegative")


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def global_scores(row, sample, config):
    q = np.asarray(row[config["global"]], dtype=float).copy()
    if config["cf"]:
        q -= config["cf"] * np.asarray(row["lcf"])
    q /= config["temperature"]
    if config["agent"] and "la" in row:
        q += config["agent"] * np.asarray([row["la"][a] for a in sample.agents])
    return q


def shortlist(q, config):
    k = config["k"] or len(q)
    return np.asarray(sorted(sorted(range(len(q)), key=lambda i: (-q[i], i))[:k]))


def local_scores(values, config):
    values = np.asarray(values, dtype=float)
    ordered = np.sort(values)
    center = ordered[min(len(values) - 1, int(config["quantile"] * (len(values) - 1) + .5))]
    z = np.clip((values - center) / config["local_temperature"], -50, 50)
    log_h = -np.logaddexp(0, -z)
    if config["aggregation"] == "first":
        log_survive = -np.logaddexp(0, z)
        log_h += np.r_[0., np.cumsum(log_survive[:-1])]
    return log_h


def predict(row, sample, config):
    q = global_scores(row, sample, config)
    if not config["local_weight"]:
        return int(np.argmax(q)), np.asarray([int(np.argmax(q))])
    candidates = shortlist(q, config)
    lv = np.asarray(row["lv"])[candidates]
    scores = q[candidates] + config["local_weight"] * local_scores(lv, config)
    return int(candidates[np.argmax(scores)]), candidates


def metrics(samples, outcomes):
    if not samples or len(samples) != len(outcomes):
        raise ValueError("Metrics require equal, nonempty sample and outcome lists")
    groups = {}
    for sample, outcome in zip(samples, outcomes):
        groups.setdefault(sample.framework, []).append(outcome)
    who = [np.mean([x["agent"] for x in rows]) for fw, rows in groups.items() if fw not in SINGLE]
    return {
        "n": len(samples),
        "When": 100 * float(np.mean([np.mean([x["step"] for x in rows]) for rows in groups.values()])),
        "Who": 100 * float(np.mean(who)) if who else None,
        "micro_step": 100 * float(np.mean([x["step"] for x in outcomes])),
        "per_framework": {fw: {"n": len(rows), "When": 100 * float(np.mean([x["step"] for x in rows]))} for fw, rows in groups.items()},
    }
