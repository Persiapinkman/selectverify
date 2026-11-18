"""Original 784-configuration dev-only search and fixed test ablations."""
import itertools
import json
import math
from collections import Counter
import numpy as np
from .algorithm import config_id, metrics, predict
from .benchmark import load
from .io import records

def screen(out_dir, cache_dir, data_root, manifest):
    samples = [s for s in load(data_root, manifest) if s.split == "dev"]
    signals = records(sorted(cache_dir.glob("signals*.jsonl")))
    aao = records(sorted(cache_dir.glob("all_at_once_4b*.jsonl")))
    assert all(s.sid in signals and s.sid in aao for s in samples)
    outcomes = {s.sid: [s.judge(i) for i in range(s.n)] for s in samples}
    counts = Counter(s.framework for s in samples)
    weights = np.asarray([1 / (len(counts) * counts[s.framework]) for s in samples])
    rows = []
    for variant, cf, agent, temperature in itertools.product(("lj", "lj_v2"), (0., .5), (0., .5), (1., 3.)):
        base = dict(global_=variant, cf=cf, agent=agent, temperature=temperature)
        base["global"] = base.pop("global_")
        for k, aggregation, quantile, lw in itertools.product((3, 5, 8, 0), ("point", "first"), (.5, .8), (.5, 1., 2.)):
            config = dict(base, k=k, aggregation=aggregation, quantile=quantile, local_temperature=3., local_weight=lw)
            predicted = [predict(signals[s.sid], s, config) for s in samples]
            hit = np.asarray([outcomes[s.sid][p]["step"] for s, (p, _) in zip(samples, predicted)])
            rows.append({"config": config, "dev_When": float(hit @ weights * 100), "dev_micro_step": float(hit.mean() * 100),
                         "mean_verified_steps": float(np.mean([len(c) for _, c in predicted]))})
        config = dict(base, k=1, aggregation="point", quantile=.5, local_temperature=3., local_weight=0.)
        hit = np.asarray([outcomes[s.sid][predict(signals[s.sid], s, config)[0]]["step"] for s in samples])
        rows.append({"config": config, "dev_When": float(hit @ weights * 100), "dev_micro_step": float(hit.mean() * 100), "mean_verified_steps": 0.})
    rows.sort(key=lambda r: (-round(r["dev_When"], 10), r["mean_verified_steps"], r["config"]["agent"] != 0, r["config"]["cf"] != 0, config_id(r["config"])))
    best = rows[0]
    lean = next(r for r in rows if r["config"]["agent"] == 0 and r["config"]["cf"] == 0 and 0 < r["config"]["k"] <= 5)
    calibrated = next(r for r in rows if r["config"]["agent"] == 0 and 0 < r["config"]["k"] <= 5)
    selected = {"best_dev": best, "lean": lean, "calibrated": calibrated}
    result = {"protocol": "Existing fixed Pro manifest; dev-only selection; no test accuracy accessed.",
              "dev_n": len(samples), "grid_size": len(rows), "aao_dev": metrics(samples, [aao[s.sid]["score"] for s in samples]),
              "selected": selected, "top20": rows[:20]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screen_dev.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, result_row in selected.items():
        (out_dir / f"config_{name}.json").write_text(json.dumps(result_row["config"], indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "top20"}, indent=2), flush=True)


def evaluate(out_dir, cache_dir, data_root, manifest):
    samples = [s for s in load(data_root, manifest) if s.split == "test"]
    signals = records(sorted(cache_dir.glob("signals*.jsonl")))
    aao = records(sorted(cache_dir.glob("all_at_once_4b*.jsonl")))
    assert all(s.sid in signals and s.sid in aao for s in samples)
    variants = {}
    for name in ("lean", "best_dev"):
        config = json.loads((out_dir / f"config_{name}.json").read_text())
        variants[name] = config
        variants[name + "_no_local"] = dict(config, local_weight=0.)
        variants[name + "_all_steps"] = dict(config, k=0)
        variants[name + "_alternate_aggregation"] = dict(config, aggregation="point" if config["aggregation"] == "first" else "first")
        if config["agent"]:
            variants[name + "_no_agent"] = dict(config, agent=0.)
        if config["cf"]:
            variants[name + "_no_cf"] = dict(config, cf=0.)
    report = {"protocol": "All configurations selected on dev; fixed-parameter test ablations; cache-based accuracy, not measured latency.",
              "baseline": metrics(samples, [aao[s.sid]["score"] for s in samples]), "methods": {}}
    per_case = []
    baseline_hit = np.asarray([aao[s.sid]["score"]["step"] for s in samples], dtype=bool)
    for name, config in variants.items():
        predicted = [predict(signals[s.sid], s, config) for s in samples]
        outcomes = [s.judge(k) for s, (k, _) in zip(samples, predicted)]
        result = metrics(samples, outcomes)
        hits = np.asarray([x["step"] for x in outcomes], dtype=bool)
        corrected = int(np.sum(hits & ~baseline_hit))
        regressed = int(np.sum(~hits & baseline_hit))
        discordant = corrected + regressed
        pvalue = min(1., 2 * sum(math.comb(discordant, j) for j in range(min(corrected, regressed) + 1)) / 2 ** discordant)
        result.update(config=config, corrected=corrected, regressed=regressed, mcnemar_p=pvalue,
                      mean_verified_steps=float(np.mean([len(c) for _, c in predicted])) if config["local_weight"] else 0.)
        report["methods"][name] = result
        for s, (k, c), outcome in zip(samples, predicted, outcomes):
            per_case.append({"id": s.sid, "method": name, "config_id": config_id(config), "pred_step": k,
                             "candidates": c.tolist(), "score": outcome})
    by_method = {name: {r["id"]: r for r in per_case if r["method"] == name} for name in variants}
    rng = np.random.default_rng(20261116)
    groups = [[i for i, s in enumerate(samples) if s.framework == fw] for fw in sorted({s.framework for s in samples})]
    bootstrap = [rng.choice(indices, size=(10000, len(indices)), replace=True) for indices in groups]
    for name in ("lean", "best_dev"):
        result = report["methods"][name]
        hits = np.asarray([by_method[name][s.sid]["score"]["step"] for s in samples], dtype=bool)
        difference = hits.astype(float) - baseline_hit
        deltas = np.mean([difference[indices].mean(axis=1) for indices in bootstrap], axis=0) * 100
        result["When_delta_ci95"] = np.quantile(deltas, [.025, .975]).tolist()
        result["candidate_recall_micro"] = 100 * float(np.mean([bool(set(by_method[name][s.sid]["candidates"]) & s.gold_steps()) for s in samples]))
        result["ablation_pairs"] = {}
        for other in variants:
            if not other.startswith(name + "_"):
                continue
            removed = np.asarray([by_method[other][s.sid]["score"]["step"] for s in samples], dtype=bool)
            corrected = int(np.sum(hits & ~removed))
            regressed = int(np.sum(~hits & removed))
            n = corrected + regressed
            result["ablation_pairs"][other] = {"full_corrected": corrected, "full_regressed": regressed,
                "mcnemar_p": min(1., 2 * sum(math.comb(n, j) for j in range(min(corrected, regressed) + 1)) / 2 ** n)}
    (out_dir / "test_cached.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "test_cached_percase.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_case), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
