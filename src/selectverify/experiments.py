"""Fresh inference and signal collection, with no dependency on historical caches."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from .algorithm import config_id, shortlist
from .benchmark import load
from .io import fingerprint, prepare_output, read_jsonl, sha256_file, write_json
from .report import summarize


def collect(model, tok, sample):
    import torch
    from .likelihood import agent_logprobs, step_logprobs
    from .inference import Meter, fit_global
    from .prompts import lv_prompt
    meter = Meter(model)
    result = {"n": sample.n, "tokens": 0}
    for key, variant in (("lj", "base"), ("lj_v2", "v2"), ("lcf", "cf")):
        ids, _ = fit_global(tok, sample, variant, "step")
        result["tokens"] += len(ids)
        result[key] = step_logprobs(meter, tok, ids, sample.n, str(model.device))
    if sample.multi_agent:
        meter.full_suffix_logits = True
        for key, variant in (("la", "base"), ("la_cf", "cfa")):
            ids, _ = fit_global(tok, sample, variant, "agent")
            result["tokens"] += len(ids)
            result[key] = agent_logprobs(meter, tok, ids, sorted(set(sample.agents)), str(model.device))
        meter.full_suffix_logits = False
    yes = [tok.encode(x, add_special_tokens=False)[0] for x in (" Yes", "Yes", " yes")]
    no = [tok.encode(x, add_special_tokens=False)[0] for x in (" No", "No", " no")]
    result["lv"] = []
    for k in range(sample.n):
        ids = tok.encode(lv_prompt(tok, sample, k), add_special_tokens=False)[-30000:]
        result["tokens"] += len(ids)
        output = meter(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
        lp = torch.log_softmax(output.logits[0, -1].float(), -1)
        result["lv"].append(torch.logsumexp(lp[yes], 0).item() - torch.logsumexp(lp[no], 0).item())
    return result


def run(args):
    import torch
    import transformers
    import whowhen_eval
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .inference import attribution, baseline
    from .algorithm import validate_config

    names = ["signals"] if args.command == "collect" else args.methods.split(",")
    allowed = {"signals"} if args.command == "collect" else {"lean", "best_dev", "aao", "binary_search", "short", "reason"}
    if not names or len(names) != len(set(names)) or not set(names) <= allowed:
        raise ValueError("Unknown or duplicate methods")
    samples = [s for s in load(args.data_root, args.manifest) if args.split == "all" or s.split == args.split]
    if args.per_framework:
        samples = [s for fw in sorted({s.framework for s in samples})
                   for s in sorted((s for s in samples if s.framework == fw),
                                   key=lambda s: hashlib.sha256(s.sid.encode()).hexdigest())[:args.per_framework]]
    if not samples:
        raise ValueError("No evaluable samples selected")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a CUDA environment for paper measurements")
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    configs = {}
    needed = set(names) & {"lean", "best_dev"}
    if set(names) & {"short", "reason"}:
        needed.add("best_dev")
    for name in sorted(needed):
        config = json.loads((args.config_dir / f"config_{name}.json").read_text(encoding="utf-8"))
        validate_config(config)
        configs[name] = config
    candidate_source = {}
    if set(names) & {"short", "reason"}:
        if not args.paired_source:
            raise ValueError("Matched controls require --paired-source from a frozen best_dev run")
        for row in read_jsonl(args.paired_source):
            if row["method"] == "best_dev":
                if row["id"] in candidate_source:
                    raise ValueError("Duplicate best_dev source prediction")
                if row.get("config_id") != config_id(configs["best_dev"]):
                    raise ValueError("Candidate source uses a different SV configuration")
                candidate_source[row["id"]] = row["candidates"]
        if any(s.sid not in candidate_source for s in samples):
            raise ValueError("Candidate source is incomplete")
    raw_rows, taxonomy, builder = {}, None, None
    if "aao" in names:
        from whowhen_eval.prompts import load_taxonomy
        wanted = {s.sid for s in samples}
        raw_rows = {r["id"]: r for r in read_jsonl(args.data_root / "data/text.jsonl") if r["id"] in wanted}
        taxonomy = load_taxonomy(args.data_root)
    if "binary_search" in names:
        from .binary import binary_prompt_builder
        if not args.binary_source or not args.binary_source.is_file():
            raise ValueError("binary_search requires --binary-source from the upstream licensed implementation")
        builder = binary_prompt_builder(args.binary_source)
    options = {"revision": args.revision} if args.revision else {}
    tok = AutoTokenizer.from_pretrained(args.model, **options)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    if tok.pad_token_id is None:
        raise ValueError("Tokenizer needs a padding or EOS token")
    for digit in range(10):
        ids = tok.encode(str(digit), add_special_tokens=False)
        if ids != [tok.convert_tokens_to_ids(str(digit))]:
            raise ValueError("The paper likelihood kernel requires single-token decimal digits")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa", **options).to(device).eval()
    torch.set_num_threads(4)
    local = Path(args.model).is_dir()
    source_root = Path(__file__).parent
    upstream_root = Path(whowhen_eval.__file__).parent
    model_config = model.config.to_dict()
    for key in ("_name_or_path", "name_or_path"):
        model_config.pop(key, None)
    meta = {"protocol_version": 1, "methods": names, "configs": configs, "split": args.split,
            "ids": [s.sid for s in samples], "local_batch": args.local_batch,
            "model_id": "local-checkpoint" if local else args.model,
            "model_revision": args.revision or getattr(model.config, "_commit_hash", None),
            "model_config_sha256": fingerprint(model_config),
            "local_checkpoint_file_stats": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(Path(args.model).glob("*.safetensors"))} if local else None,
            "local_config_sha256": sha256_file(Path(args.model) / "config.json") if local else None,
            "tokenizer_sha256": fingerprint(tok.get_vocab()),
            "model_type": model.config.model_type, "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "dtype": args.dtype, "attention": "sdpa", "thinking": False,
            "manifest_sha256": sha256_file(args.manifest),
            "rendered_data_sha256": fingerprint([asdict(s) for s in samples]),
            "source_sha256": {p.name: sha256_file(p) for p in sorted(source_root.glob("*.py"))},
            "benchmark_source_sha256": {str(p.relative_to(upstream_root)): sha256_file(p) for p in sorted(upstream_root.rglob("*.py"))},
            "binary_source_sha256": sha256_file(args.binary_source) if builder else None,
            "candidate_source_sha256": sha256_file(args.paired_source) if candidate_source else None,
            "timing": "Prompt construction through parsing, synchronized on CUDA; excludes loading, label scoring and I/O"}
    done = prepare_output(args.output, meta)
    expected = {(s.sid, name) for s in samples for name in names}
    if not done <= expected:
        raise ValueError("Output contains records outside this experiment")
    with torch.inference_mode(), args.output.open("a", encoding="utf-8") as stream:
        warm = tok("Read the execution log and identify the error.", return_tensors="pt").to(device)
        model(**warm, logits_to_keep=1)
        model.generate(**warm, max_new_tokens=3, do_sample=False, pad_token_id=tok.pad_token_id, logits_to_keep=1)
        sync()
        for index, sample in enumerate(samples):
            order = names[index % len(names):] + names[:index % len(names)]
            for name in order:
                if (sample.sid, name) in done:
                    continue
                trace = sample.trace
                sync()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                if name == "signals":
                    result = collect(model, tok, trace)
                elif name == "aao":
                    result = baseline(model, tok, trace, raw_rows[sample.sid], taxonomy, args.data_root)
                elif name in ("short", "reason"):
                    from .controls import generate
                    config = configs["best_dev"]
                    global_result = attribution(model, tok, trace, dict(config, local_weight=0.), 1)
                    candidates = candidate_source[sample.sid]
                    if shortlist(global_result["global_scores"], config).tolist() != candidates:
                        raise ValueError("Fresh shortlist differs from the matched-candidate source")
                    result = generate(model, tok, trace, candidates, name)
                    result.update(candidates=candidates, prompt_tokens=global_result["prompt_tokens"] + result["rerank_prompt_tokens"],
                                  total_tokens=global_result["total_tokens"] + result["rerank_prompt_tokens"] + result["completion_tokens"])
                elif name == "binary_search":
                    from .binary import LocalClient, run_binary
                    client = LocalClient(model, tok)
                    result = run_binary(client, trace, builder)
                    result.update(calls=client.calls, prompt_tokens=sum(c["prompt_tokens"] for c in client.calls),
                                  completion_tokens=sum(c["completion_tokens"] for c in client.calls))
                    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
                else:
                    result = attribution(model, tok, trace, configs[name], args.local_batch)
                    result["config_id"] = config_id(configs[name])
                sync()
                result.update(id=sample.sid, method=name, framework=sample.framework, n_steps=sample.n,
                              elapsed_s=time.perf_counter() - started,
                              peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.)
                if name == "aao":
                    from whowhen_eval.score import score
                    result["score"] = score(result["pred"], sample.gt, sample.framework)
                elif name != "signals":
                    k = result["pred_step"]
                    result["score"] = sample.judge(k) if k is not None else {"step": False, "agent": False, "mode": False}
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                print(f"{index + 1}/{len(samples)} {name}: {result['elapsed_s']:.3f}s", flush=True)
    if args.command == "run":
        write_json(args.output.with_suffix(".summary.json"), summarize(args.output, names, [s.sid for s in samples]))
