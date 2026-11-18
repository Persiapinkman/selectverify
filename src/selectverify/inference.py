"""Measured inference kernels; GT scoring and I/O stay outside these functions."""
import numpy as np
import torch
from .algorithm import global_scores, local_scores, shortlist
from .likelihood import agent_logprobs, step_logprobs
from .prompts import global_prompt, lv_prompt

class Meter:
    def __init__(self, model):
        self.model = model
        self.full_suffix_logits = False
        self.stages = {}
        self.stage = "global"

    def __call__(self, **kwargs):
        ids = kwargs["input_ids"]
        suffix = kwargs.get("past_key_values") is not None
        counters = self.stages.setdefault(self.stage, {"prompt_tokens": 0, "suffix_tokens": 0, "padded_tokens": 0, "forwards": 0})
        actual = int(kwargs["attention_mask"].sum().item()) if "attention_mask" in kwargs and not suffix else ids.numel()
        counters["suffix_tokens" if suffix else "prompt_tokens"] += actual
        counters["padded_tokens"] += ids.numel() - actual
        counters["forwards"] += 1
        kwargs["logits_to_keep"] = 0 if self.full_suffix_logits and suffix else 1
        return self.model(**kwargs)


def fit_global(tok, sample, variant, mode):
    budget = 0
    while True:
        ids = tok.encode(global_prompt(tok, sample, variant, mode, budget), add_special_tokens=False)
        if len(ids) <= 30000:
            return ids, budget
        if budget == 32:
            raise ValueError(f"Global prompt cannot fit: {sample.sid}")
        budget = max(32, (budget or 4096) // 2)


def attribution(model, tok, sample, config, local_batch):
    meter = Meter(model)
    row = {}
    variant = "base" if config["global"] == "lj" else "v2"
    ids, budget = fit_global(tok, sample, variant, "step")
    row[config["global"]] = step_logprobs(meter, tok, ids, sample.n, str(model.device))
    if config["cf"]:
        meter.stage = "cf"
        ids, _ = fit_global(tok, sample, "cf", "step")
        row["lcf"] = step_logprobs(meter, tok, ids, sample.n, str(model.device))
    if config["agent"] and sample.multi_agent:
        meter.stage = "agent"
        meter.full_suffix_logits = True
        ids, _ = fit_global(tok, sample, "base", "agent")
        row["la"] = agent_logprobs(meter, tok, ids, sorted(set(sample.agents)), str(model.device))
        meter.full_suffix_logits = False
    q = global_scores(row, sample, config)
    candidates = shortlist(q, config) if config["local_weight"] else np.asarray([int(np.argmax(q))])
    values = []
    if config["local_weight"]:
        meter.stage = "local"
        yes = [tok.encode(x, add_special_tokens=False)[0] for x in (" Yes", "Yes", " yes")]
        no = [tok.encode(x, add_special_tokens=False)[0] for x in (" No", "No", " no")]
        for start in range(0, len(candidates), local_batch):
            batch = [tok.encode(lv_prompt(tok, sample, int(k)), add_special_tokens=False) for k in candidates[start:start + local_batch]]
            width = max(map(len, batch))
            ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long, device=str(model.device))
            mask = torch.zeros_like(ids)
            for j, tokens in enumerate(batch):
                ids[j, -len(tokens):] = torch.tensor(tokens, device=str(model.device))
                mask[j, -len(tokens):] = 1
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            lp = torch.log_softmax(meter(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).logits[:, -1].float(), -1)
            values.extend((torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1)).tolist())
        scores = q[candidates] + config["local_weight"] * local_scores(values, config)
        chosen = int(candidates[np.argmax(scores)])
    else:
        chosen = int(np.argmax(q))
    prompt_tokens = sum(x["prompt_tokens"] for x in meter.stages.values())
    suffix_tokens = sum(x["suffix_tokens"] for x in meter.stages.values())
    return {"pred_step": chosen, "candidates": candidates.tolist(), "local_values": values,
            "global_scores": q.tolist(), "global_budget": budget, "stages": meter.stages,
            "prompt_tokens": prompt_tokens, "suffix_tokens": suffix_tokens, "completion_tokens": 0,
            "total_tokens": prompt_tokens + suffix_tokens,
            "padded_tokens": sum(x["padded_tokens"] for x in meter.stages.values()),
            "forwards": sum(x["forwards"] for x in meter.stages.values())}


def baseline(model, tok, sample, raw_row, taxonomy, data_root):
    from .benchmark import release_from_row
    from whowhen_eval.parse import parse_all_at_once
    from whowhen_eval.prompts import all_at_once, parts_to_text
    from whowhen_eval.render import get_renderer

    release = release_from_row(raw_row, "text", data_root)
    rendered = get_renderer(sample.framework)(release)
    text = parts_to_text(all_at_once(rendered, problem=sample.question, taxonomy=taxonomy))
    prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_ids = tok(prompt, return_tensors="pt").input_ids
    ids = full_ids[:, -30000:].to(str(model.device))
    output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=512, do_sample=False,
                            pad_token_id=tok.pad_token_id, logits_to_keep=1)
    completion = output[0, ids.shape[1]:]
    raw = tok.decode(completion, skip_special_tokens=True)
    pred = vars(parse_all_at_once(raw, taxonomy.codes))
    return {"pred": pred, "raw": raw, "prompt_tokens": ids.shape[1], "suffix_tokens": 0,
            "completion_tokens": len(completion), "total_tokens": ids.shape[1] + len(completion),
            "truncated_prompt_tokens": full_ids.shape[1] - ids.shape[1], "padded_tokens": 0}


def distribution(values):
    return {"mean": float(np.mean(values)), "p50": float(np.median(values)), "p90": float(np.quantile(values, .9))}
