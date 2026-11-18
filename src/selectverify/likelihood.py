"""Digit-prefix and teacher-forced agent likelihoods from the paper implementation."""
import math
import torch

@torch.no_grad()
def step_logprobs(model, tok, prompt_ids, n_steps, device):
    digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    ids = torch.tensor([prompt_ids], device=device)
    out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    base_len = ids.shape[1]
    memo = {"": torch.log_softmax(out.logits[0, -1].float(), -1)}

    def dist(prefix):
        if prefix not in memo:
            cache.crop(base_len)
            toks = [digit_ids[int(ch)] for ch in prefix]
            o = model(input_ids=torch.tensor([toks], device=device), past_key_values=cache, use_cache=True)
            memo[prefix] = torch.log_softmax(o.logits[0, -1].float(), -1)
        return memo[prefix]

    res = []
    for k in range(n_steps):
        s = str(k)
        lp = sum(dist(s[:i])[digit_ids[int(ch)]].item() for i, ch in enumerate(s))
        p_digit = torch.logsumexp(dist(s)[digit_ids], 0).item()
        res.append(lp + math.log(max(1e-12, 1 - math.exp(min(0.0, p_digit)))))
    cache.crop(base_len)
    z = torch.logsumexp(torch.tensor(res), 0).item()
    return [x - z for x in res]


@torch.no_grad()
def agent_logprobs(model, tok, prompt_ids, names, device):
    ids = torch.tensor([prompt_ids], device=device)
    out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    base_len = ids.shape[1]
    first = torch.log_softmax(out.logits[0, -1].float(), -1)
    stop_ids = [tok.encode(x, add_special_tokens=False)[0] for x in (".", "\n", ".\n", "<|im_end|>", " (", ",")]
    res = {}
    for nm in names:
        t = tok.encode(nm, add_special_tokens=False)
        cache.crop(base_len)
        o = model(input_ids=torch.tensor([t], device=device), past_key_values=cache, use_cache=True)
        lps = torch.log_softmax(o.logits[0].float(), -1)
        lp = first[t[0]].item() + sum(lps[i - 1, t[i]].item() for i in range(1, len(t)))
        lp += torch.logsumexp(lps[-1, stop_ids], 0).item()
        res[nm] = lp
    cache.crop(base_len)
    z = torch.logsumexp(torch.tensor(list(res.values())), 0).item()
    return {k: v - z for k, v in res.items()}
