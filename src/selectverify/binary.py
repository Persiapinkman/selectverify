"""Same-checkpoint Binary Search adapter; upstream prompt code is supplied separately."""
import ast
import hashlib
import random
import time
import torch

def binary_prompt_builder(source):
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_construct_binary_search_prompt")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


class LocalClient:
    def __init__(self, model, tokenizer, chief_max_new_tokens=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.chief_max_new_tokens = chief_max_new_tokens
        self.calls = []
        self.stage = ""

    def render(self, prompt, system=None):
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def chat(self, prompt, system=None, temperature=None, max_tokens=768):
        started = time.perf_counter()
        text = self.render(prompt, system)
        full_ids = self.tokenizer(text, return_tensors="pt").input_ids
        if self.stage.startswith("binary_"):
            assert full_ids.shape[1] <= 30000, "Binary segment must preserve both halves"
        ids = full_ids[:, -30000:].to(str(self.model.device))
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_tokens,
                                  do_sample=False, pad_token_id=self.tokenizer.pad_token_id, logits_to_keep=1)
        completion = out[0, ids.shape[1]:]
        raw = self.tokenizer.decode(completion, skip_special_tokens=True)
        self.calls.append({"stage": self.stage, "prompt_tokens": ids.shape[1],
                           "completion_tokens": len(completion), "max_new_tokens": max_tokens,
                           "truncated_prompt_tokens": full_ids.shape[1] - ids.shape[1],
                           "hit_generation_limit": len(completion) == max_tokens,
                           "elapsed_s": time.perf_counter() - started,
                           "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), "raw": raw})
        return raw


def run_binary(client, sample, build_prompt):
    lo, hi = 0, sample.n - 1
    seed = int(hashlib.sha256(("binary-seed0:" + sample.sid).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    decisions = []
    while lo < hi:
        mid = (lo + hi) // 2
        system = "You are an AI assistant specializing in localizing errors in conversation segments."
        budget = 0
        while True:
            lines = []
            for k in range(lo, hi + 1):
                content = sample.contents[k]
                if budget:
                    ids = client.tokenizer.encode(content, add_special_tokens=False)
                    if len(ids) > budget:
                        content = client.tokenizer.decode(ids[:budget]) + " [...]"
                lines.append(f"{sample.agents[k]}: {content}")
            prompt = build_prompt(sample.question, "", "\n".join(lines), f"from step {lo} to step {hi}",
                                  f"from step {lo} to step {mid}", f"from step {mid+1} to step {hi}")
            prompt = prompt.replace("The Answer for the problem is: \n", "", 1)
            length = len(client.tokenizer.encode(client.render(prompt, system), add_special_tokens=False))
            if length <= 30000:
                break
            if budget == 32:
                raise ValueError(f"Binary prompt cannot preserve all steps: {sample.sid}")
            budget = max(32, (budget or 4096) // 2)
        client.stage = f"binary_{lo}_{hi}"
        raw = client.chat(prompt, system=system, max_tokens=512)
        client.calls[-1]["per_step_token_budget"] = budget
        if not raw.strip():
            return {"pred_step": None, "pred_agent": None, "decisions": decisions, "empty_output": True}
        low = raw.lower()
        fallback = "upper half" not in low and "lower half" not in low
        upper = (rng.randint(0, 1) == 0) if fallback else "upper half" in low
        decisions.append({"lo": lo, "hi": hi, "mid": mid, "upper": upper, "random_fallback": fallback})
        if upper:
            hi = mid
        else:
            lo = mid + 1
    return {"pred_step": lo, "pred_agent": sample.agents[lo], "decisions": decisions,
            "random_fallbacks": sum(d["random_fallback"] for d in decisions)}
