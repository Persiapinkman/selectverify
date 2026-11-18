"""Matched-evidence generative controls with the paper-normalized parser."""
import re
from .prompts import lv_prompt

def parse_step(raw,candidates,style):
    text=raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text=re.sub(r"^```(?:json|text)?\s*|\s*```$","",text).strip()
    match=None
    if style=="short":
        text=text.strip("$").strip()
        match=re.fullmatch(r"(?:Step\s*:?[ ]*)?(\d+)[.\s]*",text,re.I)
        if match is None:
            match=re.fullmatch(r"\\boxed\{\s*(\d+)\s*\}",text)
    else:
        match=re.search(r"[\"]step[\"]\s*:\s*[\"]?(\d+)",text)
    k=int(match.group(1)) if match else None
    return k if k in candidates else None

class Capture:
    def __init__(self, tok): self.tok = tok
    def __getattr__(self, key): return getattr(self.tok, key)
    def apply_chat_template(self, messages, **kwargs): return messages[-1]["content"]


def generate(model, tok, s, candidates, style):
    packets = []
    for k in candidates:
        text = lv_prompt(Capture(tok), s, k)
        text = "Problem: " + text.split("\n\nProblem: ", 1)[1]
        packets.append(text.split(f"\n\nDoes the target Step {k} contain a mistake?", 1)[0])
    instruction = ("A multi-agent execution failed. The evidence packets describe candidate steps and local context. "
                   "Select the earliest candidate whose mistake led to the final failure and was not corrected later, "
                   f"rather than a later repetition or symptom. Select exactly one step from {candidates}. ")
    instruction += ("Output only its integer step index, with no explanation." if style == "short" else
                    "Output JSON with keys step (the integer) and reason (a brief explanation).")
    text = tok.apply_chat_template([
        {"role": "system", "content": "You are an expert at diagnosing failures of LLM-based multi-agent systems."},
        {"role": "user", "content": instruction + "\n\n" + "\n\n".join(packets)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids.to(str(model.device))
    assert ids.shape[1] <= 30000
    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                         max_new_tokens=16 if style == "short" else 512,
                         pad_token_id=tok.pad_token_id, logits_to_keep=1)
    raw = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    k = parse_step(raw, candidates, style)
    return dict(pred_step=k, raw=raw, rerank_prompt_tokens=ids.shape[1], completion_tokens=out.shape[1]-ids.shape[1])
