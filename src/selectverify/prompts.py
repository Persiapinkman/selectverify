"""Paper prompts; only label-free task and rendered trace fields are consumed."""
SYSTEM = "You are an expert at diagnosing failures of LLM-based multi-agent systems."
INSTR = {
    "base": ("A multi-agent system tried to solve the problem below and FAILED. "
             "Read the execution log and identify the decisive error: the earliest step whose mistake "
             "directly leads to the final failure and is not corrected later. "
             "Steps are 0-indexed. Answer with the step number only."),
    "v2": ("The following conversation log comes from a team of AI agents that produced a WRONG final "
           "answer to the problem. Find the step where things first went wrong, i.e. the step that "
           "introduced the error responsible for the failure (not a later step that merely repeated it). "
           "Steps are 0-indexed."),
}
INSTR["cf"] = INSTR["base"]
INSTR["cfa"] = INSTR["base"]  # 保留 agent 名、抹去内容与问题（agent 先验校准）
STEP_PREFIX = {"base": "The decisive error is at Step ", "v2": "The first wrong step is Step ",
               "cf": "The decisive error is at Step ", "cfa": "The decisive error is at Step "}
AGENT_PREFIX = "The agent responsible for the decisive error is "

LV_INSTR = ("A multi-agent system is solving the problem below (it eventually FAILED). You are shown the "
         "context of one step and the step itself. Judge ONLY the target step: does it contain a mistake "
         "(wrong reasoning, wrong fact, wrong or ineffective action/tool use, ignoring instructions or "
         "evidence, or a wrong plan) that could cause the task to fail? Answer Yes or No.")

def clip(tok, t, b):
    if not b:
        return t
    ids = tok.encode(t, add_special_tokens=False)
    return t if len(ids) <= b else tok.decode(ids[:b]) + " [...]"


def chat(tok, user):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def global_prompt(tok, s, variant, mode, budget):
    lines = []
    for i in range(s.n):
        if variant == "cf":
            lines.append(f"[Step {i}] Agent: [content]")
        elif variant == "cfa":
            lines.append(f"[Step {i}] {s.agents[i]}: [content]")
        else:
            lines.append(f"[Step {i}] {s.agents[i]}: {clip(tok, s.contents[i], budget)}")
    q = "[problem]" if variant in ("cf", "cfa") else s.question
    instr = INSTR[variant]
    if mode == "agent":
        instr = instr.replace("Steps are 0-indexed. Answer with the step number only.", "Answer with the agent name only.") \
                     .replace("Steps are 0-indexed.", "Answer with the agent name only.")
    user = f"{instr}\n\nProblem: {q}\n\nExecution log:\n" + "\n".join(lines)
    return chat(tok, user) + (AGENT_PREFIX if mode == "agent" else STEP_PREFIX[variant])


def lv_prompt(tok, s, k, fb=1500, mb=250, sb=120):
    show = {}
    for i in (0, 1):
        if i < k:
            show[i] = sb
    for i in (k - 2, k - 1):
        if i >= 0:
            show[i] = max(show.get(i, 0), mb)
    lines, last = [], -1
    for i in sorted(show):
        if last >= 0 and i > last + 1:
            lines.append("[...]")
        lines.append(f"[Step {i}] {s.agents[i]}: {clip(tok, s.contents[i], show[i])}")
        last = i
    if last >= 0 and k > last + 1:
        lines.append("[...]")
    tgt = f">>> TARGET [Step {k}] {s.agents[k]}: {clip(tok, s.contents[k], fb)}"
    after = f"[Step {k+1}] {s.agents[k+1]}: {clip(tok, s.contents[k+1], sb)}" if k + 1 < s.n else ""
    user = (f"{LV_INSTR}\n\nProblem: {clip(tok, s.question, 400)}\n\nContext:\n" + "\n".join(lines) +
            f"\n\n{tgt}\n\nWhat happened next:\n{after or '(end of log)'}\n\n"
            f"Does the target Step {k} contain a mistake? Answer Yes or No.")
    return chat(tok, user) + "Answer:"
