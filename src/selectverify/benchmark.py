"""Text-only Who&When Pro adapter. Scoring labels never enter Trace inputs."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .io import read_jsonl

ASSETS = Path(__file__).resolve().parent / "assets"
MULTI_AGENT = {"magentic-one", "mathchat", "debate", "macnet", "dylan", "metagpt"}
HEADER = re.compile(r"^Step (\S+) \| Agent: ([^\n]*)\n?")


@dataclass(frozen=True)
class Trace:
    sid: str
    framework: str
    question: str
    coords: tuple[str, ...]
    agents: tuple[str, ...]
    contents: tuple[str, ...]

    def __post_init__(self):
        if not self.coords or not len(self.coords) == len(self.agents) == len(self.contents):
            raise ValueError("Trace requires nonempty, equally sized step fields")

    @property
    def n(self):
        return len(self.coords)

    @property
    def multi_agent(self):
        return self.framework in MULTI_AGENT


@dataclass
class ProSample:
    trace: Trace
    benchmark: str
    gt: dict = field(repr=False)
    split: str = "test"

    def __getattr__(self, name):
        return getattr(self.trace, name)

    def judge(self, k):
        from whowhen_eval.score import score
        if not 0 <= k < self.n:
            raise ValueError("Predicted step is out of range")
        return score({"agent_name": self.agents[k], "step_coord": self.coords[k], "error_mode": ""}, self.gt, self.framework)

    def gold_steps(self):
        return {k for k in range(self.n) if self.judge(k)["step"]}


def split_of(sid, dev_frac=.3):
    if not 0 <= dev_frac <= 1:
        raise ValueError("dev_frac must be between zero and one")
    return "dev" if int(hashlib.md5(sid.encode()).hexdigest(), 16) % 1000 < dev_frac * 1000 else "test"


def release_from_row(row, split="text", data_root=None):
    if split != "text":
        raise ValueError("This release supports the paper's text modality only")
    result = {key: row.get(key) for key in ("id", "framework", "benchmark")}
    result["modality"] = "text"
    for key, default in (("task", {}), ("trajectory", []), ("ground_truth", {}), ("extras", {})):
        value = row.get(key)
        value = json.loads(value) if isinstance(value, str) and value else value
        result[key] = value if value is not None else default
    extras = result.pop("extras")
    if isinstance(extras, dict):
        if set(extras) & set(result):
            raise ValueError("Extras must not override reserved release fields")
        result.update(extras)
    elif extras:
        result["extras"] = extras
    return result


def render_row(row):
    from whowhen_eval.render import get_renderer
    from whowhen_eval.render.base import TASK_ANCHOR
    release = release_from_row(row)
    rendered = get_renderer(row["framework"])(release)
    coords, agents, contents = [], [], []
    for block in rendered.blocks:
        if block.coord in (None, "", TASK_ANCHOR) or not block.text:
            continue
        match = HEADER.match(block.text)
        agent, body = (match.group(2).strip(), block.text[match.end():]) if match else ("agent", block.text)
        coords.append(str(block.coord))
        agents.append(agent)
        contents.append(body.strip())
    if not coords:
        return None
    trace = Trace(row["id"], row["framework"], (release.get("task") or {}).get("query") or "",
                  tuple(coords), tuple(agents), tuple(contents))
    return ProSample(trace, row["benchmark"], release["ground_truth"])


def load(data_root, manifest=None, dev_frac=.3):
    manifest = Path(manifest) if manifest else ASSETS / "pro_manifest.json"
    wanted = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(wanted, list) or any(not isinstance(sid, str) for sid in wanted) or len(wanted) != len(set(wanted)):
        raise ValueError("Manifest must be a list of unique string IDs")
    want_set, seen, samples = set(wanted), set(), {}
    for row in read_jsonl(Path(data_root) / "data/text.jsonl"):
        if row["id"] not in want_set:
            continue
        if row["id"] in seen:
            raise ValueError("Duplicate dataset ID")
        seen.add(row["id"])
        sample = render_row(row)
        if sample is not None and sample.gold_steps():
            sample.split = split_of(sample.sid, dev_frac)
            samples[sample.sid] = sample
    if seen != want_set:
        raise ValueError(f"Dataset is missing {len(want_set - seen)} manifest IDs")
    return [samples[sid] for sid in wanted if sid in samples]
