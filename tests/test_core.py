import json
from types import SimpleNamespace

import numpy as np
import pytest

from selectverify.algorithm import global_scores, local_scores, metrics, predict, shortlist, validate_config
from selectverify.benchmark import ASSETS, Trace, release_from_row, split_of
from selectverify.controls import parse_step
from selectverify.io import prepare_output, write_json
from selectverify.prompts import global_prompt, lv_prompt
from selectverify.report import summarize


@pytest.fixture
def config():
    return json.loads((ASSETS / "config_best_dev.json").read_text())


def test_frozen_configs(config):
    validate_config(config)
    validate_config(json.loads((ASSETS / "config_lean.json").read_text()))
    assert config["k"] == 3
    assert config["local_weight"] == 2


@pytest.mark.parametrize("change", [{"temperature": 0}, {"local_temperature": -1}, {"quantile": 1.1},
                                     {"k": -1}, {"k": 2.5}, {"agent": float("nan")}, {"global": "unknown"}])
def test_invalid_configs(config, change):
    with pytest.raises(ValueError):
        validate_config(dict(config, **change))


def test_stable_shortlist(config):
    np.testing.assert_array_equal(shortlist([1, 3, 3, 0], config), [0, 1, 2])
    np.testing.assert_array_equal(shortlist([1, 3, 3, 0], dict(config, k=2)), [1, 2])
    np.testing.assert_array_equal(shortlist([0, 1], config), [0, 1])
    np.testing.assert_array_equal(shortlist([0, 1], dict(config, k=0)), [0, 1])


def test_global_formula(config):
    row = {"lj_v2": [-3., -1.], "lcf": [-2., -2.], "la": {"A": -1., "B": -2.}}
    sample = SimpleNamespace(agents=["A", "B"])
    np.testing.assert_allclose(global_scores(row, sample, config), [-2/3-.5, -1.])
    assert row["lj_v2"] == [-3., -1.]


def test_quantile_and_first_event(config):
    values = [0, 3, 6]
    h = 1 / (1 + np.exp(-np.array([-2., -1., 0.])))
    np.testing.assert_allclose(local_scores(values, config), np.log(h))
    expected = np.log(h) + [0, np.log(1-h[0]), np.log(1-h[0])+np.log(1-h[1])]
    np.testing.assert_allclose(local_scores(values, dict(config, aggregation="first")), expected)
    assert np.all(np.isfinite(local_scores([-1e6, 1e6], config)))


def test_no_local_does_not_require_local_cache(config):
    c = dict(config, cf=0., agent=0., local_weight=0.)
    k, candidates = predict({"lj_v2": [0., 1.]}, SimpleNamespace(), c)
    assert k == 1
    assert candidates.tolist() == [1]


def test_framework_macro_not_micro():
    samples = [SimpleNamespace(framework=f) for f in ["smolagents", "smolagents", "debate"]]
    result = metrics(samples, [{"step": True, "agent": True}, {"step": True, "agent": True}, {"step": False, "agent": False}])
    assert result["When"] == 50
    assert result["Who"] == 0
    assert result["micro_step"] == pytest.approx(200/3)


def test_metrics_empty_and_single_agent():
    result = metrics([SimpleNamespace(framework="smolagents")], [{"step": True, "agent": True}])
    assert result["Who"] is None
    assert result["When"] == 100
    with pytest.raises(ValueError):
        metrics([], [])
    with pytest.raises(ValueError):
        metrics([SimpleNamespace(framework="debate")], [])


@pytest.mark.parametrize("raw,style,expected", [("Step: 2.", "short", 2), (r"\boxed{2}", "short", 2),
    ("```json\n2\n```", "short", 2), ('{"step":2,"reason":"x"}', "reason", 2),
    ("7", "short", None), ("2 or 3", "short", None)])
def test_normalized_control_parser(raw, style, expected):
    assert parse_step(raw, [1, 2, 5], style) == expected


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode(errors="replace")

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)


def test_prompt_label_isolation():
    trace = Trace("synthetic", "debate", "Compute a sum", ("0", "1"), ("A", "B"), ("first", "second"))
    assert not hasattr(trace, "gt")
    wrapped = SimpleNamespace(**vars(trace), n=trace.n, gt={"secret": "DO_NOT_SHOW"}, answer="DO_NOT_SHOW")
    tok = Tokenizer()
    for builder in (lambda s: global_prompt(tok, s, "v2", "step", 0), lambda s: lv_prompt(tok, s, 1)):
        assert builder(trace) == builder(wrapped)
        assert "DO_NOT_SHOW" not in builder(wrapped)
    masked = global_prompt(tok, trace, "cf", "step", 0)
    assert "Compute a sum" not in masked and "first" not in masked


def test_split_and_validation():
    assert split_of("synthetic-id") == split_of("synthetic-id")
    assert split_of("anything", 0) == "test"
    assert split_of("anything", 1) == "dev"
    with pytest.raises(ValueError):
        Trace("x", "debate", "q", (), (), ())
    with pytest.raises(ValueError):
        release_from_row({"extras": '{"task": "override"}'})


def test_resume_metadata(tmp_path):
    output = tmp_path / "run.jsonl"
    assert prepare_output(output, {"config": 1}) == set()
    with pytest.raises(ValueError, match="different"):
        prepare_output(output, {"config": 2})
    output.write_text('{"id":"x","method":"a"}\n')
    assert prepare_output(output, {"config": 1}) == {("x", "a")}
    output.write_text('{"id":"x","method":"a"}')
    with pytest.raises(ValueError, match="newline"):
        prepare_output(output, {"config": 1})


def test_report_rejects_partial_and_duplicate(tmp_path):
    output = tmp_path / "scores.jsonl"
    row = {"id": "x", "method": "best_dev", "framework": "debate", "score": {"step": True, "agent": True}}
    output.write_text(json.dumps(row) + "\n")
    assert summarize(output)["methods"]["best_dev"]["When"] == 100
    with pytest.raises(ValueError, match="Incomplete"):
        summarize(output, ["best_dev", "aao"])
    output.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(output)


def test_archived_paper_scores():
    report = summarize(ASSETS / "paper_scores.jsonl", ["aao", "best_dev", "lean", "binary_search", "short", "reason"])
    assert report["n_paired"] == 472
    expected = {"aao": 50.20, "best_dev": 62.40, "lean": 57.98, "binary_search": 34.45, "short": 45.37, "reason": 51.51}
    for name, value in expected.items():
        assert round(report["methods"][name]["When"], 2) == value
