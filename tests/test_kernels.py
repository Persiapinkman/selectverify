"""CPU-only synthetic kernel tests; no checkpoint or Transformers download."""
import json
import math
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None


class Tokenizer:
    pad_token_id = 19

    def convert_tokens_to_ids(self, text):
        return int(text)

    def encode(self, text, **kwargs):
        if text in (" Yes", "Yes", " yes"):
            return [13]
        if text in (" No", "No", " no"):
            return [14]
        if text in (".", "\n", ".\n", "<|im_end|>", " (", ","):
            return [12]
        if text == "A":
            return [10]
        if text == "BB":
            return [11, 11]
        if text.isdigit():
            return [int(ch) for ch in text]
        return [15] * len(text)

    def decode(self, tokens):
        return "x" * len(tokens)

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)


class Cache:
    def crop(self, length):
        self.length = length


class Model:
    def __init__(self):
        self.device = torch.device("cpu")
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        self.calls.append((input_ids.clone(), kwargs))
        width = 1 if kwargs.get("logits_to_keep") == 1 else input_ids.shape[1]
        logits = torch.arange(20, dtype=torch.float32).div(10).repeat(input_ids.shape[0], width, 1)
        return SimpleNamespace(logits=logits, past_key_values=kwargs.get("past_key_values") or Cache())


@unittest.skipIf(torch is None, "Optional PyTorch not installed")
class KernelTests(unittest.TestCase):
    def test_decimal_prefix_probability(self):
        from selectverify.likelihood import step_logprobs
        actual = step_logprobs(Model(), Tokenizer(), [15, 15], 13, "cpu")
        lp = torch.log_softmax(torch.arange(20).float() / 10, 0)
        stop = math.log(1 - lp[:10].exp().sum().item())
        expected = torch.tensor([sum(lp[int(ch)].item() for ch in str(k)) + stop for k in range(13)])
        expected -= torch.logsumexp(expected, 0)
        torch.testing.assert_close(torch.tensor(actual), expected)

    def test_teacher_forced_multitoken_agents(self):
        from selectverify.inference import Meter
        from selectverify.likelihood import agent_logprobs
        meter = Meter(Model())
        meter.full_suffix_logits = True
        actual = agent_logprobs(meter, Tokenizer(), [15, 15], ["A", "BB"], "cpu")
        lp = torch.log_softmax(torch.arange(20).float() / 10, 0)
        expected = torch.tensor([lp[10] + lp[12] + math.log(6), 2 * lp[11] + lp[12] + math.log(6)])
        expected -= torch.logsumexp(expected, 0)
        torch.testing.assert_close(torch.tensor(list(actual.values())), expected)
        self.assertEqual(meter.stages["global"]["prompt_tokens"], 2)
        self.assertEqual(meter.stages["global"]["suffix_tokens"], 3)

    def test_batched_attribution_without_generation(self):
        from selectverify.benchmark import ASSETS, Trace
        from selectverify.inference import attribution
        config = json.loads((ASSETS / "config_best_dev.json").read_text())
        config.update(cf=0., agent=0.)
        trace = Trace("synthetic", "debate", "question", ("0", "1", "2"), ("A", "BB", "A"), ("start", "middle", "end"))
        result = attribution(Model(), Tokenizer(), trace, config, local_batch=2)
        self.assertEqual(result["pred_step"], 2)
        self.assertEqual(result["candidates"], [0, 1, 2])
        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["stages"]["local"]["forwards"], 2)
        self.assertEqual(result["total_tokens"], result["prompt_tokens"] + result["suffix_tokens"])


if __name__ == "__main__":
    unittest.main()
