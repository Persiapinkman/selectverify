# SelectVerify

**Select, Then Verify: Budgeted Failure Attribution for Tool-Using Agents**

SelectVerify identifies the decisive error in a failed agent trace using a frozen language model: rank steps globally, select up to three candidates, verify each in a local context, and fuse the scores. SV-Lite omits position calibration and agent scoring and uses first-event aggregation.

[![SelectVerify framework](assets/framework.png)](assets/framework.pdf)

[Vector figure (PDF)](assets/framework.pdf)

## Paper results

Results on Who&When Pro. When and Who are framework averages; micro-Step is pooled accuracy. Values are percentages.

| Method | When | Who | micro-Step |
| :--- | ---: | ---: | ---: |
| All-at-Once | 50.20 | 57.20 | 54.24 |
| MASPrism | 31.32 | 51.07 | 28.60 |
| Binary Search | 34.45 | 27.89 | 38.98 |
| Full-signal attributor | 58.46 | 60.15 | 61.65 |
| SV-Lite | 57.98 | 60.27 | 59.96 |
| **SelectVerify** | **62.40** | **61.92** | **64.62** |

## Installation

Python 3.10+. The experiments use PyTorch 2.8.0, Transformers 4.57.6, BF16, and an NVIDIA H20.

```bash
git clone https://github.com/Persiapinkman/selectverify.git
cd selectverify
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference,benchmark,test]'
```

For CPU-only result inspection and tests, install `.[test]` instead.

## Data and model

Download the [Who&When Pro dataset](https://huggingface.co/datasets/Leoxx/whowhen_pro) and obtain the [official evaluator](https://github.com/whowhenpro/whowhen_pro). Set paths to the dataset, evaluator checkout, and your local Qwen3-architecture 4B checkpoint:

```bash
export BENCHMARK_SOURCE="/path/to/whowhen_pro"
export DATA_ROOT="/path/to/whowhen_data"
export MODEL="/path/to/checkpoint"
```

`BENCHMARK_SOURCE` contains `whowhen_eval/`; `DATA_ROOT` contains `data/text.jsonl` and `taxonomy.yaml`. The sample manifest and selected configurations are in `src/selectverify/assets/`.

## Evaluation

```bash
selectverify check-data \
  --benchmark-source "$BENCHMARK_SOURCE" --data-root "$DATA_ROOT"

selectverify run \
  --benchmark-source "$BENCHMARK_SOURCE" --data-root "$DATA_ROOT" \
  --model "$MODEL" --split test --local-batch 3 \
  --methods lean,best_dev,aao --output runs/paired-test.jsonl
```

`best_dev` is SelectVerify, `lean` is SV-Lite, and `aao` is All-at-Once. For a quick check, use `--split dev --per-framework 1`. Each run saves predictions, experiment metadata, and summary metrics.

The CLI also supports `binary_search`, `short`, and `reason`. Binary Search takes `--binary-source`; the generation controls take `--paired-source`. Use `selectverify run --help` for arguments.

## Results and tests

```bash
selectverify report --paper
python -m pytest -q
```

`report --paper` summarizes bundled archived scores for SV, SV-Lite, AAO, Binary Search, and the two generation controls. `collect`, `screen`, and `evaluate` provide signal collection, development-set configuration selection, and cached ablations; each command has a `--help` page.

## Files

- `src/selectverify/`: scoring, prompts, inference, benchmark adapters, and evaluation commands.
- `src/selectverify/assets/`: fixed configurations, sample IDs, per-example scores, and experiment metadata.
- `tests/`: algorithm, parser, reporting, and inference-kernel tests.
- `assets/`: framework figure in PNG and PDF.
