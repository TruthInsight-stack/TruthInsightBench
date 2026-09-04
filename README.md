# TruthInsightBench

TruthInsightBench evaluates whether research Agents can derive reproducible, appropriately bounded scientific findings from real data without seeing the source study's conclusions.

V1.0 contains 40 blind tasks across 10 scientific domains and a 29-item evaluator that produces a 0–100 task score. All task data are included and ready after checkout.

## Repository layout

| Path | Contents |
|---|---|
| `tasks/` | Research objectives, data guides, manifests, and scientific data |
| `agents/` | Workspace assembly and four runnable Agent harness adapters |
| `evaluation/` | Submission validation, evaluator assets, and scoring code |
| `provenance/` | Upstream sources, licenses, and file-level attribution |

## Install

Python 3.11+, Node.js 20+, and Docker are recommended.

```bash
git clone https://github.com/TruthInsight-stack/TruthInsightBench.git
cd TruthInsightBench
python3 -m pip install -r evaluation/evaluator/requirements.txt
docker build -f agents/Dockerfile -t truthinsightbench-agents:v1 agents
```

## Run an Agent

Configure an endpoint for the participant model:

```bash
export TIB_MODEL_BASE_URL=http://127.0.0.1:4000/v1
export TIB_MODEL_API_KEY=your-key
```

Run one task with a published harness profile:

```bash
python3 agents/run_agent.py \
  --agent codex \
  --task-id Material_01_binary_superlattice \
  --run-root /tmp/truthinsightbench-runs/codex/Material_01_binary_superlattice \
  --container-image truthinsightbench-agents:v1
```

Valid `--agent` values are `claude`, `codex`, `openscience`, and `deepseek_harness`. Use `agents/run_suite.py` for multiple tasks or `agents/run_command.py` for a custom Agent command.

## Evaluate

First validate a frozen run without calling the evaluator model:

```bash
python3 evaluation/evaluate.py \
  --run-root /tmp/truthinsightbench-runs/codex/Material_01_binary_superlattice \
  --work-dir /tmp/truthinsightbench-evaluations-dry/codex/Material_01_binary_superlattice \
  --dry-run
```

For scoring, configure an OpenAI-compatible evaluator endpoint:

```bash
export JUDGE_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
export JUDGE_MODEL=Apsara-Stack/GLM-5.1-W4A8
export JUDGE_API_KEY=your-key

python3 evaluation/evaluate.py \
  --run-root /tmp/truthinsightbench-runs/codex/Material_01_binary_superlattice \
  --work-dir /tmp/truthinsightbench-evaluations/codex/Material_01_binary_superlattice \
  --judge-endpoint "$JUDGE_ENDPOINT" \
  --judge-model "$JUDGE_MODEL"
```

Repeat `--run-root` to score a cohort. Generated scores and model receipts are written to the selected evaluation work directory.

## Paper configuration

The accompanying paper uses DeepSeek-V4-Flash with thinking disabled and the following harness versions:

| Agent ID | Harness | Version |
|---|---|---:|
| `claude` | Claude Code | 2.1.220 |
| `codex` | Codex CLI | 0.149.0 |
| `openscience` | OpenScience | 2.0.1 |
| `deepseek_harness` | DeepSeek Harness | 0.1.0rc7 |

The evaluator is GLM-5.1, served as `Apsara-Stack/GLM-5.1-W4A8`, with thinking disabled.

## License and attribution

TruthInsightBench-authored software, documentation, benchmark metadata, and evaluation assets are licensed under Apache-2.0. Third-party scientific data and bibliographic records retain their upstream terms and are excluded from that grant; some components are limited to non-commercial research use. See [`provenance/THIRD_PARTY_NOTICES.md`](provenance/THIRD_PARTY_NOTICES.md) and [`provenance/source_attribution.json`](provenance/source_attribution.json) before reusing or redistributing task data.

When reporting benchmark results, cite **“TruthInsightBench: An Evidence-Grounded Benchmark for Automated Evaluation of Open-Ended Scientific Discovery Agents.”** For single-task analysis or data reuse, also cite the corresponding upstream study and data record.
