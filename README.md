---
title: PR Review OpenEnv
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - code-review
  - reinforcement-learning
---

# PR Review OpenEnv

An [OpenEnv](https://openenv.ai)-compatible environment for benchmarking AI agents on pull request code review. Agents read unified diffs, post review comments, and make approve/reject decisions. No real GitHub access — all 22 scenarios are self-contained JSON files.

## Environment Description

The environment presents agents with realistic pull request diffs (Python code). The agent must identify bugs and security issues through comments, then issue a final `approve` or `request_changes` decision. Rewards are shaped to encourage precise, specific feedback — not hallucination.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `pr_title` | string | Title of the pull request |
| `pr_description` | string | Author's description of the change |
| `diff` | string | Unified diff of the PR |
| `file_tree` | list[string] | Files touched (reserved, currently empty) |
| `comments_so_far` | list[object] | Comments posted this episode |
| `step_count` | integer | Steps taken so far |
| `done` | boolean | True when the episode has ended |

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | enum | `comment` · `approve` · `request_changes` |
| `file` | string (optional) | Filename the comment refers to |
| `line` | integer (optional) | Line number the comment refers to |
| `body` | string | Comment text (required for `comment`) |

Submitting `approve` or `request_changes` ends the episode and triggers scoring.

---

## Tasks

| ID | Scenarios | Max Steps | Success Threshold | Description |
|---|---|---|---|---|
| `easy` | 8 | 5 | 0.7 | Obvious bugs — off-by-one, null dereference, hardcoded secrets |
| `medium` | 7 | 10 | 0.6 | Logic bugs and security issues across multiple files |
| `hard` | 7 | 15 | 0.5 | Subtle race conditions, TOCTOU, cache invalidation, timing attacks |

Each task pool includes clean PRs (no bugs) to test false-positive behaviour.

---

## Scoring

```
score = bug_detection_rate × 0.7 + decision_correct × 0.3
```

- **Bug detection** — fraction of ground-truth bugs whose keywords appear in comments.
- **Decision correct** — +0.3 if approve/reject matches ground truth, else -0.3.
- **False rejection penalty** — −0.2 if agent rejects a clean (bug-free) PR.
- **Reward range** — `[-0.5, 1.0]` (perfect score = 1.0).

Partial rewards are given per step: each comment that correctly identifies a new bug earns `0.7 / total_bugs`. False-positive comments earn −0.05.

---

## Setup

### Docker (recommended)

```bash
docker build -t pr-review-env .
docker run -p 7860:7860 pr-review-env
```

API available at `http://localhost:7860`. Docs at `http://localhost:7860/docs`.

### Python (no Docker)

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 7860
```

---

## Running Inference

```bash
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/hf-inference/v1"
export ENV_URL="http://localhost:7860"

python inference.py
```

Structured log output:

```
[START] {"task": "easy", "env": "PRReviewEnv", "model": "meta-llama/Llama-3.1-8B-Instruct"}
[STEP] {"step": 1, "action": "off-by-one error in loop bound", "reward": 0.35, "done": false, "error": null}
[STEP] {"step": 2, "action": "request_changes", "reward": 0.3, "done": true, "error": null}
[END] {"success": true, "steps": 2, "score": 0.65, "rewards": [0.35, 0.3]}
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/reset?task=easy` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Inspect current episode state |

---

## Baseline Scores

> Run `python inference.py` with your model to populate.

| Model | Easy | Medium | Hard | Average |
|---|---|---|---|---|
| *(placeholder)* | — | — | — | — |
