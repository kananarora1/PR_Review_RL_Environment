---
title: PR Review RL Environment
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
- openenv
license: apache-2.0
short_description: RL-Improvement
---

# PR Review RL Environment 
**A Meta OpenEnv Hackathon Submission**

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-1.0.0-blue.svg)](https://github.com/meta-pytorch/OpenEnv)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)

## Overview & Motivation

Current Large Language Models (LLMs) are frequently used for code generation, but training autonomous agents to *review* code requires a completely different skill set: spatial awareness, codebase navigation, and the ability to distinguish between genuine vulnerabilities and safe abstractions. 

This OpenEnv project provides a high-fidelity simulation of a Senior Software Engineer's Pull Request workflow. Instead of spoon-feeding the agent a single static diff, this environment forces the agent to interactively traverse the repository, read dependent files, and leave precise, line-level spatial comments to secure the codebase.

### Architectural Highlights (Targeting Real-World Utility)
1. **Interactive Dependency Traversal:** Bugs rarely exist in a vacuum. A change in `api.py` might introduce a vulnerability due to a constant defined in `config.py`. Agents must actively use the `read_file` tool to explore the `file_tree` and gather context before making a decision.
2. **Spatial Action Space:** Real reviewers don't leave massive global comments. The environment forces agents to map their text generation to precise spatial coordinates (`file` and `line`), converting a simple text task into a complex alignment task.
3. **Anti-Hallucination Rigor:** The grading engine includes a strict `-0.20` penalty for **False Rejections** (rejecting a perfectly clean, bug-free PR). This provides a critical RL penalty to prevent agents from lazily guessing bugs just to farm points.
4. **Strict Spec Compliance:** Complete Pydantic typing and absolute mathematical clamping ensuring all rewards and final scores are strictly bounded within `(0.01, 0.99)`, ensuring stability across the OpenEnv evaluation pipeline.


## Tasks & Difficulty

The environment provides 25 distinct scenarios across three tiers of increasing complexity. Some scenarios are intentionally completely bug-free to train against hallucination.

| Task ID | Max Steps | Success Threshold | Description |
| :--- | :---: | :---: | :--- |
| `easy` | 8 | 0.70 | Single-file PRs. Obvious errors like off-by-one loops, missing imports, and null dereferences. |
| `medium` | 15 | 0.60 | Multi-file PRs. Requires navigating the `file_tree` to spot SQL injections, path traversals, and cross-file logic inconsistencies. |
| `hard` | 20 | 0.50 | Complex PRs. Subtle security vulnerabilities (timing attacks), TOCTOU race conditions, and late-binding closure bugs. |


## Environment Specifications

### Observation Space
The agent receives a rich state payload at every step:
```yaml
observation_space:
  pr_title: string             # The PR Title
  pr_description: string       # The PR Description
  diff: string                 # The initial git diff
  file_tree: list[string]      # Available repository files
  current_file_path: string    # The file currently being read
  current_file_content: string # The contents of the active file
  comments_so_far: list[object]# Spatial history of agent's comments
  step_count: integer          # Current step
  done: boolean                # Episode termination flag
  scenario_id: string          # The current task identifier
```

### Action Space
The agent interacts using exactly one of four typed actions per turn:
```yaml
action_space:
  action_type: "enum[read_file, comment, approve, request_changes]"
  file: "string (optional)"    # Target file for reading or commenting
  line: "integer (optional)"   # Target line number for comments
  body: "string"               # The text of the code review comment
```

### Reward Shaping
* **Neutral Exploration (`0.01`):** Granted when the agent successfully navigates to and reads a valid file. Encourages gathering context without penalizing step count.
* **Partial Success (`Variable up to 0.68`):** Granted when the agent leaves a `comment` on the correct `file` and `line` (within a ±3 line buffer) that matches a hidden bug keyword.
* **Terminal Decision (`0.31` or `0.01`):** Granted at the end of the episode for correctly choosing `approve` or `request_changes`.


## Setup & Execution

### 1. Running via Docker (Evaluation Standard)
The environment is fully containerized and deployable to Hugging Face Spaces using the Docker SDK.
```bash
# Build and start the environment server
docker-compose up --build env
```

### 2. Running Natively (Local Testing)
If you prefer to run the environment outside of Docker for rapid iteration:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the FastAPI environment server
uvicorn src.api:app --host 0.0.0.0 --port 7860
```

## Running the Baseline Agent

The baseline agent utilizes a **Multi-Turn ReAct Architecture** to interact with the environment, navigating the file system before making a decision. 

To run the agent against the server, you must provide your Hugging Face or OpenAI credentials:

```bash
# 1. Point the agent to the running environment (Update if using HF Spaces)
export ENV_URL="http://localhost:7860"

# 2. Set your LLM Inference credentials
export API_BASE_URL="[https://api-inference.huggingface.co/v1/](https://api-inference.huggingface.co/v1/)"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" 
export HF_TOKEN="hf_your_token_here"

# 3. Run the baseline evaluation
python inference.py
```

### Expected Output Format
The `inference.py` script strictly adheres to the OpenEnv standard `stdout` logging formatting requirements:

```text
[START] task=medium env=PRReviewEnv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action='{"action_type": "read_file", "file": "src/db.py"}' reward=0.01 done=false error=null
[STEP] step=2 action='{"action_type": "comment", "file": "src/db.py", "line": 3, "body": "SQL injection vulnerability..."}' reward=0.68 done=false error=null
[STEP] step=3 action='{"action_type": "request_changes"}' reward=0.31 done=true error=null
[END] success=true steps=3 score=0.990 rewards=0.01,0.68,0.31
```

## Testing

The repository includes a standalone verification script to smoke-test the grading engine, Pydantic models, and strict mathematical reward clamping without requiring a live LLM or Docker container.

```bash
chmod +x verify.sh
./verify.sh
```
*Expected Output: `PASSED: 49 / FAILED: 0`*