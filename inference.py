"""PR Review Agent — Multi-turn interactive loop with file navigation."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Optional, List

import requests
from openai import OpenAI

os.environ.setdefault("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MODEL_NAME: str = os.environ["MODEL_NAME"]

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
API_KEY: str = os.environ.get("API_KEY", HF_TOKEN)
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASK_CONFIGS: dict[str, dict] = {
    "easy":   {"max_steps": 8,  "threshold": 0.7},
    "medium": {"max_steps": 15, "threshold": 0.6},
    "hard":   {"max_steps": 20, "threshold": 0.5},
}

_SYSTEM_PROMPT = """\
You are an expert software engineer reviewing a pull request. You interact with a simulated environment.

At each step, you will receive the current state of the PR, including a file tree.
You must output exactly ONE JSON object per turn, representing your next action.

Available Actions:
1. Read a file to understand context:
{"action_type": "read_file", "file": "src/main.py"}

2. Leave a comment on a specific line to point out a bug or security issue:
{"action_type": "comment", "file": "src/main.py", "line": 42, "body": "Missing null check here."}

3. Approve the PR (only if completely bug-free):
{"action_type": "approve"}

4. Request Changes (if you found bugs):
{"action_type": "request_changes"}

Rules:
- Output ONLY valid JSON. No markdown fences, no conversational text.
- If you see a bug in the diff, you can comment immediately.
- If you need to see the rest of a file, use "read_file" first.
- End your review with either "approve" or "request_changes".
"""

def clamp_score(s: float) -> float:
    """Ensure score is strictly in (0, 1) — never 0.0 or 1.0."""
    try:
        s = float(s)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(s):
        return 0.5
    return round(max(0.01, min(0.99, s)), 4)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str = "true" if done else "false"
    error_str = error if error is not None else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def _format_observation(obs: dict) -> str:
    """Convert the environment state into a string prompt for the LLM."""
    prompt = f"PR Title: {obs.get('pr_title')}\n"
    prompt += f"Description: {obs.get('pr_description')}\n"
    prompt += f"Available Files: {', '.join(obs.get('file_tree', []))}\n\n"
    prompt += f"--- DIFF ---\n{obs.get('diff')}\n\n"
    
    if obs.get('current_file_path'):
        prompt += f"--- CONTENT OF {obs.get('current_file_path')} ---\n"
        prompt += f"{obs.get('current_file_content')}\n\n"
        
    prompt += "What is your next action JSON?"
    return prompt

def get_llm_action(messages: list) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw.strip())
        return json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM parsing failed: {exc}")
        # Safe fallback to prevent episode crash
        return {"action_type": "request_changes"}

def run_task(task: str) -> None:
    log_start(task=task, env="PRReviewEnv", model=MODEL_NAME)

    cfg = TASK_CONFIGS[task]
    rewards: list[float] = []
    score = 0.01
    steps_taken = 0

    try:
        resp = requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=10)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as exc:
        log_end(False, 0, 0.01, [0.01])
        print(f"[error] reset failed for task={task}: {exc}", flush=True)
        return

    # Initialize ReAct Memory
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _format_observation(obs)}
    ]

    for step in range(1, cfg["max_steps"] + 1):
        steps_taken = step
        
        # 1. Ask LLM for next action
        action_payload = get_llm_action(messages)
        action_type = action_payload.get("action_type", "request_changes")
        
        # Append assistant's thought to history
        messages.append({"role": "assistant", "content": json.dumps(action_payload)})

        # 2. Send Action to Environment
        step_error = None
        try:
            resp = requests.post(f"{ENV_URL}/step", json=action_payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            
            obs = result["observation"]
            reward_val = clamp_score(result["reward"]["value"])
            done = result["done"]
            if done:
                score = clamp_score(result["info"].get("score", 0.01))
                
        except Exception as exc:
            reward_val, done, step_error = 0.01, True, str(exc)
            score = 0.01

        rewards.append(reward_val)
        
        # 3. Log securely to stdout
        safe_action = repr(json.dumps(action_payload))
        log_step(steps_taken, safe_action, reward_val, done, step_error)

        if done:
            break

        # 4. Give the LLM the new observation (e.g., the file it just read)
        messages.append({"role": "user", "content": _format_observation(obs)})

    # If it hits max steps without a decision, force a failure score.
    if not done:
        score = 0.01

    log_end(score >= cfg["threshold"], steps_taken, score, rewards)

if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_task(task)