"""Pydantic models for the PR Review OpenEnv."""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, field_validator

class PRReviewAction(BaseModel):
    # Added "read_file" to the allowed actions
    action_type: Literal["comment", "approve", "request_changes", "read_file"]
    file: Optional[str] = None
    line: Optional[int] = None
    body: str = ""

class PRReviewObservation(BaseModel):
    diff: str
    pr_description: str
    pr_title: str
    file_tree: list[str] = []
    current_file_path: str = ""      # NEW: What file is the agent looking at?
    current_file_content: str = ""   # NEW: The contents of that file
    comments_so_far: list[dict] = []
    step_count: int = 0
    done: bool = False
    scenario_id: str = ""

class PRReviewReward(BaseModel):
    value: float
    breakdown: dict = {}

    @field_validator("value")
    @classmethod
    def reward_must_be_strictly_between(cls, v: float) -> float:
        if v <= 0.0:
            return 0.01
        if v >= 1.0:
            return 0.99
        return v