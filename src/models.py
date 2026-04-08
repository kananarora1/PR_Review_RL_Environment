"""Pydantic models for the PR Review OpenEnv."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PRReviewAction(BaseModel):
    action_type: str          # "comment" | "approve" | "request_changes"
    file: Optional[str] = None
    line: Optional[int] = None
    body: str = ""


class PRReviewObservation(BaseModel):
    diff: str
    pr_description: str
    pr_title: str
    file_tree: list[str] = []
    comments_so_far: list[dict] = []
    step_count: int = 0
    done: bool = False
    scenario_id: str = ""


class PRReviewReward(BaseModel):
    value: float
    breakdown: dict = {}
