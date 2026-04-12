"""Pure grading functions — no I/O, no global state."""

from __future__ import annotations
import re

def _keyword_found(keyword: str, text: str) -> bool:
    kw = keyword.lower()
    text = text.lower()
    if kw and re.match(r"\w", kw[0]) and re.match(r"\w", kw[-1]):
        return bool(re.search(r"\b" + re.escape(kw) + r"\b", text))
    return kw in text

def check_comment(comment_body: str, file: str, line: int, bugs: list) -> list[int]:
    """Return indices of bugs matched by this comment and its spatial location."""
    text = comment_body.lower()
    matched: list[int] = []
    
    for i, bug in enumerate(bugs):
        target_file = bug.get("file", "")
        target_line = bug.get("line", -1)
        keywords = bug.get("keywords", [])
        
        # Spatial Check: Must be the correct file, and within +/- 3 lines of the bug
        if file == target_file and abs(line - target_line) <= 3:
            if any(_keyword_found(kw, text) for kw in keywords):
                matched.append(i)
                
    return matched

def grade(ground_truth: dict, comments: list[dict], decision: str) -> dict:
    bugs: list = ground_truth.get("bugs", [])
    should_approve: bool = ground_truth.get("should_approve", False)

    bug_breakdown = []
    bugs_found = 0
    
    for bug in bugs:
        target_file = bug.get("file", "")
        target_line = bug.get("line", -1)
        keywords = bug.get("keywords", [])
        
        found = False
        matched_by = None
        
        # Cross-reference every comment against this specific bug's coordinates
        for c in comments:
            c_file = c.get("file")
            c_line = c.get("line")
            c_body = c.get("body", "").lower()
            
            if c_file == target_file and c_line is not None and abs(c_line - target_line) <= 3:
                matched_kw = next((kw for kw in keywords if _keyword_found(kw, c_body)), None)
                if matched_kw:
                    found = True
                    matched_by = matched_kw
                    break
                    
        if found:
            bugs_found += 1
        bug_breakdown.append({"file": target_file, "line": target_line, "found": found, "matched_by": matched_by})

    total_bugs = len(bugs)
    bug_detection_rate = bugs_found / total_bugs if total_bugs > 0 else 1.0
    decision_correct = (decision == "approve") == should_approve
    decision_score = 1.0 if decision_correct else 0.0

    false_rejection = should_approve and decision == "reject"
    false_rejection_penalty = -0.2 if false_rejection else 0.0

    raw_score = bug_detection_rate * 0.7 + decision_score * 0.3 + false_rejection_penalty
    
    final_score = round(max(0.01, min(0.99, raw_score)), 4)

    return {
        "score": final_score,
        "bug_detection_rate": round(max(0.01, min(0.99, bug_detection_rate)), 4),
        "bugs_found": bugs_found,
        "total_bugs": total_bugs,
        "decision_correct": decision_correct,
        "decision_score": round(max(0.01, min(0.99, decision_score)), 4),
        "false_rejection_penalty": round(max(-0.99, min(-0.01, false_rejection_penalty)) if false_rejection else 0.01, 4),
        "bug_breakdown": bug_breakdown,
    }