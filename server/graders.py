"""
WorkLens Environment — graders.py
===================================
Reward function for the WorkLens RL environment.

Final score = weighted sum of 3 components:
    accuracy_score   (0.0–1.0)  weight: 0.70  — did agent log the RIGHT tasks?
    efficiency_score (0.0–1.0)  weight: 0.20  — did it do so with minimal steps?
    privacy_score    (0.0–1.0)  weight: 0.10  — did it avoid logging unwanted tasks?

Each component has partial credit — agent is never punished binary pass/fail.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from worklens_env.models import TaskEntry, Difficulty


# ---------------------------------------------------------------------------
# Score result container
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    """Full breakdown returned after grading an episode."""
    final_score     : float = 0.0

    accuracy_score  : float = 0.0
    efficiency_score: float = 0.0
    privacy_score   : float = 1.0   # starts at 1.0 — only goes down

    # Human-readable explanation of each component
    accuracy_reason  : str = ""
    efficiency_reason: str = ""
    privacy_reason   : str = ""

    # Per-task breakdown
    task_scores: list[dict] = field(default_factory=list)

    # Weights
    ACCURACY_WEIGHT  : float = 0.70
    EFFICIENCY_WEIGHT: float = 0.20
    PRIVACY_WEIGHT   : float = 0.10

    def compute_final(self) -> float:
        self.final_score = round(
            self.accuracy_score   * self.ACCURACY_WEIGHT  +
            self.efficiency_score * self.EFFICIENCY_WEIGHT +
            self.privacy_score    * self.PRIVACY_WEIGHT,
            4,
        )
        return self.final_score

    def summary(self) -> str:
        lines = [
            f"Final score    : {self.final_score:.3f}",
            f"  Accuracy     : {self.accuracy_score:.3f}  (×{self.ACCURACY_WEIGHT})",
            f"  Efficiency   : {self.efficiency_score:.3f}  (×{self.EFFICIENCY_WEIGHT})",
            f"  Privacy      : {self.privacy_score:.3f}  (×{self.PRIVACY_WEIGHT})",
            f"  → {self.accuracy_reason}",
            f"  → {self.efficiency_reason}",
            f"  → {self.privacy_reason}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _parse_time(t: str) -> int:
    """Convert 'HH:MM' to total minutes since midnight."""
    try:
        h, m = t.strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return -1


def _time_closeness(logged_time: str, truth_time: str, tolerance_mins: int = 15) -> float:
    """
    Returns 1.0 if times match within tolerance,
    decays linearly to 0.0 at 2× tolerance.
    """
    lt = _parse_time(logged_time)
    tt = _parse_time(truth_time)
    if lt == -1 or tt == -1:
        return 0.3          # partial credit — time was provided but unparseable
    diff = abs(lt - tt)
    if diff <= tolerance_mins:
        return 1.0
    if diff <= tolerance_mins * 2:
        return 1.0 - (diff - tolerance_mins) / tolerance_mins
    return 0.0


def _description_quality(logged_desc: str, truth_desc: str) -> float:
    """
    Scores description quality:
    - 1.0  : covers the key technical terms from ground truth
    - 0.6  : generic but not empty
    - 0.2  : very vague (< 10 words)
    - 0.0  : empty
    """
    if not logged_desc or not logged_desc.strip():
        return 0.0

    words = logged_desc.strip().split()
    if len(words) < 10:
        return 0.2

    # Extract meaningful keywords from ground truth
    truth_keywords = set(
        w.lower() for w in re.split(r'\W+', truth_desc)
        if len(w) > 4                           # skip tiny words
    )
    logged_keywords = set(
        w.lower() for w in re.split(r'\W+', logged_desc)
    )

    if not truth_keywords:
        return 0.6                              # can't compare — give partial

    overlap = len(truth_keywords & logged_keywords) / len(truth_keywords)

    if overlap >= 0.5:
        return 1.0
    if overlap >= 0.25:
        return 0.8
    if overlap >= 0.1:
        return 0.6
    return 0.4                                  # described something, just vague


def _source_overlap(logged_ids: list[str], truth_ids: list[str]) -> float:
    """
    Fraction of ground-truth source IDs the agent correctly referenced.
    """
    if not truth_ids:
        return 1.0
    matched = len(set(logged_ids) & set(truth_ids))
    return matched / len(truth_ids)


def _title_match(logged_title: str, truth_title: str) -> float:
    """Fuzzy title match based on keyword overlap."""
    if not logged_title:
        return 0.0
    lt = set(w.lower() for w in re.split(r'\W+', logged_title) if len(w) > 3)
    tt = set(w.lower() for w in re.split(r'\W+', truth_title)  if len(w) > 3)
    if not tt:
        return 0.5
    overlap = len(lt & tt) / len(tt)
    return min(1.0, overlap * 1.2)             # slight boost for partial matches


# ---------------------------------------------------------------------------
# Per-task scorer
# ---------------------------------------------------------------------------

def _score_single_task(
    logged: TaskEntry,
    truth : TaskEntry,
) -> tuple[float, dict]:
    """
    Scores one logged task against one ground truth task.

    Returns (score: float, breakdown: dict)
    """
    title_s    = _title_match(logged.title, truth.title)
    desc_s     = _description_quality(logged.description, truth.description)
    time_s     = _time_closeness(logged.start_time, truth.start_time)
    source_s   = _source_overlap(logged.source_ids, truth.source_ids)

    # Weighted task score
    task_score = (
        title_s  * 0.20 +
        desc_s   * 0.40 +
        time_s   * 0.25 +
        source_s * 0.15
    )

    breakdown = {
        "logged_title" : logged.title,
        "truth_title"  : truth.title,
        "title_score"  : round(title_s,  3),
        "desc_score"   : round(desc_s,   3),
        "time_score"   : round(time_s,   3),
        "source_score" : round(source_s, 3),
        "task_score"   : round(task_score, 3),
    }

    return task_score, breakdown


# ---------------------------------------------------------------------------
# Accuracy scorer — matches logged entries to ground truth
# ---------------------------------------------------------------------------

def grade_accuracy(
    logged_entries     : list[TaskEntry],
    ground_truth       : list[TaskEntry],
) -> tuple[float, str, list[dict]]:
    """
    Matches each ground truth task to the best logged entry.
    Unmatched ground truth tasks score 0.
    """
    if not ground_truth:
        return 1.0, "No tasks required — full score", []

    if not logged_entries:
        return 0.0, "Agent logged nothing", []

    task_scores  = []
    used_indices = set()

    for truth in ground_truth:
        best_score = 0.0
        best_breakdown = {}
        best_idx = -1

        for i, logged in enumerate(logged_entries):
            if i in used_indices:
                continue
            score, breakdown = _score_single_task(logged, truth)
            if score > best_score:
                best_score     = score
                best_breakdown = breakdown
                best_idx       = i

        if best_idx >= 0:
            used_indices.add(best_idx)

        # Guard: if no match found insert zero-score placeholder
        if not best_breakdown:
            best_breakdown = {
                "logged_title": "",
                "truth_title" : truth.title,
                "title_score" : 0.0,
                "desc_score"  : 0.0,
                "time_score"  : 0.0,
                "source_score": 0.0,
                "task_score"  : 0.0,
            }
        task_scores.append({**best_breakdown, "matched": best_idx >= 0})

    avg = sum(t.get("task_score", 0.0) for t in task_scores) / len(task_scores)

    # Coverage penalty — if agent logged fewer tasks than expected
    coverage = min(len(logged_entries), len(ground_truth)) / len(ground_truth)
    final    = avg * (0.7 + 0.3 * coverage)   # up to 30% penalty for missing tasks

    reason = (
        f"Matched {len(used_indices)}/{len(ground_truth)} tasks. "
        f"Avg task score: {avg:.2f}. Coverage: {coverage:.0%}"
    )

    return round(final, 4), reason, task_scores


# ---------------------------------------------------------------------------
# Efficiency scorer — how many steps did agent use?
# ---------------------------------------------------------------------------

def grade_efficiency(
    steps_used    : int,
    max_steps     : int,
    clarification_rounds: int,
    difficulty    : Difficulty,
) -> tuple[float, str]:
    """
    Rewards the agent for resolving tasks with fewer steps.

    Ideal step counts per difficulty:
        Easy   : ≤ 4 steps,  0 clarification rounds
        Medium : ≤ 6 steps,  1 clarification round
        Hard   : ≤ 8 steps,  2 clarification rounds
    """
    ideal = {
        Difficulty.EASY  : (4, 0),
        Difficulty.MEDIUM: (6, 1),
        Difficulty.HARD  : (8, 2),
    }

    ideal_steps, ideal_rounds = ideal[difficulty]

    # Step efficiency: 1.0 at ideal, decays to 0.0 at max_steps
    if steps_used <= ideal_steps:
        step_score = 1.0
    else:
        excess = steps_used - ideal_steps
        budget = max_steps - ideal_steps
        step_score = max(0.0, 1.0 - (excess / budget))

    # Clarification efficiency: penalise for too many rounds
    if clarification_rounds <= ideal_rounds:
        round_score = 1.0
    elif clarification_rounds == ideal_rounds + 1:
        round_score = 0.6
    else:
        round_score = max(0.0, 0.6 - (clarification_rounds - ideal_rounds - 1) * 0.2)

    final  = step_score * 0.6 + round_score * 0.4
    reason = (
        f"Used {steps_used}/{max_steps} steps (ideal ≤{ideal_steps}). "
        f"Clarification rounds: {clarification_rounds} (ideal ≤{ideal_rounds}). "
        f"Step score: {step_score:.2f}, Round score: {round_score:.2f}"
    )

    return round(final, 4), reason


# ---------------------------------------------------------------------------
# Privacy scorer — did agent log things it shouldn't have?
# ---------------------------------------------------------------------------

def grade_privacy(
    logged_entries: list[TaskEntry],
    not_to_log    : list[str],
) -> tuple[float, str]:
    """
    Penalises agent for logging source IDs in the not_to_log list.
    Each violation costs 0.3 points, minimum score 0.0.
    """
    if not not_to_log:
        return 1.0, "No privacy constraints — full score"

    violations = 0
    violating_ids = []

    banned_set = set(not_to_log)
    for entry in logged_entries:
        for sid in entry.source_ids:
            if sid in banned_set:
                violations += 1
                violating_ids.append(sid)

    if violations == 0:
        return 1.0, "No unwanted tasks logged — full privacy score"

    penalty = min(1.0, violations * 0.3)
    score   = max(0.0, 1.0 - penalty)
    reason  = (
        f"{violations} privacy violation(s). "
        f"Logged unwanted source IDs: {violating_ids}"
    )

    return round(score, 4), reason


# ---------------------------------------------------------------------------
# Main grader — call this at episode end
# ---------------------------------------------------------------------------

def grade_episode(
    logged_entries      : list[TaskEntry],
    ground_truth        : list[TaskEntry],
    not_to_log          : list[str],
    steps_used          : int,
    max_steps           : int,
    clarification_rounds: int,
    difficulty          : Difficulty,
) -> GradeResult:
    """
    Full episode grader. Call when episode is done (LOG_ENTRY or SKIP).

    Returns a GradeResult with final_score and full breakdown.
    """
    result = GradeResult()

    # 1. Accuracy
    result.accuracy_score, result.accuracy_reason, result.task_scores = grade_accuracy(
        logged_entries, ground_truth
    )

    # 2. Efficiency
    result.efficiency_score, result.efficiency_reason = grade_efficiency(
        steps_used, max_steps, clarification_rounds, difficulty
    )

    # 3. Privacy
    result.privacy_score, result.privacy_reason = grade_privacy(
        logged_entries, not_to_log
    )

    # Final weighted score
    result.compute_final()

    return result


# ---------------------------------------------------------------------------
# Step-level reward — called after every step() for RL training signal
# ---------------------------------------------------------------------------

def step_reward(
    action_type         : str,
    match_count         : int,
    clarification_rounds: int,
    logged_count        : int,
    ground_truth_count  : int,
    is_done             : bool,
    final_grade         : GradeResult | None = None,
) -> float:
    """
    Small intermediate reward at every step so the agent gets
    a training signal throughout the episode, not just at the end.

    Returns a float in [-0.2, 1.0]
    """
    if is_done and final_grade is not None:
        # Terminal reward = full episode score
        return final_grade.final_score

    reward = 0.0

    # Reward productive actions
    if action_type == "SEARCH" and match_count > 0:
        reward += 0.05                              # found something useful

    if action_type == "ASK_QUESTION" and clarification_rounds == 1:
        reward += 0.05                              # first narrowing question

    if action_type in ("LOG_ENTRY", "AUTO_LOG"):
        progress = logged_count / max(ground_truth_count, 1)
        reward += 0.1 * progress                    # partial credit for logging

    # Penalise wasteful actions
    if action_type == "ASK_QUESTION" and clarification_rounds > 2:
        reward -= 0.05                              # too many questions

    if action_type == "SEARCH" and match_count == 0:
        reward -= 0.02                              # searched but found nothing

    return round(reward, 4)