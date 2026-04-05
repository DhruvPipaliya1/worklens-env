"""
WorkLens Environment — baseline/inference.py
=============================================
Baseline LLM agent that plays through the WorkLens environment.

The agent uses an LLM (via OpenAI-compatible API) to decide
which action to take at each step based on the current observation.

Usage
-----
# Against a live HuggingFace Space
python baseline/inference.py --url https://yourname-worklens-env.hf.space

# Against local server
python baseline/inference.py --url http://localhost:7860

# In-process (no server needed)
python baseline/inference.py --in-process

# Run all 3 difficulties and report scores
python baseline/inference.py --in-process --all-difficulties

Environment variables
---------------------
API_BASE_URL  : LLM API endpoint  (default: https://api.openai.com/v1)
API_KEY       : your API key
MODEL_NAME    : model to use      (default: gpt-4o-mini)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Optional

# Add parent dir to path so worklens_env imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Load .env file automatically ──────────────────────────
from pathlib import Path

def _load_env():
    """
    Load .env into os.environ without requiring python-dotenv.
    Searches for .env in script dir, parent, and grandparent.
    """
    candidates = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key  = key.strip()
                    val  = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            print(f"Loaded config from: {env_path}")
            return
    print("No .env file found — using system environment variables")

_load_env()
# ─────────────────────────────────────────────────────────


from worklens_env.client import WorkLensEnv, _InProcessEnv
from worklens_env.models import (
    ActionType, Difficulty, HintAction,
    HintObservation, TaskEntry,
)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def call_llm(
    client  : "OpenAI",
    model   : str,
    messages: list[dict],
) -> str:
    """Call LLM and return the text response."""
    response = client.chat.completions.create(
        model      = model,
        messages   = messages,
        temperature= 0.2,
        max_tokens = 1024,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Observation formatter — converts HintObservation → prompt text
# ---------------------------------------------------------------------------

def format_observation(obs: HintObservation) -> str:
    """Convert observation to a structured text block for the LLM prompt."""
    lines = [
        f"DEVELOPER HINT: \"{obs.user_hint}\"",
        f"STEP: {obs.step_count}/{obs.max_steps}",
        f"DIFFICULTY: {obs.task_difficulty}",
        "",
    ]

    # Git commits
    if obs.git_commits:
        lines.append("GIT COMMITS TODAY:")
        for c in obs.git_commits:
            lines.append(f"  [{c.commit_id[:8]}] {c.timestamp} — {c.message}")
            if c.files:
                lines.append(f"    Files: {', '.join(c.files[:3])}")
        lines.append("")

    # Jira items
    if obs.jira_items:
        lines.append("JIRA TICKETS TOUCHED:")
        for j in obs.jira_items:
            lines.append(f"  [{j.ticket_id}] {j.title} ({j.status})")
            if j.comment:
                lines.append(f"    Comment: {j.comment}")
        lines.append("")

    # Meetings
    if obs.meetings:
        lines.append("MEETINGS:")
        for m in obs.meetings:
            lines.append(f"  {m.start_time}–{m.end_time} | {m.title} ({m.duration_minutes}min)")
        lines.append("")

    # Search results
    if obs.matches_found:
        lines.append(f"SEARCH RESULTS ({obs.match_count} matches):")
        for i, m in enumerate(obs.matches_found):
            lines.append(f"  [{i}] [{m.source_type.value.upper()}] {m.timestamp} | {m.title}")
            lines.append(f"       {m.summary} (relevance: {m.relevance:.2f})")
        lines.append("")

    # Clarification state
    if obs.pending_question:
        lines.append(f"AGENT ASKED: \"{obs.pending_question}\"")
        lines.append(f"USER REPLIED: \"{obs.user_answer}\"")
        lines.append("")

    # Already logged
    if obs.logged_entries:
        lines.append(f"ALREADY LOGGED ({len(obs.logged_entries)} entries):")
        for e in obs.logged_entries:
            lines.append(f"  ✓ {e.title} ({e.start_time}–{e.end_time})")
        lines.append("")

    # Last feedback
    if obs.last_action_result:
        lines.append(f"LAST ACTION RESULT: {obs.last_action_result}")

    if obs.error_message:
        lines.append(f"ERROR: {obs.error_message}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a WorkLens agent. A developer gives you a short hint about work they did today.
Your job is to find the relevant work artifacts and log a precise task entry.

RULES:
1. Always SEARCH first to find matching artifacts.
2. If 1 match → you can LOG_ENTRY directly.
3. If 2-4 matches → use SHOW_LIST to let the user pick.
4. If 5+ matches → use ASK_QUESTION to narrow down first.
5. Only log tasks the developer hinted at — never log extra tasks.
6. LOG_ENTRY requires: title, description (>15 words), start_time (HH:MM), end_time, source_ids.
7. SKIP only if truly nothing matches.

RESPOND WITH VALID JSON ONLY. No explanation outside the JSON.

Action format examples:

{"action_type": "SEARCH", "hint": "updated SQL queries"}

{"action_type": "SHOW_LIST"}

{"action_type": "ASK_QUESTION", "question": "Was this frontend or backend work?"}

{"action_type": "MULTI_SELECT", "selected_indices": [0, 1]}

{"action_type": "LOG_ENTRY", "task_entry": {
    "title": "Fix login timeout bug",
    "description": "Resolved session expiry issue in auth.py causing login timeouts after 30min inactivity.",
    "start_time": "11:20",
    "end_time": "11:35",
    "source_ids": ["abc12345", "PROJ-88"],
    "project": "PROJ",
    "tags": ["bugfix", "auth"]
}}

{"action_type": "SKIP", "skip_reason": "No artifacts match the given hint."}
""".strip()


# ---------------------------------------------------------------------------
# LLM-based agent
# ---------------------------------------------------------------------------

class WorkLensLLMAgent:
    """
    LLM-based agent that plays through WorkLens episodes.
    Uses conversation history for multi-turn reasoning.
    """

    def __init__(self, client: "OpenAI", model: str):
        self.client  = client
        self.model   = model
        self.history : list[dict] = []

    def reset(self):
        """Clear conversation history for a new episode."""
        self.history = []

    def predict(self, obs: HintObservation) -> HintAction:
        """
        Given an observation, return the next HintAction.
        Uses full conversation history for multi-turn context.
        """
        obs_text = format_observation(obs)

        self.history.append({
            "role"   : "user",
            "content": obs_text,
        })

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        raw = call_llm(self.client, self.model, messages)

        self.history.append({
            "role"   : "assistant",
            "content": raw,
        })

        return self._parse_action(raw, obs)

    def _parse_action(self, raw: str, obs: HintObservation) -> HintAction:
        """Parse LLM JSON output into a HintAction. Falls back gracefully."""
        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean

        try:
            data = json.loads(clean)
            # Handle nested task_entry
            if "task_entry" in data and isinstance(data["task_entry"], dict):
                data["task_entry"] = TaskEntry(**data["task_entry"])
            return HintAction(**data)
        except Exception as e:
            # Fallback: if we have matches and no action parsed, show list
            if obs.match_count > 0:
                return HintAction(action_type=ActionType.SHOW_LIST)
            # Otherwise search
            return HintAction(
                action_type=ActionType.SEARCH,
                hint=obs.user_hint,
            )


# ---------------------------------------------------------------------------
# Rule-based fallback agent (no LLM needed — for testing)
# ---------------------------------------------------------------------------

class WorkLensRuleAgent:
    """
    Simple rule-based agent — no LLM required.
    Follows a deterministic policy:
        SEARCH → (ASK if hard) → SHOW_LIST → LOG all selected → done
    Used as a reproducible baseline when no API key is available.
    """

    def __init__(self):
        self._searched   = False
        self._asked      = False
        self._listed     = False
        self._log_idx    = 0

    def reset(self):
        self._searched = False
        self._asked    = False
        self._listed   = False
        self._log_idx  = 0

    def predict(self, obs: HintObservation) -> HintAction:
        difficulty = obs.task_difficulty

        # Step 1: Always search first
        if not self._searched:
            self._searched = True
            return HintAction(
                action_type=ActionType.SEARCH,
                hint=obs.user_hint,
            )

        # Step 2: For hard tasks, ask a narrowing question
        if difficulty == Difficulty.HARD and not self._asked:
            self._asked = True
            return HintAction(
                action_type=ActionType.ASK_QUESTION,
                question="Was this frontend or backend work?",
            )

        # Step 3: Show list to simulate user selection
        if not self._listed and obs.match_count > 0:
            self._listed = True
            return HintAction(action_type=ActionType.SHOW_LIST)

        # Step 4: Log selected entries one by one
        if obs.matches_found and self._log_idx < len(obs.matches_found):
            match = obs.matches_found[self._log_idx]
            self._log_idx += 1

            # Build a task entry from the match
            entry = TaskEntry(
                title       = match.title[:80],
                description = (
                    f"Work performed based on {match.source_type.value} activity. "
                    f"{match.summary}. Related to: {obs.user_hint}."
                ),
                start_time  = match.timestamp,
                end_time    = _add_minutes(match.timestamp, 30),
                source_ids  = [match.id],
            )
            return HintAction(
                action_type=ActionType.LOG_ENTRY,
                task_entry=entry,
            )

        # Step 5: Skip if nothing to log
        return HintAction(
            action_type=ActionType.SKIP,
            skip_reason="No more items to log.",
        )


def _add_minutes(time_str: str, minutes: int) -> str:
    """Add minutes to HH:MM string."""
    try:
        h, m = map(int, time_str.split(":"))
        total = h * 60 + m + minutes
        return f"{(total // 60) % 24:02d}:{total % 60:02d}"
    except Exception:
        return time_str


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env       ,              # WorkLensEnv or _InProcessEnv
    agent     ,              # LLMAgent or RuleAgent
    difficulty: str,
    seed      : int  = 42,
    verbose   : bool = True,
) -> dict:
    """
    Run one complete episode and return results.

    Returns
    -------
    dict with keys: difficulty, score, steps, reward_total, done
    """
    agent.reset()
    obs          = env.reset(difficulty=difficulty, seed=seed)
    reward_total = 0.0
    steps        = 0
    done         = False

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Difficulty : {difficulty.upper()}")
        print(f"  Hint       : \"{obs.user_hint}\"")
        print(f"{'='*55}")

    while not done:
        action = agent.predict(obs)

        if verbose:
            print(f"\nStep {steps + 1}: {action.action_type.value}", end="")
            if action.hint:
                print(f" → \"{action.hint}\"", end="")
            if action.question:
                print(f" → \"{action.question}\"", end="")
            if action.task_entry:
                print(f" → \"{action.task_entry.title[:50]}\"", end="")
            print()

        result        = env.step(action)
        obs           = result.observation
        reward_total += result.reward
        steps        += 1
        done          = result.done

        if verbose:
            print(f"       reward={result.reward:.4f} | {obs.last_action_result[:70]}")

        if steps >= obs.max_steps:
            break

    # Get final score from state
    try:
        st    = env.state()
        score = st.current_score
    except Exception:
        score = result.info.get("final_score", 0.0) or 0.0

    if verbose:
        print(f"\n  ✓ Episode done")
        print(f"    Steps        : {steps}")
        print(f"    Total reward : {reward_total:.4f}")
        print(f"    Final score  : {score:.4f}")

    return {
        "difficulty"  : difficulty,
        "score"       : score,
        "steps"       : steps,
        "reward_total": round(reward_total, 4),
        "done"        : done,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WorkLens baseline inference script"
    )
    parser.add_argument(
        "--url",
        type    = str,
        default = None,
        help    = "Server URL (e.g. https://yourname-worklens-env.hf.space)",
    )
    parser.add_argument(
        "--in-process",
        action  = "store_true",
        help    = "Run environment in-process (no server needed)",
    )
    parser.add_argument(
        "--difficulty",
        type    = str,
        default = "easy",
        choices = ["easy", "medium", "hard"],
        help    = "Task difficulty (default: easy)",
    )
    parser.add_argument(
        "--all-difficulties",
        action  = "store_true",
        help    = "Run all 3 difficulties and report scores",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Random seed (default: 42)",
    )
    parser.add_argument(
        "--use-llm",
        action  = "store_true",
        help    = "Use LLM agent (requires API_KEY env var)",
    )
    parser.add_argument(
        "--quiet",
        action  = "store_true",
        help    = "Suppress step-by-step output",
    )
    args = parser.parse_args()

    # --- Build agent ---
    if args.use_llm:
        if not HAS_OPENAI:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)
        api_key  = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: set API_KEY environment variable")
            sys.exit(1)
        llm_client = OpenAI(
            api_key  = api_key,
            base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
        )
        model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        agent = WorkLensLLMAgent(llm_client, model)
        print(f"Using LLM agent: {model}")
    else:
        agent = WorkLensRuleAgent()
        print("Using rule-based baseline agent (no LLM)")

    difficulties = (
        ["easy", "medium", "hard"]
        if args.all_difficulties
        else [args.difficulty]
    )

    # --- Run episodes ---
    all_results = []

    if args.in_process:
        print("Mode: in-process (no server)\n")
        with WorkLensEnv.in_process() as env:
            for diff in difficulties:
                result = run_episode(
                    env        = env,
                    agent      = agent,
                    difficulty = diff,
                    seed       = args.seed,
                    verbose    = not args.quiet,
                )
                all_results.append(result)
    else:
        url = args.url or "http://localhost:7860"
        print(f"Mode: HTTP → {url}\n")
        with WorkLensEnv(base_url=url) as env:
            for diff in difficulties:
                result = run_episode(
                    env        = env,
                    agent      = agent,
                    difficulty = diff,
                    seed       = args.seed,
                    verbose    = not args.quiet,
                )
                all_results.append(result)

    # --- Summary ---
    print(f"\n{'='*55}")
    print("  RESULTS SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Difficulty':<12} {'Score':>8} {'Steps':>7} {'Reward':>10}")
    print(f"  {'-'*42}")
    for r in all_results:
        print(
            f"  {r['difficulty']:<12} "
            f"{r['score']:>8.4f} "
            f"{r['steps']:>7} "
            f"{r['reward_total']:>10.4f}"
        )

    if len(all_results) > 1:
        avg_score = sum(r["score"] for r in all_results) / len(all_results)
        print(f"  {'-'*42}")
        print(f"  {'AVERAGE':<12} {avg_score:>8.4f}")

    print(f"{'='*55}\n")

    return all_results


if __name__ == "__main__":
    main()