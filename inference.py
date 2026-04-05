"""
WorkLens Environment — inference.py
=====================================
Baseline inference script for WorkLens RL environment.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL      LLM API endpoint
    MODEL_NAME        Model identifier
    HF_TOKEN          HuggingFace / API key
    SPACE_URL         Your HF Space URL (e.g. https://yourname-worklens-env.hf.space)

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=worklens-env model=<model_name>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

# ── Load .env file ────────────────────────────────────────────
def _load_env():
    candidates = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            return

_load_env()

# ── Config ────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
SPACE_URL    = os.getenv("SPACE_URL",    "http://localhost:7860")
BENCHMARK    = "worklens-env"
MAX_STEPS    = 10
SUCCESS_THRESHOLD = 0.5

# Tasks to run
TASKS = [
    {"id": "easy_single_match",  "difficulty": "easy",   "seed": 42},
    {"id": "medium_multi_match", "difficulty": "medium",  "seed": 42},
    {"id": "hard_ambiguous_hint","difficulty": "hard",    "seed": 42},
]

# ── Stdout loggers (exact format) ─────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitize action string — no newlines or spaces
    action_clean = action.replace(" ", "_").replace("\n", "")[:60]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── HTTP helpers ──────────────────────────────────────────────
import urllib.request
import urllib.error

def _http_post(url: str, data: dict) -> dict:
    body    = json.dumps(data).encode()
    req     = urllib.request.Request(
        url,
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def _http_get(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

# ── LLM agent ─────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a WorkLens agent. A developer gives you a short hint about work they did today.
Your job: search workday artifacts and log ONLY the relevant task entries.

RULES:
1. Always SEARCH first.
2. 1 match → LOG_ENTRY directly.
3. 2-4 matches → SHOW_LIST first.
4. 5+ matches → ASK_QUESTION to narrow down.
5. Never log tasks user did not ask for.
6. LOG_ENTRY needs: title, description (>15 words), start_time HH:MM, end_time HH:MM, source_ids.

Respond ONLY with valid JSON. No explanation outside JSON.

Examples:
{"action_type": "SEARCH", "hint": "updated SQL queries"}
{"action_type": "SHOW_LIST"}
{"action_type": "ASK_QUESTION", "question": "Was this frontend or backend work?"}
{"action_type": "LOG_ENTRY", "task_entry": {"title": "Fix login bug", "description": "Resolved session expiry in auth.py causing login timeouts after 30 min.", "start_time": "11:20", "end_time": "11:35", "source_ids": ["abc123", "PROJ-88"], "project": "PROJ", "tags": ["bugfix"]}}
{"action_type": "SKIP", "skip_reason": "No matching artifacts found."}
""").strip()


def get_llm_action(client: OpenAI, obs: dict, history: list) -> dict:
    """Ask LLM what action to take given current observation."""

    # Format observation as readable text
    lines = [
        f"HINT: \"{obs['user_hint']}\"",
        f"STEP: {obs['step_count']}/{obs['max_steps']}",
        f"DIFFICULTY: {obs['task_difficulty']}",
    ]
    if obs.get("git_commits"):
        lines.append("GIT COMMITS:")
        for c in obs["git_commits"][:5]:
            lines.append(f"  [{c['commit_id'][:8]}] {c['timestamp']} — {c['message']}")
    if obs.get("jira_items"):
        lines.append("JIRA:")
        for j in obs["jira_items"][:3]:
            lines.append(f"  [{j['ticket_id']}] {j['title']} ({j['status']})")
    if obs.get("matches_found"):
        lines.append(f"MATCHES ({obs['match_count']}):")
        for i, m in enumerate(obs["matches_found"][:6]):
            lines.append(f"  [{i}] {m['timestamp']} [{m['source_type'].upper()}] {m['title']}")
            lines.append(f"       {m['summary']}")
    if obs.get("pending_question"):
        lines.append(f"YOU ASKED: \"{obs['pending_question']}\"")
        lines.append(f"USER SAID: \"{obs['user_answer']}\"")
    if obs.get("logged_entries"):
        lines.append(f"ALREADY LOGGED: {len(obs['logged_entries'])} entries")
    if obs.get("last_action_result"):
        lines.append(f"LAST RESULT: {obs['last_action_result']}")
    if obs.get("error_message"):
        lines.append(f"ERROR: {obs['error_message']}")

    obs_text = "\n".join(lines)
    history.append({"role": "user", "content": obs_text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-8:]

    try:
        resp  = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = messages,
            temperature = 0.2,
            max_tokens  = 512,
        )
        raw = (resp.choices[0].message.content or "").strip()
        history.append({"role": "assistant", "content": raw})

        # Strip markdown fences
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])

        return json.loads(raw)

    except Exception as e:
        # Fallback: if we have matches, show list; else search
        if obs.get("match_count", 0) > 0:
            return {"action_type": "SHOW_LIST"}
        return {"action_type": "SEARCH", "hint": obs["user_hint"]}


def run_task(client: OpenAI, task: dict, base_url: str) -> dict:
    """
    Run one complete task episode.
    Returns dict with score, steps, rewards, success.
    """
    task_id    = task["id"]
    difficulty = task["difficulty"]
    seed       = task["seed"]

    rewards     : List[float] = []
    steps_taken : int         = 0
    score       : float       = 0.0
    success     : bool        = False
    history     : list        = []
    session_id  : str         = ""

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset
        reset_resp = _http_post(f"{base_url}/reset", {
            "difficulty": difficulty,
            "seed"      : seed,
        })
        session_id = reset_resp["session_id"]
        obs        = reset_resp["observation"]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("episode_done", False):
                break

            # Get action from LLM
            action_dict = get_llm_action(client, obs, history)
            action_type = action_dict.get("action_type", "SKIP")

            error_msg = None
            try:
                step_resp = _http_post(f"{base_url}/step", {
                    "session_id": session_id,
                    "action"    : action_dict,
                })
                obs    = step_resp["observation"]
                reward = float(step_resp.get("reward", 0.0))
                done   = bool(step_resp.get("done", False))
                error_msg = obs.get("error_message")

            except Exception as e:
                reward    = 0.0
                done      = True
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step   = step,
                action = action_type,
                reward = reward,
                done   = done,
                error  = error_msg,
            )

            if done:
                break

        # Get final score from state
        try:
            state_resp = _http_get(f"{base_url}/state/{session_id}")
            score = float(state_resp["state"].get("current_score", 0.0))
        except Exception:
            score = rewards[-1] if rewards else 0.0

        # Clamp to [0, 1]
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        error_str = str(e)
        log_step(step=steps_taken + 1, action="ERROR", reward=0.0, done=True, error=error_str)

    finally:
        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = rewards,
        )

    return {
        "task_id" : task_id,
        "score"   : score,
        "steps"   : steps_taken,
        "rewards" : rewards,
        "success" : success,
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY not set. Check your .env file.", flush=True)
        sys.exit(1)

    client   = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    base_url = SPACE_URL.rstrip("/")

    # Verify server is alive
    try:
        health = _http_get(f"{base_url}/health")
        assert health.get("status") == "healthy"
    except Exception as e:
        print(f"[ERROR] Server not reachable at {base_url}: {e}", flush=True)
        sys.exit(1)

    all_results = []
    for task in TASKS:
        result = run_task(client, task, base_url)
        all_results.append(result)
        print(flush=True)  # blank line between tasks

    # Final summary to stderr (doesn't interfere with stdout format)
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"# Average score: {avg:.3f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()