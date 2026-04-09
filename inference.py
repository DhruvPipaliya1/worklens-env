"""
WorkLens Environment — inference.py
=====================================
Baseline inference script.

Required env vars (injected by validator):
    API_KEY       Your API key
    API_BASE_URL  LLM endpoint
    MODEL_NAME    Model identifier
    SPACE_URL     HF Space URL
"""

import json
import os
import sys
import textwrap
import urllib.request
import urllib.error
from typing import List, Optional

# ── Read env vars directly — no .env loading ──────────────────
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
SPACE_URL    = os.environ.get("SPACE_URL",    "http://localhost:7860")
BENCHMARK    = "worklens-env"
MAX_STEPS    = 10
SUCCESS_THRESHOLD = 0.5

TASKS = [
    {"id": "easy_single_match",   "difficulty": "easy",   "seed": 42},
    {"id": "medium_multi_match",  "difficulty": "medium", "seed": 42},
    {"id": "hard_ambiguous_hint", "difficulty": "hard",   "seed": 42},
]

# ── Stdout loggers ─────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_clean = str(action).replace(" ", "_").replace("\n", "")[:60]
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── HTTP helpers ───────────────────────────────────────────────
def http_post(url: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        url,
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def http_get(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

# ── LLM call ──────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a WorkLens agent. A developer gives a short hint about work they did today.
Find relevant artifacts and log each matching task as a SEPARATE LOG_ENTRY.

RULES:
1. Always SEARCH first.
2. 1 match → LOG_ENTRY. 2-4 matches → SHOW_LIST. 5+ → ASK_QUESTION first.
3. After SHOW_LIST: call LOG_ENTRY for EACH selected item separately.
4. Never log items user did not select. SKIP only if nothing matches.
5. Each LOG_ENTRY needs: title, description (15+ words), start_time HH:MM, end_time HH:MM, source_ids.

Respond ONLY with valid JSON. No text outside JSON.

Examples:
{"action_type": "SEARCH", "hint": "updated SQL queries"}
{"action_type": "SHOW_LIST"}
{"action_type": "ASK_QUESTION", "question": "Was this frontend or backend work?"}
{"action_type": "LOG_ENTRY", "task_entry": {"title": "Fix login bug", "description": "Resolved session expiry in auth.py causing login timeouts after 30 minutes of inactivity.", "start_time": "11:20", "end_time": "11:35", "source_ids": ["abc123", "PROJ-88"], "project": "PROJ", "tags": ["bugfix"]}}
{"action_type": "SKIP", "skip_reason": "No matching artifacts found."}
""").strip()


def call_llm(api_key: str, api_base_url: str, model_name: str, messages: list) -> str:
    """Call LLM using urllib — no openai package needed."""
    payload = {
        "model"      : model_name,
        "messages"   : messages,
        "temperature": 0.2,
        "max_tokens" : 512,
    }
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{api_base_url.rstrip('/')}/chat/completions",
        data    = body,
        headers = {
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method = "POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()


def get_action(api_key: str, api_base_url: str, model_name: str,
               obs: dict, history: list) -> dict:
    """Get next action from LLM."""
    lines = [
        f"HINT: \"{obs.get('user_hint', '')}\"",
        f"STEP: {obs.get('step_count', 0)}/{obs.get('max_steps', 10)}",
        f"DIFFICULTY: {obs.get('task_difficulty', 'easy')}",
    ]
    commits = obs.get("git_commits", [])
    if commits:
        lines.append("GIT COMMITS:")
        for c in commits[:5]:
            lines.append(f"  [{c.get('commit_id','')[:8]}] {c.get('timestamp','')} — {c.get('message','')}")

    jira = obs.get("jira_items", [])
    if jira:
        lines.append("JIRA:")
        for j in jira[:3]:
            lines.append(f"  [{j.get('ticket_id','')}] {j.get('title','')} ({j.get('status','')})")

    matches = obs.get("matches_found", [])
    if matches:
        lines.append(f"MATCHES ({obs.get('match_count', 0)}):")
        for i, m in enumerate(matches[:6]):
            lines.append(f"  [{i}] {m.get('timestamp','')} [{m.get('source_type','').upper()}] {m.get('title','')}")
            lines.append(f"       {m.get('summary','')}")

    if obs.get("pending_question"):
        lines.append(f"YOU ASKED: \"{obs['pending_question']}\"")
        lines.append(f"USER SAID: \"{obs.get('user_answer', '')}\"")

    logged = obs.get("logged_entries", [])
    if logged:
        remaining = len(matches) - len(logged)
        lines.append(f"ALREADY LOGGED: {len(logged)} entries")
        if remaining > 0:
            lines.append(f"STILL NEED: {remaining} more LOG_ENTRY calls")

    if obs.get("last_action_result"):
        lines.append(f"LAST RESULT: {obs['last_action_result']}")
    if obs.get("error_message"):
        lines.append(f"ERROR: {obs['error_message']}")

    obs_text = "\n".join(lines)
    history.append({"role": "user", "content": obs_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-8:]

    try:
        raw = call_llm(api_key, api_base_url, model_name, messages)
        history.append({"role": "assistant", "content": raw})
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        return json.loads(raw)
    except Exception:
        step  = obs.get("step_count", 0)
        count = obs.get("match_count", 0)
        logged = len(obs.get("logged_entries", []))
        matches = obs.get("matches_found", [])

        if step == 0 or count == 0:
            return {"action_type": "SEARCH", "hint": obs.get("user_hint", "")}

        if count == 1 and matches:
            m = matches[0]
            return {
                "action_type": "AUTO_LOG",
                "task_entry": {
                    "title":       m.get("title", obs.get("user_hint", "Work completed")),
                    "description": f"Completed work: {m.get('summary', obs.get('user_hint', ''))}",
                    "start_time":  m.get("timestamp", "09:00"),
                    "end_time":    m.get("timestamp", "10:00"),
                    "source_ids":  [m.get("id", "")],
                    "project":     None,
                    "tags":        [],
                }
            }

        if count > 1 and step <= 2:
            return {"action_type": "SHOW_LIST"}

        if count > 1 and step == 3:
            return {"action_type": "ASK_QUESTION", "question": "Which specific task would you like me to log?"}

        # step 4+: log the best match
        if matches:
            m = matches[0]
            return {
                "action_type": "LOG_ENTRY",
                "task_entry": {
                    "title":       m.get("title", obs.get("user_hint", "Work completed")),
                    "description": f"Completed work on: {m.get('summary', obs.get('user_hint', ''))}. Related to developer hint: {obs.get('user_hint', '')}.",
                    "start_time":  m.get("timestamp", "09:00"),
                    "end_time":    m.get("timestamp", "10:00"),
                    "source_ids":  [m.get("id", "")],
                    "project":     None,
                    "tags":        [],
                }
            }

        return {"action_type": "SKIP", "skip_reason": "No matching artifacts found."}


# ── Episode runner ─────────────────────────────────────────────
def run_task(api_key: str, api_base_url: str, model_name: str,
             task: dict, base_url: str) -> None:
    task_id    = task["id"]
    difficulty = task["difficulty"]
    seed       = task["seed"]

    rewards    : List[float] = []
    steps_taken: int         = 0
    score      : float       = 0.0
    success    : bool        = False
    history    : list        = []
    session_id : str         = ""

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    try:
        # Reset
        reset_resp = http_post(f"{base_url}/reset", {
            "difficulty": difficulty,
            "seed"      : seed,
        })
        session_id = reset_resp["session_id"]
        obs        = reset_resp["observation"]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("episode_done", False):
                break

            action_dict = get_action(api_key, api_base_url, model_name, obs, history)
            action_type = action_dict.get("action_type", "SKIP")
            error_msg   = None

            try:
                step_resp = http_post(f"{base_url}/step", {
                    "session_id": session_id,
                    "action"    : action_dict,
                })
                obs       = step_resp["observation"]
                reward    = float(step_resp.get("reward", 0.0))
                done      = bool(step_resp.get("done", False))
                error_msg = obs.get("error_message")
            except Exception as e:
                reward    = 0.0
                done      = True
                error_msg = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_type, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Get final score
        try:
            state_resp = http_get(f"{base_url}/state/{session_id}")
            score = float(state_resp["state"].get("current_score", 0.0))
        except Exception:
            score = rewards[-1] if rewards else 0.0

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken + 1, action="ERROR", reward=0.0, done=True, error=str(e)[:100])

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ───────────────────────────────────────────────────────
def main():
    api_key      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")

    # SPACE_URL: try env var, then HF auto-injected SPACE_HOST
    space_host = os.environ.get("SPACE_HOST", "")
    space_url  = os.environ.get(
        "SPACE_URL",
        f"https://{space_host}" if space_host else "http://localhost:7860"
    )
    base_url = space_url.rstrip("/")

    print(f"[INFO] API_BASE_URL={api_base_url}", flush=True)
    print(f"[INFO] MODEL_NAME={model_name}", flush=True)
    print(f"[INFO] SPACE_URL={base_url}", flush=True)
    print(f"[INFO] API_KEY set={bool(api_key)}", flush=True)

    if not api_key:
        api_key = "no-key-set"
        print("[WARN] API_KEY not found in environment", flush=True)

    # Verify server reachable
    try:
        health = http_get(f"{base_url}/health")
        print(f"[INFO] Server health={health.get('status','unknown')}", flush=True)
    except Exception as e:
        print(f"[ERROR] Server not reachable at {base_url}: {e}", flush=True)
        sys.exit(1)

    for task in TASKS:
        try:
            run_task(api_key, api_base_url, model_name, task, base_url)
        except Exception as e:
            print(f"[ERROR] Task {task['id']} failed: {e}", flush=True)
            log_end(False, 0, 0.0, [])
        print(flush=True)


if __name__ == "__main__":
    main()