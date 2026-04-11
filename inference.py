"""
WorkLens Environment — inference.py
=====================================
Baseline inference script.

Required env vars (injected by validator):
    HF_TOKEN      Your Hugging Face / API key  (validator injects this)
    API_BASE_URL  LLM endpoint
    MODEL_NAME    Model identifier
    SPACE_URL     HF Space URL
"""

import json
import os
import sys
import textwrap
import urllib.request
from typing import List, Optional

from openai import OpenAI

# ── Config — read directly from environment, no .env loading ──
# API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
# API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
# MODEL_NAME   = os.getenv("MODEL_NAME")   or "meta-llama/Llama-3.3-70B-Instruct"
# SPACE_URL    = os.getenv("SPACE_URL")    or "http://localhost:7860"

BENCHMARK         = "worklens-env"
MAX_STEPS         = 10
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
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    action_clean = str(action).replace(" ", "_").replace("\n", "")[:60]
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── HTTP helpers (for env server only) ────────────────────────
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

# ── System prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a WorkLens agent. A developer gives a short hint about work they did today.
Find relevant artifacts and log each matching task as a SEPARATE LOG_ENTRY.

RULES:
1. Always SEARCH first using the developer's hint.
2. 1 match → AUTO_LOG directly. 2-4 matches → SHOW_LIST. 5+ → ASK_QUESTION first.
3. After SHOW_LIST: call LOG_ENTRY for EACH selected item separately.
4. Never log items user did not select. SKIP only if nothing matches.
5. Each LOG_ENTRY needs: title, description (15+ words), start_time HH:MM, end_time HH:MM, source_ids.

Respond ONLY with valid JSON. No prose, no markdown fences, no text outside JSON.

Examples:
{"action_type": "SEARCH", "hint": "updated SQL queries"}
{"action_type": "SHOW_LIST"}
{"action_type": "ASK_QUESTION", "question": "Was this frontend or backend work?"}
{"action_type": "AUTO_LOG", "task_entry": {"title": "Fix login bug", "description": "Resolved session expiry in auth.py causing login timeouts after 30 minutes of inactivity.", "start_time": "11:20", "end_time": "11:35", "source_ids": ["abc123"], "project": null, "tags": ["bugfix"]}}
{"action_type": "LOG_ENTRY", "task_entry": {"title": "Fix login bug", "description": "Resolved session expiry in auth.py causing login timeouts after 30 minutes of inactivity.", "start_time": "11:20", "end_time": "11:35", "source_ids": ["abc123", "PROJ-88"], "project": "PROJ", "tags": ["bugfix"]}}
{"action_type": "SKIP", "skip_reason": "No matching artifacts found."}
""").strip()


# ── Fallback action (when LLM fails) ──────────────────────────
def fallback_action(obs: dict, step_in_episode: int) -> dict:
    count   = obs.get("match_count", 0)
    matches = obs.get("matches_found", [])

    if step_in_episode == 1 or count == 0:
        return {"action_type": "SEARCH", "hint": obs.get("user_hint", "")}

    if count == 1 and matches:
        m = matches[0]
        return {
            "action_type": "AUTO_LOG",
            "task_entry": {
                "title":       m.get("title", obs.get("user_hint", "Work completed")),
                "description": f"Completed work on: {m.get('summary', obs.get('user_hint', ''))}. Developer hint: {obs.get('user_hint', '')}.",
                "start_time":  m.get("timestamp", "09:00"),
                "end_time":    m.get("timestamp", "10:00"),
                "source_ids":  [m.get("id", "")],
                "project":     None,
                "tags":        [],
            }
        }

    if count > 1 and step_in_episode <= 2:
        return {"action_type": "SHOW_LIST"}

    if count > 1 and step_in_episode == 3:
        return {"action_type": "ASK_QUESTION", "question": "Which specific task would you like me to log?"}

    if matches:
        m = matches[0]
        return {
            "action_type": "LOG_ENTRY",
            "task_entry": {
                "title":       m.get("title", obs.get("user_hint", "Work completed")),
                "description": f"Completed work on: {m.get('summary', obs.get('user_hint', ''))}. Developer hint: {obs.get('user_hint', '')}.",
                "start_time":  m.get("timestamp", "09:00"),
                "end_time":    m.get("timestamp", "10:00"),
                "source_ids":  [m.get("id", "")],
                "project":     None,
                "tags":        [],
            }
        }

    return {"action_type": "SKIP", "skip_reason": "No matching artifacts found."}


# ── LLM action via OpenAI client ──────────────────────────────
def get_action(client: OpenAI, model_name: str,
               obs: dict, history: list, step_in_episode: int) -> dict:

    if client is None:
        return fallback_action(obs, step_in_episode)

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
            lines.append(f"       id={m.get('id','')} — {m.get('summary','')}")

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
        completion = client.chat.completions.create(
            model      = model_name,
            messages   = messages,
            temperature= 0.2,
            max_tokens = 512,
        )
        raw = (completion.choices[0].message.content or "").strip()
        history.append({"role": "assistant", "content": raw})

        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1]).strip()

        action = json.loads(raw)

        valid = {"SEARCH","AUTO_LOG","SHOW_LIST","ASK_QUESTION","MULTI_SELECT","GENERATE_DESC","LOG_ENTRY","SKIP"}
        if action.get("action_type") not in valid:
            raise ValueError(f"Invalid action_type: {action.get('action_type')}")

        return action

    except Exception as e:
        err_str = str(e)
        if "402" in err_str or "credits" in err_str.lower():
            print("[WARN] LLM credits exhausted — using fallback for this step.", flush=True)
        else:
            print(f"[WARN] LLM call failed: {e}", flush=True)
        return fallback_action(obs, step_in_episode)


# ── Episode runner ─────────────────────────────────────────────
def run_task(client: OpenAI, model_name: str, task: dict, base_url: str) -> float:
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
        reset_resp = http_post(f"{base_url}/reset", {
            "difficulty": difficulty,
            "seed"      : seed,
        })
        session_id = reset_resp["session_id"]
        obs        = reset_resp["observation"]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("episode_done", False):
                break

            action_dict = get_action(client, model_name, obs, history, step)
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

    return score


# ── Main ───────────────────────────────────────────────────────
def main():
    # Read env vars exactly as the hackathon sample requires — HF_TOKEN first
    api_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
    api_base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_name   = os.getenv("MODEL_NAME")   or "meta-llama/Llama-3.3-70B-Instruct"

    # SPACE_URL with SPACE_HOST fallback (HF auto-injects SPACE_HOST)
    space_host = os.getenv("SPACE_HOST", "")
    base_url   = (
        os.getenv("SPACE_URL")
        or (f"https://{space_host}" if space_host else "http://localhost:7860")
    ).rstrip("/")

    print(f"[INFO] API_BASE_URL={api_base_url}", flush=True)
    print(f"[INFO] MODEL_NAME={model_name}", flush=True)
    print(f"[INFO] SPACE_URL={base_url}", flush=True)
    print(f"[INFO] API_KEY set={bool(api_key)}", flush=True)

    # Build OpenAI client — exactly as the hackathon sample requires
    client = None
    try:
        if not api_key:
            print("[WARN] No API key found — will use fallback agent only.", flush=True)
            client = None
        else:
            client = OpenAI(
                base_url = api_base_url,
                api_key  = api_key,  # real key only, never a placeholder
            )
            print("[INFO] OpenAI client initialized.", flush=True)
    except Exception as e:
        print(f"[WARN] OpenAI client init failed: {e}", flush=True)

    # Verify env server is reachable
    try:
        health = http_get(f"{base_url}/health")
        print(f"[INFO] Server health={health.get('status', 'unknown')}", flush=True)
    except Exception as e:
        print(f"[ERROR] Server not reachable at {base_url}: {e}", flush=True)
        sys.exit(1)

    scores = []
    for task in TASKS:
        try:
            s = run_task(client, model_name, task, base_url)
            scores.append(s)
        except Exception as e:
            print(f"[ERROR] Task {task['id']} failed: {e}", flush=True)
            log_end(False, 0, 0.0, [])
            scores.append(0.0)
        print(flush=True)

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"# Average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        sys.exit(0)