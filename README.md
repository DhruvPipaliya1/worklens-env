# WorkLens Environment

**Hint-driven developer task logging RL environment**

Built for the Meta OpenEnv Hackathon.

---

## The Problem

Whenever a task is assigned to a developer, they work on it throughout the day — writing code, fixing bugs, attending meetings, reviewing pull requests, updating configurations, and much more. At the end of the day, they are expected to remember exactly when each task started, when it ended, and write a meaningful description of what they did — then manually submit all of this into their team's task tracking system.

This is genuinely painful. A developer might context-switch between five different tasks in a single day. By 6 PM, remembering that a specific SQL query fix happened at 2:30 PM and took 45 minutes — and then writing a clear description of what changed and why — requires mental effort that has nothing to do with actual engineering work.

WorkLens solves this by training an AI agent that passively observes all developer activity throughout the day — every git commit, every file change, every Jira update, every meeting attended, every Slack discussion. The agent continuously collects these signals in the background. At the end of the day, the developer simply gives a short natural language hint about what they want to log:

> *"updated the SQL queries for the users table"*

The agent then searches through everything it observed, finds the relevant evidence, and automatically generates a complete task entry with the correct time, duration, and a meaningful description — ready to submit. If multiple matching activities exist, it asks the developer a quick clarifying question rather than guessing. The developer stays in control with minimal effort.

## The Solution

WorkLens is an RL environment where an AI agent learns to solve this. The developer gives a short natural-language hint:

```
"updated SQL queries for the users table"
```

The agent searches workday artifacts — git commits, Jira tickets, file changes, meetings, Slack messages — finds the relevant evidence, clarifies with the user when needed, and logs a precise task entry automatically.

---

## Environment Overview

```
Developer hint → Agent searches artifacts → Clarifies if needed → Logs task
```

The agent must learn to:
- Search multiple sources intelligently
- Show the user a list when multiple matches exist
- Ask a narrowing question when the hint is vague
- Log only what the user asked for — never more
- Write accurate descriptions from evidence

---

## Quick Start

```bash
# Install
pip install fastapi uvicorn pydantic requests openai

# Run server
cd Desktop
uvicorn worklens_env.server.app:app --host 0.0.0.0 --port 7860

# Test
curl http://localhost:7860/health

# Run baseline agent
python worklens_env/baseline/inference.py --url http://localhost:7860 --all-difficulties
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server health check |
| GET | `/info` | Environment metadata |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take one action |
| GET | `/state/{session_id}` | Full state + scores |

### Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "seed": 42}'
```

### Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "action": {
      "action_type": "SEARCH",
      "hint": "fixed the login bug"
    }
  }'
```

---

## Action Space

| Action | When to use |
|---|---|
| `SEARCH` | Always first — find matching artifacts |
| `AUTO_LOG` | When exactly 1 match found |
| `SHOW_LIST` | When 2–4 matches found |
| `ASK_QUESTION` | When 5+ matches — narrow down first |
| `MULTI_SELECT` | User picks multiple specific items |
| `GENERATE_DESC` | Build description from evidence |
| `LOG_ENTRY` | Final submit to Jira/Azure |
| `SKIP` | Nothing found |

---

## Observation Space

```python
{
  "user_hint"      : str,           # developer's hint
  "git_commits"    : list[Commit],  # commits today
  "file_changes"   : list[File],    # files modified
  "jira_items"     : list[Jira],    # tickets touched
  "azure_logs"     : list[Azure],   # work items
  "meetings"       : list[Meeting], # calendar events
  "slack_messages" : list[Slack],   # message threads
  "matches_found"  : list[Match],   # search results
  "match_count"    : int,
  "pending_question": str,          # agent's question
  "user_answer"    : str,           # user's reply
  "logged_entries" : list[Task],    # logged so far
  "step_count"     : int,
  "max_steps"      : int,
  "episode_done"   : bool,
  "last_action_result": str
}
```

---

## Reward Function

Final score = weighted sum of 3 components:

| Component | Weight | Description |
|---|---|---|
| Accuracy | 0.70 | Right task, right time, good description |
| Efficiency | 0.20 | Minimal steps and clarification rounds |
| Privacy | 0.10 | Did not log unrequested tasks |

Partial credit at every step — agent is never punished binary pass/fail.

```
Perfect agent  → 1.000
Vague descriptions → ~0.75
Privacy violation  → ~0.90
Logs nothing   → ~0.10
```

---

## Tasks

### Easy — Single Clear Match
```
Hint: "fixed the login bug"
Artifacts: 1 matching git commit + Jira ticket
Expected flow: SEARCH → SHOW_LIST → LOG_ENTRY
Ideal steps: ≤ 4
```

### Medium — Multiple Matches, User Picks
```
Hint: "updated SQL queries"
Artifacts: 3 SQL commits across different tables
Expected flow: SEARCH → SHOW_LIST → LOG × 2
Ideal steps: ≤ 6
```

### Hard — Vague Hint, Needs Narrowing
```
Hint: "worked on the dashboard"
Artifacts: 8+ items across frontend and backend
Expected flow: SEARCH → ASK_QUESTION → SHOW_LIST → LOG × 2
Ideal steps: ≤ 8
```

---

## Baseline Agent

```bash
# Rule-based agent (no API key needed)
python worklens_env/baseline/inference.py \
  --url http://localhost:7860 \
  --all-difficulties

# LLM agent (Groq free tier)
export HF_TOKEN=gsk_your_groq_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama3-8b-8192

python worklens_env/baseline/inference.py \
  --url http://localhost:7860 \
  --all-difficulties \
  --use-llm
```

### Baseline Scores (rule-based agent)

| Difficulty | Score | Steps |
|---|---|---|
| Easy | 0.836 | 3 |
| Medium | 0.544 | 4 |
| Hard | 0.864 | 5 |
| **Average** | **0.748** | |

---

## Project Structure

```
worklens_env/
├── __init__.py
├── models.py              # Pydantic models — Action, Observation, State
├── client.py              # Python client for training code
├── inference.py           # Baseline LLM + rule agents
├── openenv.yaml           # Environment manifest
├── README.md
└── server/
    ├── __init__.py
    ├── app.py                   # FastAPI server
    ├── worklens_environment.py  # reset() / step() / state()
    ├── data_generator.py        # Synthetic workday generator
    ├── graders.py               # Reward function
    ├── requirements.txt
    └── Dockerfile

```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `HF_TOKEN` | LLM API key | — |
| `MODEL_NAME` | LLM model name | `gpt-4o-mini` |
| `PORT` | Server port | `7860` |

---

## Real-World Impact

Every software company using Jira or Azure DevOps faces this problem. Developers at companies like Microsoft, Google, and startups worldwide spend millions of hours per year on manual task logging. WorkLens trains an agent to eliminate this friction — the same way a senior developer would intuitively know what to log and when.
