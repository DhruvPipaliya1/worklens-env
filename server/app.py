"""
WorkLens Environment — app.py
================================
FastAPI server exposing the WorkLens RL environment
over HTTP following the OpenEnv spec.

Endpoints
---------
GET  /health          → server status
POST /reset           → start new episode
POST /step            → take one action
GET  /state           → full internal state
GET  /info            → environment metadata
"""

from __future__ import annotations
import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from worklens_env.models import (
    Difficulty, HintAction, HintObservation, HintState,
)
from worklens_env.server.worklens_environment import (
    WorkLensEnvironment, StepResult,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "WorkLens Environment",
    description = (
        "Hint-driven developer task logging RL environment. "
        "The agent receives a natural-language hint from a developer "
        "and must find, clarify, and log only the relevant work entries."
    ),
    version = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Session management — one env per session_id
# ---------------------------------------------------------------------------

_sessions: dict[str, WorkLensEnvironment] = {}
_session_meta: dict[str, dict] = {}

def _get_env(session_id: str) -> WorkLensEnvironment:
    if session_id not in _sessions:
        raise HTTPException(
            status_code = 404,
            detail      = f"Session '{session_id}' not found. Call /reset first.",
        )
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty : Optional[str] = "easy"   # "easy" | "medium" | "hard"
    seed       : Optional[int] = 42
    session_id : Optional[str] = None    # pass to reuse a session slot


class ResetResponse(BaseModel):
    session_id  : str
    observation : dict
    info        : dict


class StepRequest(BaseModel):
    session_id  : str
    action      : dict       # HintAction as dict


class StepResponse(BaseModel):
    session_id  : str
    observation : dict
    reward      : float
    done        : bool
    info        : dict


class StateResponse(BaseModel):
    session_id : str
    state      : dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """OpenEnv required health check endpoint."""
    return {
        "status"      : "healthy",
        "environment" : "worklens-env",
        "version"     : "1.0.0",
        "sessions"    : len(_sessions),
        "timestamp"   : time.time(),
    }


@app.get("/info")
def info():
    """Environment metadata — action space, observation space, tasks."""
    return {
        "name"        : "worklens-env",
        "version"     : "1.0.0",
        "description" : (
            "Hint-driven developer task logging environment. "
            "Agent receives a natural-language hint and logs only "
            "the relevant work entries from simulated workday artifacts."
        ),
        "tasks": [
            {
                "id"         : "easy_single_match",
                "difficulty" : "easy",
                "description": "Single unambiguous match — agent searches and logs directly.",
                "score_range": [0.0, 1.0],
            },
            {
                "id"         : "medium_multi_match",
                "difficulty" : "medium",
                "description": "Multiple SQL changes — agent shows list, user picks which to log.",
                "score_range": [0.0, 1.0],
            },
            {
                "id"         : "hard_ambiguous_hint",
                "difficulty" : "hard",
                "description": "Vague hint with 8+ matches — agent must narrow down then log.",
                "score_range": [0.0, 1.0],
            },
        ],
        "action_space": {
            "type"   : "discrete",
            "actions": [
                "SEARCH",
                "AUTO_LOG",
                "SHOW_LIST",
                "ASK_QUESTION",
                "MULTI_SELECT",
                "GENERATE_DESC",
                "LOG_ENTRY",
                "SKIP",
            ],
        },
        "observation_space": {
            "type"  : "dict",
            "fields": [
                "user_hint",
                "git_commits",
                "file_changes",
                "jira_items",
                "azure_logs",
                "meetings",
                "slack_messages",
                "matches_found",
                "match_count",
                "pending_question",
                "user_answer",
                "logged_entries",
                "step_count",
                "max_steps",
                "task_difficulty",
                "episode_done",
                "last_action_result",
                "error_message",
            ],
        },
        "reward": {
            "range"      : [0.0, 1.0],
            "components" : {
                "accuracy" : {"weight": 0.70, "description": "Logged the right tasks with good descriptions"},
                "efficiency": {"weight": 0.20, "description": "Resolved in minimal steps/clarifications"},
                "privacy"  : {"weight": 0.10, "description": "Did not log unrequested tasks"},
            },
        },
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    """
    Start a new episode.

    Parameters
    ----------
    difficulty : "easy" | "medium" | "hard"
    seed       : random seed for reproducibility
    session_id : optional — reuse an existing session slot
    """
    # Validate difficulty
    try:
        Difficulty(req.difficulty)
    except ValueError:
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid difficulty '{req.difficulty}'. Choose: easy | medium | hard",
        )

    session_id = req.session_id or uuid.uuid4().hex[:16]

    env = WorkLensEnvironment()
    obs = env.reset(difficulty=req.difficulty, seed=req.seed)

    _sessions[session_id]      = env
    _session_meta[session_id]  = {
        "difficulty" : req.difficulty,
        "seed"       : req.seed,
        "created_at" : time.time(),
        "steps"      : 0,
    }

    return ResetResponse(
        session_id  = session_id,
        observation = obs.model_dump(),
        info        = {
            "session_id" : session_id,
            "difficulty" : req.difficulty,
            "seed"       : req.seed,
            "max_steps"  : obs.max_steps,
            "hint"       : obs.user_hint,
        },
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """
    Take one action in the environment.

    Parameters
    ----------
    session_id : from /reset response
    action     : HintAction as a dict
    """
    env = _get_env(req.session_id)

    # Parse action
    try:
        action = HintAction(**req.action)
    except Exception as e:
        raise HTTPException(
            status_code = 422,
            detail      = f"Invalid action format: {e}",
        )

    # Execute step
    result: StepResult = env.step(action)

    # Update session metadata
    if req.session_id in _session_meta:
        _session_meta[req.session_id]["steps"] += 1
        if result.done:
            _session_meta[req.session_id]["completed_at"] = time.time()
            _session_meta[req.session_id]["final_score"]  = result.info.get("final_score")

    # Clean up completed sessions after storing result
    # (keep them for state() calls — remove on next reset)

    return StepResponse(
        session_id  = req.session_id,
        observation = result.observation.model_dump(),
        reward      = result.reward,
        done        = result.done,
        info        = result.info,
    )


@app.get("/state/{session_id}", response_model=StateResponse)
def state(session_id: str):
    """
    Get full internal state of an episode.
    Includes ground truth, scores, and full breakdown.
    Used for debugging and evaluation.
    """
    env          = _get_env(session_id)
    full_state   = env.state()
    meta         = _session_meta.get(session_id, {})

    return StateResponse(
        session_id = session_id,
        state      = {
            **full_state.model_dump(),
            "session_meta": meta,
        },
    )


@app.get("/sessions")
def list_sessions():
    """List all active sessions."""
    return {
        "count"   : len(_sessions),
        "sessions": [
            {
                "session_id": sid,
                **meta,
            }
            for sid, meta in _session_meta.items()
        ],
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Clean up a session when done."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del _sessions[session_id]
    _session_meta.pop(session_id, None)
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code = 500,
        content     = {
            "error"  : type(exc).__name__,
            "detail" : str(exc),
            "path"   : str(request.url),
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)