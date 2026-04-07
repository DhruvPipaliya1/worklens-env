"""
WorkLens Environment — client.py
==================================
Python client for connecting to the WorkLens RL environment server.

Supports three connection modes:
    1. Live URL     → connect to a deployed HuggingFace Space
    2. Local server → connect to uvicorn running locally
    3. In-process   → spin up the env directly (no server needed)
"""

from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Optional, Generator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from worklens_env.models import (
    ActionType, Difficulty,
    HintAction, HintObservation, HintState, TaskEntry,
)


# ---------------------------------------------------------------------------
# StepResult — mirrors server response
# ---------------------------------------------------------------------------

class ClientStepResult:
    """Returned by env.step() — mirrors StepResult from the server."""

    def __init__(self, data: dict):
        self.observation : HintObservation = HintObservation(**data["observation"])
        self.reward      : float           = data["reward"]
        self.done        : bool            = data["done"]
        self.info        : dict            = data.get("info", {})

    def __repr__(self) -> str:
        return (
            f"StepResult(reward={self.reward:.4f}, done={self.done}, "
            f"steps={self.info.get('step_count', '?')})"
        )


# ---------------------------------------------------------------------------
# Main Client
# ---------------------------------------------------------------------------

class WorkLensEnv:
    """
    HTTP client for the WorkLens RL environment.

    Parameters
    ----------
    base_url   : server URL (no trailing slash)
    timeout    : request timeout in seconds
    max_retries: number of retries on connection failure
    """

    def __init__(
        self,
        base_url   : str  = "http://localhost:7860",
        timeout    : int  = 30,
        max_retries: int  = 3,
    ):
        self.base_url   = base_url.rstrip("/")
        self.timeout    = timeout
        self.session_id : Optional[str] = None

        # HTTP session with retry logic
        self._http = requests.Session()
        retry = Retry(
            total            = max_retries,
            backoff_factor   = 0.5,
            status_forcelist = [500, 502, 503, 504],
        )
        self._http.mount("http://",  HTTPAdapter(max_retries=retry))
        self._http.mount("https://", HTTPAdapter(max_retries=retry))

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "WorkLensEnv":
        self._wait_for_server()
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Clean up session on the server."""
        if self.session_id:
            try:
                self._http.delete(
                    f"{self.base_url}/sessions/{self.session_id}",
                    timeout=5,
                )
            except Exception:
                pass
        self._http.close()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        difficulty : str = "easy",
        seed       : int = 42,
    ) -> HintObservation:
        """
        Start a new episode.

        Parameters
        ----------
        difficulty : "easy" | "medium" | "hard"
        seed       : random seed for reproducibility

        Returns
        -------
        HintObservation — initial observation
        """
        resp = self._post("/reset", {
            "difficulty": difficulty,
            "seed"      : seed,
            "session_id": self.session_id,  # reuse slot if exists
        })
        self.session_id = resp["session_id"]
        return HintObservation(**resp["observation"])

    def step(self, action: HintAction) -> ClientStepResult:
        """
        Take one action in the environment.

        Parameters
        ----------
        action : HintAction

        Returns
        -------
        ClientStepResult(observation, reward, done, info)
        """
        if not self.session_id:
            raise RuntimeError("Call reset() before step().")

        resp = self._post("/step", {
            "session_id": self.session_id,
            "action"    : action.model_dump(),
        })
        return ClientStepResult(resp)

    def state(self) -> HintState:
        """
        Get full internal state — ground truth, scores, breakdown.
        Useful for debugging and evaluation.
        """
        if not self.session_id:
            raise RuntimeError("Call reset() before state().")

        resp = self._get(f"/state/{self.session_id}")
        return HintState(**resp["state"])

    def health(self) -> dict:
        """Check server health."""
        return self._get("/health")

    def info(self) -> dict:
        """Get environment metadata."""
        return self._get("/info")

    # ------------------------------------------------------------------
    # Convenience helpers for agents
    # ------------------------------------------------------------------

    def search(self, hint: Optional[str] = None) -> ClientStepResult:
        """Shortcut: SEARCH action."""
        return self.step(HintAction(action_type=ActionType.SEARCH, hint=hint))

    def show_list(self) -> ClientStepResult:
        """Shortcut: SHOW_LIST action."""
        return self.step(HintAction(action_type=ActionType.SHOW_LIST))

    def ask(self, question: str) -> ClientStepResult:
        """Shortcut: ASK_QUESTION action."""
        return self.step(HintAction(
            action_type=ActionType.ASK_QUESTION,
            question=question,
        ))

    def multi_select(self, indices: list[int]) -> ClientStepResult:
        """Shortcut: MULTI_SELECT action."""
        return self.step(HintAction(
            action_type=ActionType.MULTI_SELECT,
            selected_indices=indices,
        ))

    def generate_desc(self, source_ids: list[str]) -> ClientStepResult:
        """Shortcut: GENERATE_DESC action."""
        return self.step(HintAction(
            action_type=ActionType.GENERATE_DESC,
            source_ids=source_ids,
        ))

    def log_entry(self, entry: TaskEntry) -> ClientStepResult:
        """Shortcut: LOG_ENTRY action."""
        return self.step(HintAction(
            action_type=ActionType.LOG_ENTRY,
            task_entry=entry,
        ))

    def skip(self, reason: str = "") -> ClientStepResult:
        """Shortcut: SKIP action."""
        return self.step(HintAction(
            action_type=ActionType.SKIP,
            skip_reason=reason,
        ))

    # ------------------------------------------------------------------
    # In-process mode (no server needed)
    # ------------------------------------------------------------------

    @classmethod
    @contextmanager
    def in_process(cls) -> Generator["_InProcessEnv", None, None]:
        """
        Spin up the environment in-process — no HTTP server needed.
        Fastest option for local development and unit tests.

        Usage
        -----
        with WorkLensEnv.in_process() as env:
            obs = env.reset(difficulty="hard")
            result = env.step(HintAction(...))
        """
        env = _InProcessEnv()
        try:
            yield env
        finally:
            pass   # nothing to clean up

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, data: dict) -> dict:
        try:
            resp = self._http.post(
                f"{self.base_url}{path}",
                json    = data,
                timeout = self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to WorkLens server at {self.base_url}. "
                f"Is the server running? Original error: {e}"
            )
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                pass
            raise RuntimeError(f"Server error {resp.status_code}: {detail or str(e)}")

    def _get(self, path: str) -> dict:
        try:
            resp = self._http.get(
                f"{self.base_url}{path}",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to WorkLens server at {self.base_url}. "
                f"Original error: {e}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Server error {resp.status_code}: {str(e)}")

    def _wait_for_server(self, retries: int = 10, delay: float = 1.0):
        """Wait for the server to become healthy before proceeding."""
        for attempt in range(retries):
            try:
                resp = self._http.get(
                    f"{self.base_url}/health",
                    timeout=5,
                )
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(delay)
        raise ConnectionError(
            f"WorkLens server at {self.base_url} did not become healthy "
            f"after {retries} attempts."
        )


# ---------------------------------------------------------------------------
# In-process environment (wraps WorkLensEnvironment directly)
# ---------------------------------------------------------------------------

class _InProcessEnv:
    """
    Wraps WorkLensEnvironment directly — same interface as WorkLensEnv
    but no HTTP overhead. Perfect for fast local iteration.
    """

    def __init__(self):
        from worklens_env.server.worklens_environment import WorkLensEnvironment
        self._env = WorkLensEnvironment()

    def reset(
        self,
        difficulty: str = "easy",
        seed      : int = 42,
    ) -> HintObservation:
        return self._env.reset(difficulty=difficulty, seed=seed)

    def step(self, action: HintAction) -> ClientStepResult:
        result = self._env.step(action)
        return ClientStepResult({
            "observation": result.observation.model_dump(),
            "reward"     : result.reward,
            "done"       : result.done,
            "info"       : result.info,
        })

    def state(self) -> HintState:
        return self._env.state()

    # Shortcut helpers (same as WorkLensEnv)
    def search(self, hint: Optional[str] = None) -> ClientStepResult:
        return self.step(HintAction(action_type=ActionType.SEARCH, hint=hint))

    def show_list(self) -> ClientStepResult:
        return self.step(HintAction(action_type=ActionType.SHOW_LIST))

    def ask(self, question: str) -> ClientStepResult:
        return self.step(HintAction(action_type=ActionType.ASK_QUESTION, question=question))

    def log_entry(self, entry: TaskEntry) -> ClientStepResult:
        return self.step(HintAction(action_type=ActionType.LOG_ENTRY, task_entry=entry))

    def skip(self, reason: str = "") -> ClientStepResult:
        return self.step(HintAction(action_type=ActionType.SKIP, skip_reason=reason))