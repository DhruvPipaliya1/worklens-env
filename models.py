"""
HintLog Environment — models.py
================================
All typed Pydantic models for Action, Observation, and State.
These are shared between the client (training code) and server (env logic).
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Every possible action the agent can take."""
    SEARCH       = "SEARCH"        # Scan all sources for matches to the hint
    AUTO_LOG     = "AUTO_LOG"      # Directly log a single unambiguous match
    SHOW_LIST    = "SHOW_LIST"     # Present a list of matches to the user
    ASK_QUESTION = "ASK_QUESTION"  # Ask user a narrowing question (5+ matches)
    MULTI_SELECT = "MULTI_SELECT"  # Let user pick multiple items from list
    GENERATE_DESC= "GENERATE_DESC" # Write a task description from evidence
    LOG_ENTRY    = "LOG_ENTRY"     # Final submit to Jira / Azure DevOps
    SKIP         = "SKIP"          # Nothing found — tell user and stop


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class SourceType(str, Enum):
    GIT      = "git"
    FILE     = "file"
    JIRA     = "jira"
    AZURE    = "azure"
    MEETING  = "meeting"
    SLACK    = "slack"


# ---------------------------------------------------------------------------
# Sub-models — building blocks of the Observation
# ---------------------------------------------------------------------------

class GitCommit(BaseModel):
    """One git commit from the simulated workday."""
    commit_id    : str
    timestamp    : str                      # "HH:MM" 24h format
    message      : str
    files        : list[str]               # list of file paths changed
    author       : str
    lines_added  : int = 0
    lines_removed: int = 0


class FileChange(BaseModel):
    """A file that was edited during the day."""
    filepath     : str
    timestamp    : str
    change_type  : str                     # "modified" | "created" | "deleted"
    lines_changed: int = 0
    language     : Optional[str] = None   # "python" | "sql" | "js" etc.


class JiraItem(BaseModel):
    """An existing Jira ticket that was touched today."""
    ticket_id : str                        # e.g. "PROJ-123"
    title     : str
    status    : str                        # "In Progress" | "Done" etc.
    timestamp : str
    comment   : Optional[str] = None


class AzureLog(BaseModel):
    """An Azure DevOps work item or log entry."""
    work_item_id: str
    title       : str
    type        : str                      # "Task" | "Bug" | "Feature"
    timestamp   : str
    comment     : Optional[str] = None


class Meeting(BaseModel):
    """A calendar meeting the developer attended."""
    title           : str
    start_time      : str
    end_time        : str
    duration_minutes: int
    attendees       : list[str] = []
    notes           : Optional[str] = None


class SlackMessage(BaseModel):
    """A Slack/Teams message thread."""
    channel  : str
    timestamp: str
    topic    : str
    snippet  : str                         # short summary, not full text


class TaskEntry(BaseModel):
    """A single task entry the agent wants to log."""
    title      : str
    description: str
    start_time : str
    end_time   : str
    source_ids : list[str] = []           # which commit/file/ticket IDs back this
    project    : Optional[str] = None
    tags       : list[str] = []


class MatchItem(BaseModel):
    """One item the agent found that matches the user's hint."""
    id         : str
    source_type: SourceType
    title      : str
    timestamp  : str
    relevance  : float = Field(ge=0.0, le=1.0)  # how well it matches the hint
    summary    : str


# ---------------------------------------------------------------------------
# Action — what the agent sends on each step()
# ---------------------------------------------------------------------------

class HintAction(BaseModel):
    """
    The agent's action for one step.

    Examples
    --------
    Search:
        HintAction(action_type=ActionType.SEARCH, hint="updated SQL queries")

    Ask a narrowing question:
        HintAction(action_type=ActionType.ASK_QUESTION,
                   question="Was this frontend or backend work?")

    Log a final entry:
        HintAction(action_type=ActionType.LOG_ENTRY,
                   task_entry=TaskEntry(...))
    """
    action_type     : ActionType

    # Used with SEARCH
    hint            : Optional[str] = None

    # Used with SHOW_LIST / MULTI_SELECT — indices into matches_found
    selected_indices: Optional[list[int]] = None

    # Used with ASK_QUESTION
    question        : Optional[str] = None

    # Used with GENERATE_DESC
    source_ids      : Optional[list[str]] = None

    # Used with AUTO_LOG / LOG_ENTRY
    task_entry      : Optional[TaskEntry] = None

    # Used with SKIP
    skip_reason     : Optional[str] = None


# ---------------------------------------------------------------------------
# Observation — what the agent receives after each step()
# ---------------------------------------------------------------------------

class HintObservation(BaseModel):
    """
    Everything the agent can see at any given step.

    The agent always receives the full workday context so it can
    re-search or refine at any point in the episode.
    """
    # Workday context (always present)
    user_hint      : str
    git_commits    : list[GitCommit]    = []
    file_changes   : list[FileChange]  = []
    jira_items     : list[JiraItem]    = []
    azure_logs     : list[AzureLog]    = []
    meetings       : list[Meeting]     = []
    slack_messages : list[SlackMessage]= []

    # Search results (populated after SEARCH action)
    matches_found  : list[MatchItem]   = []
    match_count    : int               = 0

    # Clarification state
    pending_question : Optional[str]  = None  # question agent asked user
    user_answer      : Optional[str]  = None  # user's reply (if any)

    # Logging state
    logged_entries      : list[TaskEntry] = []
    already_logged_ids  : list[str]       = []

    # Episode metadata
    step_count     : int       = 0
    max_steps      : int       = 10
    task_difficulty: Difficulty = Difficulty.EASY
    episode_done   : bool      = False

    # Feedback to agent
    last_action_result: Optional[str] = None  # "Search found 3 matches"
    error_message     : Optional[str] = None  # if last action was invalid


# ---------------------------------------------------------------------------
# State — full internal server state (returned by state() API)
# ---------------------------------------------------------------------------

class HintState(BaseModel):
    """
    Complete internal state of one episode.
    Returned by the state() endpoint — used for debugging and evaluation.
    """
    # Ground truth — what SHOULD have been logged
    ground_truth_entries: list[TaskEntry] = []

    # Current observation snapshot
    observation: HintObservation

    # Scoring breakdown
    current_score   : float = 0.0
    accuracy_score  : float = 0.0
    efficiency_score: float = 0.0
    privacy_score   : float = 1.0  # starts at 1.0, penalised for over-logging

    # Episode tracking
    episode_id          : str        = ""
    difficulty          : Difficulty = Difficulty.EASY
    is_done             : bool       = False
    step_count          : int        = 0
    clarification_rounds: int        = 0  # how many ASK_QUESTION rounds used