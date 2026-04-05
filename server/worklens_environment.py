"""
WorkLens Environment — worklens_environment.py
================================================
Core RL environment implementing reset() / step() / state().

Flow per episode:
    reset(difficulty) → initial HintObservation
    step(HintAction)  → StepResult(observation, reward, done)
    state()           → HintState (full internal state for debugging)
"""

from __future__ import annotations
import uuid
from typing import Optional

from worklens_env.models import (
    ActionType, Difficulty,
    HintAction, HintObservation, HintState,
    MatchItem, SourceType, TaskEntry,
)
from worklens_env.server.data_generator import generate_scenario
from worklens_env.server.graders import (
    GradeResult, grade_episode, step_reward,
)


# ---------------------------------------------------------------------------
# StepResult — returned after every step()
# ---------------------------------------------------------------------------

class StepResult:
    def __init__(
        self,
        observation: HintObservation,
        reward     : float,
        done       : bool,
        info       : dict | None = None,
    ):
        self.observation = observation
        self.reward      = reward
        self.done        = done
        self.info        = info or {}

    def to_dict(self) -> dict:
        return {
            "observation": self.observation.model_dump(),
            "reward"     : self.reward,
            "done"       : self.done,
            "info"       : self.info,
        }


# ---------------------------------------------------------------------------
# Main Environment class
# ---------------------------------------------------------------------------

class WorkLensEnvironment:
    """
    WorkLens RL Environment.

    The agent receives a developer's natural-language hint and must:
        1. Search workday artifacts for matching evidence
        2. Clarify with the user when multiple matches exist
        3. Log only the tasks the user asked for — nothing more
    """

    # Maximum clarification rounds before penalising hard
    MAX_CLARIFICATION_ROUNDS = 3

    def __init__(self):
        self._obs           : Optional[HintObservation] = None
        self._ground_truth  : list[TaskEntry]           = []
        self._not_to_log    : list[str]                 = []
        self._user_selections: list[int]                = []
        self._narrowing_answer: Optional[str]           = None
        self._difficulty    : Difficulty                = Difficulty.EASY
        self._episode_id    : str                       = ""
        self._step_count    : int                       = 0
        self._clarification_rounds: int                 = 0
        self._final_grade   : Optional[GradeResult]    = None
        self._done          : bool                      = False

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        difficulty: str  = "easy",
        seed      : int  = 42,
    ) -> HintObservation:
        """
        Start a new episode.

        Parameters
        ----------
        difficulty : "easy" | "medium" | "hard"
        seed       : random seed for reproducibility

        Returns
        -------
        HintObservation — initial state the agent sees
        """
        diff = Difficulty(difficulty)
        scenario = generate_scenario(diff, seed=seed)

        self._obs                = scenario["observation"]
        self._ground_truth       = scenario["ground_truth"]
        self._not_to_log         = scenario["not_to_log"]
        self._user_selections    = scenario["user_selections"]
        self._narrowing_answer   = scenario["narrowing_answer"]
        self._difficulty         = diff
        self._episode_id         = uuid.uuid4().hex[:12]
        self._step_count         = 0
        self._clarification_rounds = 0
        self._final_grade        = None
        self._done               = False

        # Reset observation episode metadata
        self._obs.step_count         = 0
        self._obs.episode_done       = False
        self._obs.matches_found      = []
        self._obs.match_count        = 0
        self._obs.logged_entries     = []
        self._obs.already_logged_ids = []
        self._obs.pending_question   = None
        self._obs.user_answer        = None
        self._obs.last_action_result = (
            f"Episode started. Difficulty: {diff.value}. "
            f"Hint: \"{self._obs.user_hint}\". "
            f"Max steps: {self._obs.max_steps}."
        )
        self._obs.error_message = None

        return self._obs

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: HintAction) -> StepResult:
        """
        Process one agent action and return the result.

        Parameters
        ----------
        action : HintAction

        Returns
        -------
        StepResult(observation, reward, done)
        """
        if self._done:
            return StepResult(
                observation=self._obs,
                reward=self._final_grade.final_score if self._final_grade else 0.0,
                done=True,
                info={
                    "error"      : "Episode already done. Call reset() to start a new one.",
                    "final_score": self._final_grade.final_score if self._final_grade else 0.0,
                },
            )

        self._step_count        += 1
        self._obs.step_count     = self._step_count
        self._obs.error_message  = None

        # --- Route to handler ---
        handler = {
            ActionType.SEARCH       : self._handle_search,
            ActionType.AUTO_LOG     : self._handle_auto_log,
            ActionType.SHOW_LIST    : self._handle_show_list,
            ActionType.ASK_QUESTION : self._handle_ask_question,
            ActionType.MULTI_SELECT : self._handle_multi_select,
            ActionType.GENERATE_DESC: self._handle_generate_desc,
            ActionType.LOG_ENTRY    : self._handle_log_entry,
            ActionType.SKIP         : self._handle_skip,
        }.get(action.action_type)

        if handler is None:
            self._obs.error_message = f"Unknown action type: {action.action_type}"
            return StepResult(self._obs, reward=-0.05, done=False)

        done, reward_override = handler(action)

        # --- Check step limit ---
        if self._step_count >= self._obs.max_steps and not done:
            done = True
            self._obs.last_action_result += " (Max steps reached — episode ending.)"
            self._obs.episode_done = True

        # --- Compute reward ---
        if done and self._final_grade is None:
            self._final_grade = grade_episode(
                logged_entries       = self._obs.logged_entries,
                ground_truth         = self._ground_truth,
                not_to_log           = self._not_to_log,
                steps_used           = self._step_count,
                max_steps            = self._obs.max_steps,
                clarification_rounds = self._clarification_rounds,
                difficulty           = self._difficulty,
            )

        reward = reward_override if reward_override is not None else step_reward(
            action_type          = action.action_type.value,
            match_count          = self._obs.match_count,
            clarification_rounds = self._clarification_rounds,
            logged_count         = len(self._obs.logged_entries),
            ground_truth_count   = len(self._ground_truth),
            is_done              = done,
            final_grade          = self._final_grade,
        )

        self._done             = done
        self._obs.episode_done = done

        return StepResult(
            observation = self._obs,
            reward      = round(reward, 4),
            done        = done,
            info        = {
                "episode_id"          : self._episode_id,
                "step_count"          : self._step_count,
                "clarification_rounds": self._clarification_rounds,
                "logged_count"        : len(self._obs.logged_entries),
                "final_score"         : self._final_grade.final_score if self._final_grade else None,
            },
        )

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> HintState:
        """Return full internal state for debugging and evaluation."""
        return HintState(
            ground_truth_entries = self._ground_truth,
            observation          = self._obs,
            current_score        = self._final_grade.final_score   if self._final_grade else 0.0,
            accuracy_score       = self._final_grade.accuracy_score if self._final_grade else 0.0,
            efficiency_score     = self._final_grade.efficiency_score if self._final_grade else 0.0,
            privacy_score        = self._final_grade.privacy_score  if self._final_grade else 1.0,
            episode_id           = self._episode_id,
            difficulty           = self._difficulty,
            is_done              = self._done,
            step_count           = self._step_count,
            clarification_rounds = self._clarification_rounds,
        )

    # ------------------------------------------------------------------
    # Action handlers (private)
    # ------------------------------------------------------------------

    def _handle_search(self, action: HintAction) -> tuple[bool, Optional[float]]:
        """
        SEARCH — scan all workday sources for items matching the hint.
        Returns matches ordered by relevance.
        """
        hint = (action.hint or self._obs.user_hint).lower()
        matches: list[MatchItem] = []

        # Search git commits
        for commit in self._obs.git_commits:
            score = self._relevance(hint, f"{commit.message} {' '.join(commit.files)}")
            if score > 0.15:
                matches.append(MatchItem(
                    id          = commit.commit_id,
                    source_type = SourceType.GIT,
                    title       = commit.message,
                    timestamp   = commit.timestamp,
                    relevance   = score,
                    summary     = f"Changed {len(commit.files)} file(s): {', '.join(commit.files[:2])}",
                ))

        # Search file changes
        for fc in self._obs.file_changes:
            score = self._relevance(hint, fc.filepath)
            if score > 0.15:
                matches.append(MatchItem(
                    id          = fc.filepath,
                    source_type = SourceType.FILE,
                    title       = f"File: {fc.filepath}",
                    timestamp   = fc.timestamp,
                    relevance   = score,
                    summary     = f"{fc.change_type} — {fc.lines_changed} lines ({fc.language or 'unknown'})",
                ))

        # Search Jira items
        for jira in self._obs.jira_items:
            score = self._relevance(hint, f"{jira.title} {jira.comment or ''}")
            if score > 0.15:
                matches.append(MatchItem(
                    id          = jira.ticket_id,
                    source_type = SourceType.JIRA,
                    title       = f"[{jira.ticket_id}] {jira.title}",
                    timestamp   = jira.timestamp,
                    relevance   = score,
                    summary     = jira.comment or jira.status,
                ))

        # Search Azure logs
        for az in self._obs.azure_logs:
            score = self._relevance(hint, az.title)
            if score > 0.15:
                matches.append(MatchItem(
                    id          = az.work_item_id,
                    source_type = SourceType.AZURE,
                    title       = az.title,
                    timestamp   = az.timestamp,
                    relevance   = score,
                    summary     = f"{az.type} — {az.comment or ''}",
                ))

        # Search meetings
        for mt in self._obs.meetings:
            score = self._relevance(hint, f"{mt.title} {mt.notes or ''}")
            if score > 0.15:
                matches.append(MatchItem(
                    id          = f"meeting-{mt.start_time}",
                    source_type = SourceType.MEETING,
                    title       = mt.title,
                    timestamp   = mt.start_time,
                    relevance   = score,
                    summary     = f"{mt.duration_minutes}min with {len(mt.attendees)} attendees",
                ))

        # Search Slack messages
        for sl in self._obs.slack_messages:
            score = self._relevance(hint, f"{sl.topic} {sl.snippet}")
            if score > 0.15:
                matches.append(MatchItem(
                    id          = f"slack-{sl.timestamp}",
                    source_type = SourceType.SLACK,
                    title       = f"#{sl.channel}: {sl.topic}",
                    timestamp   = sl.timestamp,
                    relevance   = score,
                    summary     = sl.snippet,
                ))

        # Sort by relevance descending
        matches.sort(key=lambda m: m.relevance, reverse=True)

        self._obs.matches_found  = matches
        self._obs.match_count    = len(matches)
        self._obs.last_action_result = (
            f"Search complete. Found {len(matches)} match(es) for \"{hint}\"."
            + (" Agent should AUTO_LOG." if len(matches) == 1
               else " Agent should SHOW_LIST." if len(matches) <= 4
               else " Agent should ASK_QUESTION to narrow down.")
        )

        return False, None   # not done, use default step reward

    def _handle_ask_question(self, action: HintAction) -> tuple[bool, Optional[float]]:
        """
        ASK_QUESTION — agent asks user to narrow down.
        Simulator responds with the pre-defined narrowing_answer.
        """
        self._clarification_rounds += 1

        if self._clarification_rounds > self.MAX_CLARIFICATION_ROUNDS:
            self._obs.error_message      = "Too many clarification rounds. Try SHOW_LIST instead."
            self._obs.last_action_result = "Clarification limit reached."
            return False, -0.1

        question = action.question or "Could you be more specific?"
        answer   = self._narrowing_answer or "please proceed with all matches"

        self._obs.pending_question   = question
        self._obs.user_answer        = answer
        self._obs.last_action_result = (
            f"Question: \"{question}\" → User answered: \"{answer}\". "
            f"Now filter your matches accordingly and use SHOW_LIST."
        )

        # Filter matches based on the answer keyword
        if self._obs.matches_found and answer:
            kw = answer.lower()
            filtered = [
                m for m in self._obs.matches_found
                if kw in m.title.lower() or kw in m.summary.lower()
            ]
            if filtered:
                self._obs.matches_found = filtered
                self._obs.match_count   = len(filtered)

        return False, None

    def _handle_show_list(self, action: HintAction) -> tuple[bool, Optional[float]]:
        """
        SHOW_LIST — present matches to user.
        Simulator auto-selects based on user_selections.
        """
        if not self._obs.matches_found:
            self._obs.error_message      = "No matches to show. Run SEARCH first."
            self._obs.last_action_result = "SHOW_LIST failed — no matches available."
            return False, -0.05

        # Simulate user picking from the list
        selections = self._user_selections or [0]
        valid_idx  = [i for i in selections if i < len(self._obs.matches_found)]

        selected   = [self._obs.matches_found[i] for i in valid_idx]
        self._obs.last_action_result = (
            f"Presented {len(self._obs.matches_found)} options. "
            f"User selected {len(selected)}: "
            f"{[m.title[:40] for m in selected]}"
        )

        # Narrow matches to only what user selected
        self._obs.matches_found = selected
        self._obs.match_count   = len(selected)

        return False, None

    def _handle_multi_select(self, action: HintAction) -> tuple[bool, Optional[float]]:
        """
        MULTI_SELECT — user picks multiple specific items.
        Same as SHOW_LIST but agent explicitly passes selected_indices.
        """
        indices = action.selected_indices or self._user_selections or [0]
        valid   = [i for i in indices if i < len(self._obs.matches_found)]

        if not valid:
            self._obs.error_message      = "No valid indices selected."
            self._obs.last_action_result = "MULTI_SELECT failed — invalid indices."
            return False, -0.05

        selected = [self._obs.matches_found[i] for i in valid]
        self._obs.matches_found  = selected
        self._obs.match_count    = len(selected)
        self._obs.last_action_result = (
            f"User selected {len(selected)} item(s): "
            f"{[m.title[:40] for m in selected]}"
        )

        return False, None

    def _handle_generate_desc(self, action: HintAction) -> tuple[bool, Optional[float]]:
        """
        GENERATE_DESC — build a description from selected matches.
        Returns a suggested description in last_action_result.
        """
        source_ids = action.source_ids or [m.id for m in self._obs.matches_found]
        relevant   = [
            m for m in self._obs.matches_found
            if m.id in source_ids
        ]

        if not relevant:
            self._obs.error_message      = "No matches selected to generate description from."
            self._obs.last_action_result = "GENERATE_DESC failed — run SEARCH or SHOW_LIST first."
            return False, -0.02

        # Build a suggested description from match summaries
        parts = [f"• [{m.source_type.value.upper()}] {m.title}: {m.summary}" for m in relevant]
        suggested = "Based on workday evidence:\n" + "\n".join(parts)

        self._obs.last_action_result = (
            f"Description generated from {len(relevant)} source(s). "
            f"Suggested: \"{suggested[:120]}...\". "
            f"Now call LOG_ENTRY with this description."
        )

        return False, 0.02    # small positive reward for using description generation

    def _handle_auto_log(self, action: HintAction) -> tuple[bool, Optional[float]]:
        """
        AUTO_LOG — log the single unambiguous match directly.
        Only valid when exactly 1 match exists.
        """
        if self._obs.match_count != 1:
            self._obs.error_message = (
                f"AUTO_LOG requires exactly 1 match. "
                f"Found {self._obs.match_count}. Use SHOW_LIST instead."
            )
            self._obs.last_action_result = "AUTO_LOG rejected."
            return False, -0.05

        if action.task_entry is None:
            self._obs.error_message      = "AUTO_LOG requires a task_entry in the action."
            self._obs.last_action_result = "AUTO_LOG failed — no task_entry provided."
            return False, -0.05

        return self._commit_entry(action.task_entry)

    def _handle_log_entry(self, action: HintAction) -> tuple[bool, Optional[float]]:
        """
        LOG_ENTRY — final submit of a completed task entry.
        """
        if action.task_entry is None:
            self._obs.error_message      = "LOG_ENTRY requires a task_entry in the action."
            self._obs.last_action_result = "LOG_ENTRY failed — no task_entry provided."
            return False, -0.05

        return self._commit_entry(action.task_entry)

    def _handle_skip(self, action: HintAction) -> tuple[bool, Optional[float]]:
        """
        SKIP — agent found nothing relevant. Episode ends.
        """
        reason = action.skip_reason or "No relevant matches found."
        self._obs.last_action_result = f"Agent skipped: {reason}"

        # If ground truth is empty, SKIP is correct → full score
        # Otherwise it's a failure
        if not self._ground_truth:
            return True, 1.0
        return True, None   # grader will assign a low score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _commit_entry(self, entry: TaskEntry) -> tuple[bool, Optional[float]]:
        """Add a TaskEntry to logged_entries and track source IDs."""
        self._obs.logged_entries.append(entry)
        self._obs.already_logged_ids.extend(entry.source_ids)

        self._obs.last_action_result = (
            f"Logged: \"{entry.title}\" "
            f"({entry.start_time}–{entry.end_time}). "
            f"Total logged: {len(self._obs.logged_entries)}."
        )

        # Episode ends when agent has logged everything it intends to
        # (agent signals completion by logging — further LOG_ENTRY calls
        #  can continue until max_steps or SKIP)
        done = len(self._obs.logged_entries) >= len(self._ground_truth)
        return done, None

    @staticmethod
    def _relevance(hint: str, text: str) -> float:
        """
        Simple keyword relevance score between hint and a text string.
        Returns 0.0–1.0.
        """
        hint_words = set(w for w in hint.lower().split() if len(w) > 2)
        text_lower = text.lower()

        if not hint_words:
            return 0.0

        matched = sum(1 for w in hint_words if w in text_lower)
        return round(matched / len(hint_words), 3)


# Monkey-patch tighter relevance (fixes noisy easy-scenario matches)
def _tight_relevance(hint: str, text: str) -> float:
    STOP = {"the","and","for","with","from","this","that","have","been","will","are","was"}
    hint_words = set(w for w in hint.lower().split() if len(w) > 3 and w not in STOP)
    text_lower = text.lower()
    if not hint_words:
        return 0.0
    matched = sum(1 for w in hint_words if w in text_lower)
    if matched == 0:
        return 0.0
    if matched == 1 and len(hint_words) > 2:
        return round((matched / len(hint_words)) * 0.5, 3)
    return round(matched / len(hint_words), 3)

WorkLensEnvironment._relevance = staticmethod(_tight_relevance)