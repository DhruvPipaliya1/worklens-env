"""
WorkLens Environment — data_generator.py
==========================================
Generates realistic synthetic workdays for all 3 task difficulties.

Each workday contains:
- Git commits with timestamps, files, messages
- File changes
- Jira tickets touched
- Azure DevOps logs
- Calendar meetings
- Slack/Teams messages

The generator also produces ground_truth_entries so the grader
knows exactly what should have been logged.
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta

from worklens_env.models import (
    GitCommit, FileChange, JiraItem, AzureLog,
    Meeting, SlackMessage, TaskEntry, HintObservation,
    Difficulty, SourceType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time(hour: int, minute: int = 0) -> str:
    return f"{hour:02d}:{minute:02d}"


def _uid() -> str:
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# EASY scenario — single clear match
# ---------------------------------------------------------------------------

def easy_scenario(seed: int = 42) -> dict:
    """
    Developer says: 'fixed the login bug'
    Only ONE commit matches — auth.py touched at 11:23.
    Agent should AUTO_LOG without asking questions.
    """
    random.seed(seed)

    # The one matching commit
    matching_commit = GitCommit(
        commit_id=f"a{_uid()}",
        timestamp=_time(11, 23),
        message="fix: resolve login timeout on session expiry",
        files=["src/auth/auth.py", "src/auth/session.py"],
        author="dev@company.com",
        lines_added=18,
        lines_removed=6,
    )

    # Noise commits — unrelated to login
    noise_commits = [
        GitCommit(
            commit_id=f"b{_uid()}",
            timestamp=_time(9, 15),
            message="chore: update pip dependencies",
            files=["requirements.txt"],
            author="dev@company.com",
            lines_added=4,
            lines_removed=4,
        ),
        GitCommit(
            commit_id=f"c{_uid()}",
            timestamp=_time(16, 40),
            message="docs: update README with setup steps",
            files=["README.md"],
            author="dev@company.com",
            lines_added=12,
            lines_removed=2,
        ),
    ]

    file_changes = [
        FileChange(filepath="src/auth/auth.py",     timestamp=_time(11, 20), change_type="modified", lines_changed=24, language="python"),
        FileChange(filepath="src/auth/session.py",  timestamp=_time(11, 22), change_type="modified", lines_changed=10, language="python"),
        FileChange(filepath="requirements.txt",      timestamp=_time(9, 14),  change_type="modified", lines_changed=4),
        FileChange(filepath="README.md",             timestamp=_time(16, 38), change_type="modified", lines_changed=14, language="markdown"),
    ]

    jira_items = [
        JiraItem(ticket_id="PROJ-88",  title="Fix login timeout bug",          status="Done",        timestamp=_time(11, 30), comment="Resolved session expiry issue"),
        JiraItem(ticket_id="PROJ-91",  title="Update documentation for v2.1",  status="In Progress", timestamp=_time(16, 45)),
    ]

    meetings = [
        Meeting(title="Daily standup", start_time=_time(9, 30), end_time=_time(9, 45),
                duration_minutes=15, attendees=["dev@company.com", "lead@company.com"]),
    ]

    slack_messages = [
        SlackMessage(channel="#backend", timestamp=_time(11, 0),  topic="login bug",    snippet="Can someone look at the session timeout issue?"),
        SlackMessage(channel="#general", timestamp=_time(14, 0),  topic="team lunch",   snippet="Lunch at 1pm today"),
    ]

    ground_truth = [
        TaskEntry(
            title="Fix login timeout bug",
            description="Resolved session expiry issue causing login timeouts. "
                        "Updated auth.py and session.py to properly handle token refresh.",
            start_time=_time(11, 20),
            end_time=_time(11, 35),
            source_ids=[matching_commit.commit_id, "PROJ-88"],
            project="PROJ",
            tags=["bugfix", "auth"],
        )
    ]

    obs = HintObservation(
        user_hint="fixed the login bug",
        git_commits=[matching_commit] + noise_commits,
        file_changes=file_changes,
        jira_items=jira_items,
        meetings=meetings,
        slack_messages=slack_messages,
        task_difficulty=Difficulty.EASY,
        max_steps=8,
    )

    return {"observation": obs, "ground_truth": ground_truth}


# ---------------------------------------------------------------------------
# MEDIUM scenario — multiple matches, user picks
# ---------------------------------------------------------------------------

def medium_scenario(seed: int = 42) -> dict:
    """
    Developer says: 'updated SQL queries'
    THREE commits match — different tables, different times.
    Agent must SHOW_LIST → user picks which ones to log.
    """
    random.seed(seed)

    sql_commits = [
        GitCommit(
            commit_id=f"s1{_uid()}",
            timestamp=_time(10, 15),
            message="feat: migrate users table — add phone_number column",
            files=["db/migrations/0042_users_add_phone.sql", "models/user.py"],
            author="dev@company.com",
            lines_added=34,
            lines_removed=2,
        ),
        GitCommit(
            commit_id=f"s2{_uid()}",
            timestamp=_time(14, 30),
            message="perf: optimise reports query — add composite index",
            files=["db/queries/reports.sql", "db/indexes.sql"],
            author="dev@company.com",
            lines_added=12,
            lines_removed=8,
        ),
        GitCommit(
            commit_id=f"s3{_uid()}",
            timestamp=_time(16, 45),
            message="fix: update stored procedure for order_summary",
            files=["db/procedures/order_summary.sql"],
            author="dev@company.com",
            lines_added=20,
            lines_removed=15,
        ),
    ]

    noise_commits = [
        GitCommit(
            commit_id=f"n1{_uid()}",
            timestamp=_time(9, 0),
            message="chore: morning sync — rebase from main",
            files=[],
            author="dev@company.com",
        ),
        GitCommit(
            commit_id=f"n2{_uid()}",
            timestamp=_time(12, 0),
            message="style: fix linting errors in auth module",
            files=["src/auth/auth.py"],
            author="dev@company.com",
            lines_added=3,
            lines_removed=3,
        ),
    ]

    file_changes = [
        FileChange(filepath="db/migrations/0042_users_add_phone.sql", timestamp=_time(10, 10), change_type="created",  lines_changed=34, language="sql"),
        FileChange(filepath="models/user.py",                          timestamp=_time(10, 12), change_type="modified", lines_changed=8,  language="python"),
        FileChange(filepath="db/queries/reports.sql",                  timestamp=_time(14, 25), change_type="modified", lines_changed=12, language="sql"),
        FileChange(filepath="db/indexes.sql",                          timestamp=_time(14, 28), change_type="modified", lines_changed=6,  language="sql"),
        FileChange(filepath="db/procedures/order_summary.sql",         timestamp=_time(16, 40), change_type="modified", lines_changed=20, language="sql"),
        FileChange(filepath="src/auth/auth.py",                        timestamp=_time(12, 0),  change_type="modified", lines_changed=3,  language="python"),
    ]

    jira_items = [
        JiraItem(ticket_id="DB-14",   title="Users table schema update",           status="Done",        timestamp=_time(10, 20), comment="Added phone_number column with migration"),
        JiraItem(ticket_id="DB-19",   title="Reports dashboard slow query",         status="Done",        timestamp=_time(14, 45), comment="Added composite index on date+user_id"),
        JiraItem(ticket_id="DB-22",   title="Order summary procedure incorrect",    status="In Progress", timestamp=_time(17, 0)),
    ]

    meetings = [
        Meeting(title="DB schema review",    start_time=_time(9, 30),  end_time=_time(10, 0),  duration_minutes=30, attendees=["dev@company.com", "dba@company.com"]),
        Meeting(title="Sprint planning",     start_time=_time(13, 0),  end_time=_time(14, 0),  duration_minutes=60, attendees=["dev@company.com", "pm@company.com", "lead@company.com"]),
    ]

    slack_messages = [
        SlackMessage(channel="#database",  timestamp=_time(10, 5),  topic="users migration",     snippet="Starting the phone_number migration now"),
        SlackMessage(channel="#database",  timestamp=_time(14, 20), topic="reports performance", snippet="Found the slow query — composite index should fix it"),
        SlackMessage(channel="#general",   timestamp=_time(12, 30), topic="team lunch",          snippet="Lunch orders?"),
    ]

    # Ground truth — user will pick s1 and s2 (first two SQL commits)
    # s3 (stored procedure) user chose NOT to log in this scenario
    ground_truth = [
        TaskEntry(
            title="Users table migration — add phone_number",
            description="Added phone_number column to users table via migration 0042. "
                        "Updated User model accordingly.",
            start_time=_time(10, 10),
            end_time=_time(10, 25),
            source_ids=[sql_commits[0].commit_id, "DB-14"],
            project="DB",
            tags=["migration", "sql", "users"],
        ),
        TaskEntry(
            title="Optimise reports query — composite index",
            description="Identified slow reports dashboard query. Added composite index "
                        "on (date, user_id) in db/indexes.sql. Query time reduced significantly.",
            start_time=_time(14, 25),
            end_time=_time(14, 50),
            source_ids=[sql_commits[1].commit_id, "DB-19"],
            project="DB",
            tags=["performance", "sql", "reports"],
        ),
    ]

    # The 3rd SQL commit is NOT in ground truth — user didn't ask for it
    not_to_log = [sql_commits[2].commit_id]

    obs = HintObservation(
        user_hint="updated SQL queries",
        git_commits=sql_commits + noise_commits,
        file_changes=file_changes,
        jira_items=jira_items,
        meetings=meetings,
        slack_messages=slack_messages,
        task_difficulty=Difficulty.MEDIUM,
        max_steps=10,
    )

    return {
        "observation": obs,
        "ground_truth": ground_truth,
        "not_to_log": not_to_log,      # grader penalises if agent logs these
        # Simulated user responses for SHOW_LIST / MULTI_SELECT
        "user_selections": [0, 1],     # picks index 0 and 1 from the list
    }


# ---------------------------------------------------------------------------
# HARD scenario — vague hint, multi-source, needs narrowing question
# ---------------------------------------------------------------------------

def hard_scenario(seed: int = 42) -> dict:
    """
    Developer says: 'worked on the dashboard'
    8+ items across git/jira/slack/meetings all mention 'dashboard'.
    Agent must ASK_QUESTION to narrow down, then SHOW_LIST, then log.
    """
    random.seed(seed)

    # 4 dashboard commits — mix of frontend and backend
    dashboard_commits = [
        GitCommit(
            commit_id=f"d1{_uid()}",
            timestamp=_time(9, 45),
            message="feat(api): add /dashboard/metrics endpoint",
            files=["api/routes/dashboard.py", "api/serializers/metrics.py"],
            author="dev@company.com",
            lines_added=65,
            lines_removed=0,
        ),
        GitCommit(
            commit_id=f"d2{_uid()}",
            timestamp=_time(11, 30),
            message="feat(api): dashboard data aggregation service",
            files=["services/dashboard_aggregator.py", "tests/test_aggregator.py"],
            author="dev@company.com",
            lines_added=110,
            lines_removed=5,
        ),
        GitCommit(
            commit_id=f"d3{_uid()}",
            timestamp=_time(14, 0),
            message="feat(ui): dashboard chart components",
            files=["frontend/components/DashboardChart.jsx", "frontend/styles/dashboard.css"],
            author="dev@company.com",
            lines_added=80,
            lines_removed=10,
        ),
        GitCommit(
            commit_id=f"d4{_uid()}",
            timestamp=_time(15, 45),
            message="feat(ui): dashboard filter sidebar",
            files=["frontend/components/DashboardFilter.jsx"],
            author="dev@company.com",
            lines_added=55,
            lines_removed=0,
        ),
    ]

    noise_commits = [
        GitCommit(
            commit_id=f"n1{_uid()}",
            timestamp=_time(8, 50),
            message="chore: morning rebase from main",
            files=[],
            author="dev@company.com",
        ),
        GitCommit(
            commit_id=f"n2{_uid()}",
            timestamp=_time(17, 0),
            message="chore: update .gitignore",
            files=[".gitignore"],
            author="dev@company.com",
            lines_added=2,
            lines_removed=0,
        ),
    ]

    file_changes = [
        FileChange(filepath="api/routes/dashboard.py",              timestamp=_time(9, 40),  change_type="created",  lines_changed=65, language="python"),
        FileChange(filepath="api/serializers/metrics.py",           timestamp=_time(9, 43),  change_type="created",  lines_changed=28, language="python"),
        FileChange(filepath="services/dashboard_aggregator.py",     timestamp=_time(11, 25), change_type="created",  lines_changed=110, language="python"),
        FileChange(filepath="tests/test_aggregator.py",             timestamp=_time(11, 28), change_type="created",  lines_changed=45, language="python"),
        FileChange(filepath="frontend/components/DashboardChart.jsx",  timestamp=_time(13, 55), change_type="created", lines_changed=80, language="javascript"),
        FileChange(filepath="frontend/components/DashboardFilter.jsx", timestamp=_time(15, 40), change_type="created", lines_changed=55, language="javascript"),
        FileChange(filepath="frontend/styles/dashboard.css",           timestamp=_time(13, 58), change_type="created", lines_changed=40, language="css"),
    ]

    jira_items = [
        JiraItem(ticket_id="DASH-01", title="Dashboard metrics API endpoint",      status="Done",        timestamp=_time(9, 50),  comment="Endpoint live at /dashboard/metrics"),
        JiraItem(ticket_id="DASH-02", title="Dashboard data aggregation service",  status="Done",        timestamp=_time(11, 45), comment="Aggregator handles 30-day rolling window"),
        JiraItem(ticket_id="DASH-03", title="Dashboard chart UI components",       status="In Progress", timestamp=_time(14, 15)),
        JiraItem(ticket_id="DASH-04", title="Dashboard filter sidebar",            status="In Progress", timestamp=_time(16, 0)),
    ]

    meetings = [
        Meeting(
            title="Dashboard design review",
            start_time=_time(10, 0), end_time=_time(11, 0),
            duration_minutes=60,
            attendees=["dev@company.com", "designer@company.com", "pm@company.com"],
            notes="Agreed on chart types — bar for daily, line for trends",
        ),
        Meeting(
            title="Backend sync",
            start_time=_time(13, 0), end_time=_time(13, 30),
            duration_minutes=30,
            attendees=["dev@company.com", "lead@company.com"],
        ),
    ]

    slack_messages = [
        SlackMessage(channel="#dashboard-squad", timestamp=_time(9, 0),  topic="dashboard api",      snippet="Starting the metrics endpoint today"),
        SlackMessage(channel="#dashboard-squad", timestamp=_time(11, 50),topic="aggregator ready",   snippet="Aggregator service done — PR up for review"),
        SlackMessage(channel="#frontend",        timestamp=_time(14, 30),topic="dashboard ui",       snippet="Chart component pushed — looks clean"),
        SlackMessage(channel="#frontend",        timestamp=_time(16, 10),topic="filter sidebar",     snippet="Filter sidebar WIP, need design clarification"),
        SlackMessage(channel="#general",         timestamp=_time(12, 0), topic="lunch",              snippet="Pizza today!"),
    ]

    # Ground truth — BACKEND work only (agent must ask and user answers "backend")
    # Frontend commits d3, d4 and the meeting are NOT logged
    ground_truth = [
        TaskEntry(
            title="Dashboard metrics API endpoint",
            description="Built /dashboard/metrics REST endpoint with serializer. "
                        "Returns aggregated KPIs for the last 30 days.",
            start_time=_time(9, 40),
            end_time=_time(10, 0),
            source_ids=[dashboard_commits[0].commit_id, "DASH-01"],
            project="DASH",
            tags=["api", "backend", "dashboard"],
        ),
        TaskEntry(
            title="Dashboard data aggregation service",
            description="Implemented DashboardAggregator service with 30-day rolling window. "
                        "Added unit tests covering edge cases.",
            start_time=_time(11, 25),
            end_time=_time(11, 50),
            source_ids=[dashboard_commits[1].commit_id, "DASH-02"],
            project="DASH",
            tags=["service", "backend", "dashboard"],
        ),
    ]

    not_to_log = [
        dashboard_commits[2].commit_id,   # frontend chart
        dashboard_commits[3].commit_id,   # frontend filter
        "DASH-03", "DASH-04",             # frontend jira tickets
    ]

    obs = HintObservation(
        user_hint="worked on the dashboard",
        git_commits=dashboard_commits + noise_commits,
        file_changes=file_changes,
        jira_items=jira_items,
        meetings=meetings,
        slack_messages=slack_messages,
        task_difficulty=Difficulty.HARD,
        max_steps=12,
    )

    return {
        "observation": obs,
        "ground_truth": ground_truth,
        "not_to_log": not_to_log,
        # Simulated user responses
        "narrowing_answer": "backend",          # answer to ASK_QUESTION
        "user_selections": [0, 1],              # picks backend commits after filter
    }


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

SCENARIOS = {
    Difficulty.EASY:   easy_scenario,
    Difficulty.MEDIUM: medium_scenario,
    Difficulty.HARD:   hard_scenario,
}


def generate_scenario(difficulty: Difficulty, seed: int = 42) -> dict:
    """
    Main entry point. Returns a dict with:
        observation    : HintObservation  — what the agent sees
        ground_truth   : list[TaskEntry]  — what should be logged
        not_to_log     : list[str]        — source IDs agent must NOT log
        user_selections: list[int]        — simulated user picks (for SHOW_LIST)
        narrowing_answer: str | None      — simulated user answer (for ASK_QUESTION)
    """
    fn = SCENARIOS[difficulty]
    result = fn(seed=seed)
    # Ensure all optional keys exist
    result.setdefault("not_to_log", [])
    result.setdefault("user_selections", [])
    result.setdefault("narrowing_answer", None)
    return result
