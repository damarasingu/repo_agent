from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


REPO_KEYWORDS = {
    "repo",
    "repository",
    "code",
    "file",
    "files",
    "folder",
    "directory",
    "function",
    "class",
    "method",
    "module",
    "dependency",
    "dependencies",
    "readme",
    "bug",
    "error",
    "traceback",
    "test",
    "implement",
    "implementation",
    "symbol",
    "search",
    "line",
    "lines",
    "refactor",
    "mcp",
    "langgraph",
    "ollama",
}

GITHUB_URL_PATTERN = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?")


def is_repo_related_query(query: str) -> bool:
    normalized = (query or "").strip().lower()
    if not normalized:
        return False

    if any(keyword in normalized for keyword in REPO_KEYWORDS):
        return True

    if re.search(r"\b[a-z0-9_\-/]+\.(py|js|ts|tsx|jsx|md|json|toml|yaml|yml|sol|rs|go|java|cpp|c|h)\b", normalized):
        return True

    if re.search(r"\b(def|class|import|export|build|dependency|dependencies|stack trace|exception|tool call)\b", normalized):
        return True

    return False


def out_of_scope_message() -> str:
    return (
        "I can only help with questions about this repository, its code, files, dependencies, "
        "or implementation details. Ask me about a file, symbol, error, or subsystem in the repo."
    )


def _normalize_repo_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def resolve_repo_path_from_query(query: str, workspace_root: str) -> Optional[str]:
    github_match = GITHUB_URL_PATTERN.search(query or "")
    if github_match:
        return github_match.group(0)

    base = Path(workspace_root)
    cloned_root = base / "cloned_repos"
    if not cloned_root.exists():
        return None

    normalized_query = _normalize_repo_token(query)
    for repo_dir in cloned_root.iterdir():
        if not repo_dir.is_dir():
            continue
        normalized_repo_name = _normalize_repo_token(repo_dir.name)
        if normalized_repo_name and normalized_repo_name in normalized_query:
            return str(repo_dir.resolve())

    return None
