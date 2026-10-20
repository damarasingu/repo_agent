from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv


load_dotenv()

# Langfuse's current Python SDK documents LANGFUSE_BASE_URL. Keep older
# project configuration using LANGFUSE_HOST compatible with the SDK.
if os.environ.get("LANGFUSE_HOST") and not os.environ.get("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]


_LOGGER_CACHE: Dict[str, "ActionLogger"] = {}
_CACHE_LOCK = threading.Lock()
_LAST_LANGFUSE_HANDLER: Any = None


class ActionLogger:
    def __init__(self, role: str):
        self.role = role
        self.session_id = os.environ.get("REPO_AGENT_SESSION_ID")
        if not self.session_id:
            self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            os.environ["REPO_AGENT_SESSION_ID"] = self.session_id

        base_dir = os.environ.get("REPO_AGENT_OBSERVABILITY_DIR")
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path.cwd() / "logs"

        self.log_dir = self.base_dir / self.session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.log_dir / f"{self.role}_{os.getpid()}.jsonl"
        self._lock = threading.Lock()

    def log_event(self, event: str, **payload: Any) -> Path:
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "role": self.role,
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "event": event,
        }
        record.update(payload)

        with self._lock:
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")

        return self.file_path


def get_observability_logger(role: str) -> ActionLogger:
    with _CACHE_LOCK:
        logger = _LOGGER_CACHE.get(role)
        if logger is None:
            logger = ActionLogger(role)
            _LOGGER_CACHE[role] = logger
        return logger


def log_action(role: str, event: str, **payload: Any) -> Path:
    return get_observability_logger(role).log_event(event, **payload)


def get_langfuse_run_config(repo_path: str, query: str) -> Dict[str, Any]:
    """Return LangGraph config with a Langfuse callback when credentials are set."""
    session_id = os.environ.get("REPO_AGENT_SESSION_ID")
    config: Dict[str, Any] = {
        "run_name": "repo_agent_query",
        "tags": ["repo-agent", "repository-analysis"],
        "metadata": {
            "session_id": session_id,
            "repo_path": repo_path,
            "query": query,
        },
    }

    global _LAST_LANGFUSE_HANDLER
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        from langfuse.langchain import CallbackHandler

        _LAST_LANGFUSE_HANDLER = CallbackHandler()
        config["callbacks"] = [_LAST_LANGFUSE_HANDLER]

    return config


def get_langfuse_trace_id() -> str | None:
    """Return the active Langfuse trace id, if Langfuse tracing is configured."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None

    if getattr(_LAST_LANGFUSE_HANDLER, "last_trace_id", None):
        return _LAST_LANGFUSE_HANDLER.last_trace_id

    from langfuse import get_client

    return get_client().get_current_trace_id()


def record_direct_trace(query: str, answer: str, repo_path: str) -> str | None:
    """Create a trace for deterministic answers so feedback has a trace target."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None

    try:
        from langfuse import get_client

        client = get_client()
        observation = client.start_observation(
            name="repo_agent_direct_answer",
            as_type="chain",
            input=query,
            metadata={
                "repo_path": repo_path,
                "session_id": os.environ.get("REPO_AGENT_SESSION_ID"),
                "source": "deterministic_tool",
            },
        )
        observation.update(output=answer)
        trace_id = observation.trace_id
        observation.end()
        client.flush()
        return trace_id
    except Exception as exc:
        get_observability_logger("cli").log_event("direct_trace_error", error=str(exc), query=query)
        return None


def submit_feedback(
    query: str,
    answer: str,
    correctness: int,
    usefulness: int,
    comment: str = "",
    trace_id: str | None = None,
) -> bool:
    """Persist human quality scores locally and send them to Langfuse when enabled."""
    payload = {
        "query": query,
        "answer_preview": answer[:500],
        "correctness": correctness,
        "usefulness": usefulness,
        "comment": comment,
        "trace_id": trace_id,
    }
    evaluation_logger = get_observability_logger("evaluation")
    session_id = evaluation_logger.session_id
    payload["session_id"] = session_id
    evaluation_logger.log_event("answer_feedback", **payload)

    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    ):
        return False

    from langfuse import get_client

    client = get_client()
    for name, value in (("correctness", correctness), ("usefulness", usefulness)):
        score_args = {
            "name": name,
            "value": float(value),
            "data_type": "NUMERIC",
            "comment": comment or None,
        }
        if trace_id:
            score_args["trace_id"] = trace_id
        elif session_id:
            score_args["session_id"] = session_id
        else:
            return False
        client.create_score(**score_args)
    client.flush()
    return True
