from typing import Sequence, TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RepositoryAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    repo_path: str
    step_count: int
    last_tool_called: Optional[str]
