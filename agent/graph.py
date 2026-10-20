import os
from typing import List, Dict, Any, Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from agent.state import RepositoryAgentState
from agent.guardrails import is_repo_related_query, out_of_scope_message
from agent.observability import get_observability_logger

SYSTEM_PROMPT = """You are an expert Repository Analyst AI Agent.
Your job is to thoroughly analyze a codebase and answer user questions accurately by inspecting files, dependencies, and code patterns.

If the request is not about the active repository, its files, code, dependencies, or implementation details, do not invent an answer. Reply briefly that you can only help with repository questions.

Methodical Guidelines:
1. Always start by using `get_repo_summary` or `list_files` if you need to understand the repository structure.
2. Use `search_code` to locate functions, classes, variables, or keywords.
3. Use `read_file` to read specific code files and understand logic before giving an answer.
4. Use `analyze_dependencies` for package or dependency questions.
5. Use `find_symbol_references` to track how a component or symbol is used across the project.
6. Provide structured, accurate markdown answers referencing file names and line numbers where appropriate.
"""


import json
import re


logger = get_observability_logger("graph")

def extract_json_tool_call(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None
    content = content.strip()
    
    # 1. Try standard JSON loads
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "name" in data:
                args = data.get("arguments", data.get("parameters", {}))
                if not isinstance(args, dict):
                    args = {}
                return {"name": data["name"], "args": args, "id": "fallback_call_1"}
        except Exception:
            pass

    # 2. Try extracting tool name regex if JSON is truncated/malformed
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', content)
    if name_match:
        tool_name = name_match.group(1)
        return {"name": tool_name, "args": {}, "id": "fallback_call_1"}

    return None


def build_repo_agent_graph(
    tools: List[StructuredTool],
    model_name: str = "qwen2.5-coder:0.5b",
    base_url: str = "http://localhost:11434"
):
    """Constructs a compiled LangGraph workflow using ChatOllama and MCP tools."""
    
    # Configure Ollama with CPU optimizations
    llm = ChatOllama(
        model=model_name,
        temperature=0.1,
        base_url=base_url,
        keep_alive="10m",
        num_ctx=2048,
        num_predict=256,
    )

    # Bind MCP tools to Ollama LLM
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    async def agent_node(state: RepositoryAgentState):
        messages = list(state["messages"])
        step_count = state.get("step_count", 0)
        logger.log_event("agent_node_start", step_count=step_count, message_count=len(messages))

        latest_user_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        if latest_user_message and not is_repo_related_query(latest_user_message.content):
            response = AIMessage(content=out_of_scope_message())
            logger.log_event("agent_node_out_of_scope", query=latest_user_message.content)
            return {
                "messages": [response],
                "step_count": step_count + 1
            }

        # Inject system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = await llm_with_tools.ainvoke(messages)

        # Fallback for smaller LLMs that output raw JSON text instead of structured tool_calls
        if not getattr(response, "tool_calls", None) and getattr(response, "content", None):
            fallback_tc = extract_json_tool_call(response.content)
            if fallback_tc:
                response.tool_calls = [fallback_tc]
                response.content = ""  # Clear raw JSON string so it doesn't print raw JSON or confuse small LLMs on next turn

        logger.log_event(
            "agent_node_complete",
            step_count=step_count,
            has_tool_calls=bool(getattr(response, "tool_calls", None)),
            content_preview=(getattr(response, "content", "") or "")[:240]
        )

        return {
            "messages": [response],
            "step_count": step_count + 1
        }

    def should_continue(state: RepositoryAgentState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        step_count = state.get("step_count", 0)

        # Stop after a few tool rounds to keep interactive responses responsive.
        if step_count > 6:
            logger.log_event("graph_stop_max_steps", step_count=step_count)
            return END

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.log_event("graph_continue_tools", step_count=step_count, tool_call_count=len(last_message.tool_calls))
            return "tools"

        logger.log_event("graph_stop_no_tool_calls", step_count=step_count)
        return END

    # Create StateGraph
    workflow = StateGraph(RepositoryAgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    return workflow.compile()
