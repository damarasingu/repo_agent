import os
import sys
import asyncio
import argparse
import re
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.status import Status
from rich.table import Table
from rich.syntax import Syntax

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent.mcp_client import MCPClientBridge
from agent.graph import build_repo_agent_graph
from agent.guardrails import is_repo_related_query, out_of_scope_message, resolve_repo_path_from_query
from agent.observability import (
    get_langfuse_run_config,
    get_langfuse_trace_id,
    get_observability_logger,
    record_direct_trace,
    submit_feedback,
)
from mcp_server.tools import repo_ctx, get_repo_summary, analyze_dependencies, list_files, describe_files

console = Console()


class RepoAgentCLI:
    def __init__(self, model_name: str = "qwen2.5-coder:0.5b", repo_path: str = "."):
        self.model_name = model_name
        self.initial_repo_path = repo_path
        self.mcp_bridge: Optional[MCPClientBridge] = None
        self.graph = None
        self.chat_history = []
        self.logger = get_observability_logger("cli")
        self.last_query: Optional[str] = None
        self.last_answer: Optional[str] = None
        self.last_trace_id: Optional[str] = None

    async def initialize(self):
        """Connects MCP Client to Server and builds LangGraph workflow."""
        self.logger.log_event("initialize_start", model_name=self.model_name, repo_path=self.initial_repo_path)
        console.print(f"[bold cyan]Connecting to MCP Server via stdio...[/bold cyan]")
        self.mcp_bridge = MCPClientBridge()
        mcp_tools = await self.mcp_bridge.connect()
        
        console.print(f"[bold green]Connected! Discovered {len(mcp_tools)} MCP tools.[/bold green]")

        # Set initial repo path
        set_res = repo_ctx.set_repo(self.initial_repo_path)
        console.print(f"[dim]{set_res}[/dim]\n")

        # Get LangChain tools and build LangGraph graph
        lc_tools = self.mcp_bridge.get_langchain_tools()
        self.graph = build_repo_agent_graph(
            tools=lc_tools,
            model_name=self.model_name
        )
        self.logger.log_event("initialize_complete", tool_count=len(mcp_tools))

    async def shutdown(self):
        """Disconnects MCP Client session."""
        self.logger.log_event("shutdown_start")
        if self.mcp_bridge:
            await self.mcp_bridge.disconnect()
            console.print("[dim]MCP Client disconnected cleanly.[/dim]")
        self.logger.log_event("shutdown_complete")

    def print_banner(self):
        """Displays formatted welcome banner."""
        active_repo = str(repo_ctx.get_repo_path())
        text = (
            f"[bold magenta]RepoAgent CLI[/bold magenta] | [yellow]Ollama Local LLM + LangGraph + MCP[/yellow]\n"
            f"[bold white]Active Repo:[/bold white] [green]{active_repo}[/green] | [bold white]Model:[/bold white] [cyan]{self.model_name}[/cyan]\n"
            f"[dim]Commands: [bold]/repo <path_or_url>[/bold] | [bold]/summary[/bold] | [bold]/deps[/bold] | [bold]/files[/bold] | [bold]/feedback 1-5 1-5[/bold] | [bold]/clear[/bold] | [bold]/exit[/bold][/dim]"
        )
        console.print(Panel(text, title="[Agent] Repository Analysis System", border_style="cyan"))

    async def run_query_simple(self, query: str):
        """Executes query through LangGraph agent and prints markdown response."""
        self.logger.log_event("user_query_received", query=query)

        resolved_repo = resolve_repo_path_from_query(query, str(Path(__file__).resolve().parent))
        if resolved_repo:
            set_res = repo_ctx.set_repo(resolved_repo)
            self.logger.log_event("query_repo_autoset", query=query, resolved_repo=resolved_repo, result=set_res)
            console.print(f"[dim]{set_res}[/dim]")

        normalized_query = query.strip().lower()
        if any(phrase in normalized_query for phrase in ["summary of the repo", "summary of repo", "repository summary", "repo summary", "overview of the repo"]):
            self.logger.log_event("query_fast_path_summary", query=query)
            summary = get_repo_summary()
            self.last_query = query
            self.last_answer = summary
            self.last_trace_id = record_direct_trace(query, summary, str(repo_ctx.get_repo_path()))
            console.print(f"\n[bold cyan]> User:[/bold cyan] {query}")
            console.print("\n[bold green][Agent Response]:[/bold green]")
            console.print(Panel(Syntax(summary, "json", theme="monokai"), title="Repository Summary", border_style="green"))
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=summary))
            return

        file_overview_phrases = [
            "explanation of each file",
            "explain each file",
            "explain every file",
            "what each file does",
            "overview of the files",
            "overview of files",
            "files and directories",
            "input and output files",
            "describe the files",
            "brief explanation of the files",
        ]
        if any(phrase in normalized_query for phrase in file_overview_phrases):
            self.logger.log_event("query_fast_path_file_descriptions", query=query)
            descriptions = describe_files()
            self.last_query = query
            self.last_answer = descriptions
            self.last_trace_id = record_direct_trace(query, descriptions, str(repo_ctx.get_repo_path()))
            console.print(f"\n[bold cyan]> User:[/bold cyan] {query}")
            console.print("\n[bold green][Agent Response]:[/bold green]")
            console.print(Panel(descriptions, title="File Descriptions", border_style="green"))
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=descriptions))
            return

        file_question = re.search(r"[`']?([^`'\s]+\.(?:py|js|ts|md))[`']?", normalized_query)
        if file_question and any(phrase in normalized_query for phrase in ["what does", "what is", "what do", "purpose of", "explain"]):
            requested_file = file_question.group(1)
            descriptions = describe_files()
            matching_description = next(
                (line for line in descriptions.splitlines() if requested_file.lower() in line.lower()),
                f"No verified description was found for {requested_file}.",
            )
            self.logger.log_event("query_fast_path_file_question", query=query, file=requested_file)
            self.last_query = query
            self.last_answer = matching_description
            self.last_trace_id = record_direct_trace(query, matching_description, str(repo_ctx.get_repo_path()))
            console.print(f"\n[bold cyan]> User:[/bold cyan] {query}")
            console.print("\n[bold green][Agent Response]:[/bold green]")
            console.print(Panel(matching_description, title="Verified File Description", border_style="green"))
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=matching_description))
            return

        if any(phrase in normalized_query for phrase in ["dependencies", "dependency files", "required packages", "packages used", "what packages"]):
            self.logger.log_event("query_fast_path_dependencies", query=query)
            dependencies = analyze_dependencies()
            self.last_query = query
            self.last_answer = dependencies
            self.last_trace_id = record_direct_trace(query, dependencies, str(repo_ctx.get_repo_path()))
            console.print(f"\n[bold cyan]> User:[/bold cyan] {query}")
            console.print("\n[bold green][Agent Response]:[/bold green]")
            if dependencies.startswith("{"):
                console.print(Panel(Syntax(dependencies, "json", theme="monokai"), title="Dependencies", border_style="green"))
            else:
                console.print(Panel(dependencies, title="Dependencies", border_style="green"))
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=dependencies))
            return

        if any(phrase in normalized_query for phrase in ["key files", "important files", "main files", "file list", "what files"]):
            self.logger.log_event("query_fast_path_files", query=query)
            console.print(f"\n[bold cyan]> User:[/bold cyan] {query}")
            files_str = list_files(max_depth=2)
            self.last_query = query
            self.last_answer = files_str
            self.last_trace_id = record_direct_trace(query, files_str, str(repo_ctx.get_repo_path()))
            console.print("\n[bold green][Agent Response]:[/bold green]")
            console.print(Panel(files_str, title="📁 Key Files"))
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=files_str))
            return

        self.chat_history.append(HumanMessage(content=query))

        if not is_repo_related_query(query):
            response = out_of_scope_message()
            self.logger.log_event("query_rejected_out_of_scope", query=query, response=response)
            console.print(f"\n[bold cyan]> User:[/bold cyan] {query}")
            console.print("\n[bold green][Agent Response]:[/bold green]")
            console.print(Panel(Markdown(response), border_style="green"))
            self.chat_history.append(AIMessage(content=response))
            return

        state = {
            "messages": self.chat_history[-8:],
            "repo_path": str(repo_ctx.get_repo_path()),
            "step_count": 0,
            "last_tool_called": None
        }

        console.print(f"\n[bold cyan]> User:[/bold cyan] {query}")

        with Status("[bold yellow]Analyzing codebase...[/bold yellow]", console=console, spinner="dots") as status:
            try:
                started_at = time.perf_counter()
                self.logger.log_event("graph_invoke_start", repo_path=str(repo_ctx.get_repo_path()))
                res = await self.graph.ainvoke(
                    state,
                    config=get_langfuse_run_config(
                        repo_path=str(repo_ctx.get_repo_path()),
                        query=query,
                    ),
                )
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                self.last_trace_id = get_langfuse_trace_id()
                self.chat_history = list(res["messages"])
                self.logger.log_event(
                    "graph_invoke_complete",
                    message_count=len(res.get("messages", [])),
                    duration_ms=duration_ms,
                )

                # Find final AI message
                final_ai_msg = None
                for msg in reversed(res["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_ai_msg = msg
                        break

                if final_ai_msg:
                    self.last_query = query
                    self.last_answer = final_ai_msg.content
                    console.print("\n[bold green][Agent Response]:[/bold green]")
                    console.print(Panel(Markdown(final_ai_msg.content), border_style="green"))
                    console.print(f"[dim]Response time: {duration_ms / 1000:.2f}s[/dim]")
                else:
                    fallback_summary = get_repo_summary()
                    self.last_query = query
                    self.last_answer = fallback_summary
                    self.logger.log_event("graph_empty_response_fallback", query=query)
                    console.print("\n[bold green][Agent Response]:[/bold green]")
                    console.print(
                        Panel(
                            Syntax(fallback_summary, "json", theme="monokai"),
                            title="Repository Summary",
                            border_style="green",
                        )
                    )
                    console.print("[dim]The model returned no text, so RepoAgent displayed the repository summary directly.[/dim]")
            except Exception as e:
                self.logger.log_event("graph_invoke_error", error=str(e))
                console.print(f"\n[bold red]Error executing query:[/bold red] {str(e)}")

    async def repl(self):
        """Interactive REPL loop for user commands and questions."""
        self.print_banner()

        while True:
            try:
                user_input = Prompt.ask("\n[bold magenta]repo-agent>[/bold magenta]").strip()
                if not user_input:
                    continue

                trailing_feedback = re.search(r"\s+/feedback(?:\s+(\d)\s+(\d))?\s*$", user_input, re.IGNORECASE)
                if trailing_feedback and not user_input.lower().startswith("/feedback"):
                    question = user_input[:trailing_feedback.start()].strip()
                    await self.run_query_simple(question)
                    if not self.last_answer:
                        continue
                    if trailing_feedback.group(1) and trailing_feedback.group(2):
                        feedback_input = f"/feedback {trailing_feedback.group(1)} {trailing_feedback.group(2)}"
                    else:
                        feedback_input = Prompt.ask("[bold magenta]Feedback scores (correctness usefulness)[/bold magenta]").strip()
                        if not feedback_input.startswith("/feedback"):
                            feedback_input = f"/feedback {feedback_input}"
                    user_input = feedback_input

                if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                    self.logger.log_event("command_exit")
                    console.print("[bold yellow]Exiting RepoAgent. Goodbye![/bold yellow]")
                    break

                elif user_input.startswith("/repo"):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) > 1:
                        target = parts[1].strip()
                        self.logger.log_event("command_repo_set_start", target=target)
                        res = repo_ctx.set_repo(target)
                        self.logger.log_event("command_repo_set_complete", target=target, result=res)
                        console.print(f"[bold green]{res}[/bold green]")
                        if len(parts) > 2:
                            question = parts[2].strip()
                            await self.run_query_simple(question)
                    else:
                        self.logger.log_event("command_repo_query_current")
                        console.print(f"[yellow]Current repo: {repo_ctx.get_repo_path()}[/yellow]")

                elif user_input == "/summary":
                    self.logger.log_event("command_summary")
                    summary = get_repo_summary()
                    console.print(Panel(Syntax(summary, "json", theme="monokai"), title="📊 Repository Telemetry"))

                elif user_input == "/deps":
                    self.logger.log_event("command_dependencies")
                    deps = analyze_dependencies()
                    console.print(Panel(Syntax(deps, "json", theme="monokai"), title="📦 Dependencies"))

                elif user_input == "/files":
                    self.logger.log_event("command_files")
                    files_str = list_files()
                    console.print(Panel(files_str, title="📁 Files List"))

                elif user_input == "/clear":
                    self.chat_history = []
                    self.logger.log_event("command_clear_history")
                    console.print("[bold green]Cleared conversation memory.[/bold green]")

                elif user_input.startswith("/feedback"):
                    parts = user_input.split(maxsplit=3)
                    if len(parts) < 3 or not self.last_query or not self.last_answer:
                        console.print("[yellow]Run a query first, then use /feedback <correctness 1-5> <usefulness 1-5> [comment].[/yellow]")
                        continue
                    try:
                        correctness = int(parts[1])
                        usefulness = int(parts[2])
                        if not 1 <= correctness <= 5 or not 1 <= usefulness <= 5:
                            raise ValueError
                    except ValueError:
                        console.print("[yellow]Scores must be integers from 1 to 5.[/yellow]")
                        continue
                    comment = parts[3] if len(parts) == 4 else ""
                    submitted_to_langfuse = submit_feedback(
                        query=self.last_query,
                        answer=self.last_answer,
                        correctness=correctness,
                        usefulness=usefulness,
                        comment=comment,
                        trace_id=self.last_trace_id,
                    )
                    message = "Feedback scores submitted to Langfuse." if submitted_to_langfuse else "Feedback saved locally; no Langfuse trace was available."
                    average_score = (correctness + usefulness) / 2
                    console.print(
                        f"[green]{message}\n"
                        f"Correctness: {correctness}/5 | Usefulness: {usefulness}/5 | "
                        f"Average: {average_score:.1f}/5[/green]"
                    )

                elif user_input == "/help":
                    self.logger.log_event("command_help")
                    self.print_banner()

                else:
                    await self.run_query_simple(user_input)

            except (KeyboardInterrupt, EOFError):
                self.logger.log_event("command_interrupt_exit")
                console.print("\n[bold yellow]Exiting RepoAgent. Goodbye![/bold yellow]")
                break


async def run_cli(
    repo_path: str,
    model_name: str,
    query: Optional[str],
    correctness: Optional[int] = None,
    usefulness: Optional[int] = None,
    feedback_comment: str = "",
):
    agent = RepoAgentCLI(model_name=model_name, repo_path=repo_path)
    await agent.initialize()
    try:
        if query:
            await agent.run_query_simple(query)
            if correctness is not None and usefulness is not None and agent.last_answer:
                submitted_to_langfuse = submit_feedback(
                    query=agent.last_query or query,
                    answer=agent.last_answer,
                    correctness=correctness,
                    usefulness=usefulness,
                    comment=feedback_comment,
                    trace_id=agent.last_trace_id,
                )
                message = "Feedback scores submitted to Langfuse." if submitted_to_langfuse else "Feedback saved locally; no Langfuse trace was available."
                average_score = (correctness + usefulness) / 2
                console.print(
                    f"[green]{message}\n"
                    f"Correctness: {correctness}/5 | Usefulness: {usefulness}/5 | "
                    f"Average: {average_score:.1f}/5[/green]"
                )
        else:
            await agent.repl()
    finally:
        await agent.shutdown()


def main():
    parser = argparse.ArgumentParser(description="RepoAgent CLI")
    parser.add_argument("--repo", default=".", help="Repository path or URL to analyze")
    parser.add_argument("--model", default="qwen2.5-coder:0.5b", help="Ollama model name")
    parser.add_argument("--query", help="Run a single query and exit")
    parser.add_argument("--correctness", type=int, choices=range(1, 6), help="Score the answer correctness from 1 to 5")
    parser.add_argument("--usefulness", type=int, choices=range(1, 6), help="Score the answer usefulness from 1 to 5")
    parser.add_argument("--feedback-comment", default="", help="Optional comment for the answer scores")
    args = parser.parse_args()

    if (args.correctness is None) != (args.usefulness is None):
        parser.error("--correctness and --usefulness must be provided together")

    asyncio.run(run_cli(args.repo, args.model, args.query, args.correctness, args.usefulness, args.feedback_comment))


if __name__ == "__main__":
    main()
