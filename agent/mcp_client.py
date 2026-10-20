import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from langchain_core.tools import StructuredTool
from pydantic import create_model, BaseModel, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.observability import get_observability_logger


class MCPClientBridge:
    def __init__(self, server_module: str = "mcp_server.server"):
        self.server_module = server_module
        self.session: Optional[ClientSession] = None
        self._read_stream = None
        self._write_stream = None
        self._client_cm = None
        self._session_cm = None
        self.mcp_tools_raw = []
        self.logger = get_observability_logger("mcp_client")

    async def connect(self):
        """Initializes stdio connection to the MCP Server."""
        self.logger.log_event("connect_start", server_module=self.server_module)
        env = os.environ.copy()
        project_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = project_root + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
        env["REPO_AGENT_SESSION_ID"] = os.environ.get("REPO_AGENT_SESSION_ID", env.get("REPO_AGENT_SESSION_ID", ""))
        if not env["REPO_AGENT_SESSION_ID"]:
            env["REPO_AGENT_SESSION_ID"] = self.logger.session_id

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", self.server_module],
            env=env,
            cwd=project_root
        )

        self._client_cm = stdio_client(params)
        self._read_stream, self._write_stream = await self._client_cm.__aenter__()

        self._session_cm = ClientSession(self._read_stream, self._write_stream)
        self.session = await self._session_cm.__aenter__()

        await self.session.initialize()
        tools_response = await self.session.list_tools()
        self.mcp_tools_raw = tools_response.tools
        self.logger.log_event("connect_complete", tool_count=len(self.mcp_tools_raw))
        return self.mcp_tools_raw

    async def disconnect(self):
        """Cleanly closes MCP session and subprocess."""
        self.logger.log_event("disconnect_start")
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._client_cm:
            try:
                await self._client_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self.logger.log_event("disconnect_complete")

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Executes a tool on the MCP server and returns the text result."""
        if not self.session:
            raise RuntimeError("MCP Session is not connected.")

        self.logger.log_event("tool_call_start", tool_name=tool_name, arguments=arguments)
        try:
            result = await self.session.call_tool(tool_name, arguments)
        except Exception as exc:
            self.logger.log_event("tool_call_error", tool_name=tool_name, error=str(exc))
            raise

        content_items = []
        for item in result.content:
            if hasattr(item, "text"):
                content_items.append(item.text)
            else:
                content_items.append(str(item))

        output = "\n".join(content_items) if content_items else "Tool executed with no output."
        self.logger.log_event("tool_call_complete", tool_name=tool_name, output_preview=output[:240])
        return output

    def get_langchain_tools(self) -> List[StructuredTool]:
        """Converts raw MCP tool definitions into executable LangChain StructuredTools for LangGraph."""
        langchain_tools = []

        for mcp_tool in self.mcp_tools_raw:
            tool_name = mcp_tool.name
            description = mcp_tool.description or f"Tool {tool_name}"
            input_schema_dict = getattr(mcp_tool, "input_schema", getattr(mcp_tool, "inputSchema", {}))
            if hasattr(input_schema_dict, "model_dump"):
                input_schema_dict = input_schema_dict.model_dump()
            elif hasattr(input_schema_dict, "dict"):
                input_schema_dict = input_schema_dict.dict()
            elif not isinstance(input_schema_dict, dict):
                input_schema_dict = dict(input_schema_dict) if input_schema_dict else {}

            schema_properties = input_schema_dict.get("properties", {})
            required_props = input_schema_dict.get("required", [])

            # Dynamically construct Pydantic input schema model
            fields = {}
            for prop_name, prop_spec in schema_properties.items():
                prop_type = str
                if prop_spec.get("type") == "integer":
                    prop_type = int
                elif prop_spec.get("type") == "boolean":
                    prop_type = bool
                elif prop_spec.get("type") == "number":
                    prop_type = float

                default_val = ... if prop_name in required_props else prop_spec.get("default", None)
                fields[prop_name] = (
                    prop_type,
                    Field(default=default_val, description=prop_spec.get("description", ""))
                )

            ArgsSchema = create_model(f"{tool_name}_input_schema", **fields)

            # Define sync and async wrapper functions
            def make_sync_executor(t_name: str):
                def sync_func(**kwargs) -> str:
                    try:
                        return asyncio.run(self.execute_tool(t_name, kwargs))
                    except Exception as e:
                        return f"Error executing tool '{t_name}': {str(e)}"
                return sync_func

            def make_async_executor(t_name: str):
                async def async_func(**kwargs) -> str:
                    try:
                        return await self.execute_tool(t_name, kwargs)
                    except Exception as e:
                        return f"Error executing tool '{t_name}': {str(e)}"
                return async_func

            tool_instance = StructuredTool(
                name=tool_name,
                description=description,
                func=make_sync_executor(tool_name),
                coroutine=make_async_executor(tool_name),
                args_schema=ArgsSchema
            )
            langchain_tools.append(tool_instance)

        return langchain_tools
