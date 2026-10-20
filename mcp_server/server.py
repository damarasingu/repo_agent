import sys
from mcp.server.mcpserver import MCPServer
from mcp_server import tools

mcp_server = MCPServer("repo-analysis-server")


@mcp_server.tool()
def set_repository_path(path_or_url: str) -> str:
    """Set active local repository path or clone a GitHub URL."""
    return tools.set_repository_path(path_or_url)


@mcp_server.tool()
def get_repo_summary() -> str:
    """Generates an overview summary of the active repository including language breakdown and key configs."""
    return tools.get_repo_summary()


@mcp_server.tool()
def list_files(subfolder: str = ".", max_depth: int = 4) -> str:
    """Lists files recursively with relative paths, sizes, and line counts."""
    return tools.list_files(subfolder, max_depth)


@mcp_server.tool()
def read_file(relative_path: str, start_line: int = 1, max_lines: int = 200) -> str:
    """Reads lines from a specific code file with line numbers."""
    return tools.read_file(relative_path, start_line, max_lines)


@mcp_server.tool()
def search_code(query: str, extension: str = "", is_regex: bool = False) -> str:
    """Searches for pattern or text query across all code files in the repository."""
    return tools.search_code(query, extension, is_regex)


@mcp_server.tool()
def analyze_dependencies() -> str:
    """Parses project dependency files (requirements.txt, package.json, pyproject.toml, Cargo.toml, go.mod)."""
    return tools.analyze_dependencies()


@mcp_server.tool()
def find_symbol_references(symbol_name: str) -> str:
    """Finds definitions and usages of a class, function, or symbol name across files."""
    return tools.find_symbol_references(symbol_name)


if __name__ == "__main__":
    mcp_server.run()
