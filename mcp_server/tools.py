import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

from agent.observability import get_observability_logger

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv", "env", ".env",
    "dist", "build", ".next", ".nuxt", ".idea", ".vscode", "target", "bin", "obj"
}


logger = get_observability_logger("mcp_server")


def observable_tool(tool_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log_event("tool_start", tool_name=tool_name, args=args, kwargs=kwargs)
            try:
                result = func(*args, **kwargs)
                logger.log_event("tool_complete", tool_name=tool_name, result_preview=str(result)[:240])
                return result
            except Exception as exc:
                logger.log_event("tool_error", tool_name=tool_name, error=str(exc))
                raise
        return wrapper
    return decorator

IGNORE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".zip", ".tar", ".gz", ".7z", ".pdf", ".exe", ".dll", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".wav", ".lock"
}


class RepositoryContext:
    def __init__(self):
        self.active_repo_path: Optional[Path] = None

    def set_repo(self, target: str) -> str:
        target = target.strip()
        
        # If input starts with http/https/git, extract just the URL part
        if target.startswith("http://") or target.startswith("https://") or target.startswith("git@"):
            url = target.split()[0]
            try:
                import git
                repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
                cloned_dir = Path("./cloned_repos") / repo_name
                cloned_dir.parent.mkdir(parents=True, exist_ok=True)

                if cloned_dir.exists():
                    self.active_repo_path = cloned_dir.resolve()
                    return f"Repository already cloned. Set active repository to: {self.active_repo_path}"

                git.Repo.clone_from(url, cloned_dir)
                self.active_repo_path = cloned_dir.resolve()
                return f"Successfully cloned remote repository '{url}'. Set active repository to: {self.active_repo_path}"
            except Exception as e:
                return f"Failed to clone repository: {str(e)}"
        else:
            p = Path(target).resolve()
            if not p.exists():
                return f"Error: Specified path '{target}' does not exist."
            if not p.is_dir():
                return f"Error: Specified path '{target}' is not a directory."
            self.active_repo_path = p
            return f"Set active repository to: {self.active_repo_path}"

    def get_repo_path(self) -> Path:
        if not self.active_repo_path:
            # Default to current working directory if not explicitly set
            self.active_repo_path = Path(".").resolve()
        return self.active_repo_path


# Global Context Instance
repo_ctx = RepositoryContext()


def set_repository_path(path_or_url: str) -> str:
    """Set active local repository path or clone a GitHub URL."""
    return repo_ctx.set_repo(path_or_url)


@observable_tool("get_repo_summary")
def get_repo_summary() -> str:
    """Generates an overview summary of the active repository including language breakdown and key configs."""
    root = repo_ctx.get_repo_path()
    if not root.exists():
        return f"Error: Repository path {root} does not exist."

    total_files = 0
    lang_stats: Dict[str, int] = {}
    config_files: List[str] = []

    known_configs = {
        "requirements.txt", "pyproject.toml", "package.json", "Cargo.toml",
        "go.mod", "pom.xml", "build.gradle", "Dockerfile", "docker-compose.yml",
        "README.md", "README.rst", "Makefile", "tsconfig.json"
    }

    for path in root.rglob("*"):
        if any(ignored in path.parts for ignored in IGNORE_DIRS):
            continue
        if path.is_file():
            if path.name in known_configs:
                config_files.append(str(path.relative_to(root)))

            ext = path.suffix.lower() or "no_ext"
            if ext in IGNORE_EXTENSIONS:
                continue

            total_files += 1
            lang_stats[ext] = lang_stats.get(ext, 0) + 1

    summary = {
        "repository_path": str(root),
        "total_files": total_files,
        "language_extensions_breakdown": dict(sorted(lang_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
        "key_config_files": config_files
    }
    return json.dumps(summary, indent=2)


@observable_tool("list_files")
def list_files(subfolder: str = ".", max_depth: int = 4) -> str:
    """Lists files recursively with relative paths and sizes."""
    root = repo_ctx.get_repo_path()
    
    # Sanitize subfolder argument from small LLM hallucinations
    if not subfolder or subfolder.lower() in ["none", "null", "main", "path/to/repository"]:
        subfolder = "."

    target = (root / subfolder).resolve()

    if not str(target).startswith(str(root)) or not target.exists():
        target = root
        subfolder = "."

    results = []

    def _walk(current: Path, depth: int):
        if depth > max_depth:
            return
        for item in sorted(current.iterdir()):
            if item.name in IGNORE_DIRS or item.name.startswith("."):
                continue
            rel_path = item.relative_to(root)
            if item.is_dir():
                results.append(f"[DIR]  {rel_path}/")
                _walk(item, depth + 1)
            else:
                if item.suffix.lower() in IGNORE_EXTENSIONS:
                    continue
                size_kb = item.stat().st_size / 1024
                results.append(f"[FILE] {rel_path} ({size_kb:.1f} KB)")

    if target.is_file():
        results.append(f"[FILE] {target.relative_to(root)}")
    else:
        _walk(target, 1)

    if not results:
        return f"No files found in '{subfolder}'."

    return "\n".join(results[:300]) + ("\n... [truncated]" if len(results) > 300 else "")


@observable_tool("describe_files")
def describe_files(max_files: int = 100) -> str:
    """Provides brief local descriptions for repository files without LLM calls."""
    root = repo_ctx.get_repo_path()
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".ico"}
    descriptions: Dict[str, str] = {}

    readme_candidates = [root / "README.md", root / "README.rst", root / "README.txt"]
    for readme_path in readme_candidates:
        if not readme_path.exists():
            continue
        try:
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    match = re.match(r"\s*[-*]\s+[`']?([^`'—-]+)[`']?\s+[—-]\s+(.+)", line.strip())
                    if match:
                        descriptions[match.group(1).strip()] = match.group(2).strip()
        except OSError:
            pass
        break

    results = [f"Repository files in {root.name}:"]
    files_seen = 0
    for path in sorted(root.rglob("*")):
        if files_seen >= max_files:
            break
        if not path.is_file() or any(ignored in path.parts for ignored in IGNORE_DIRS):
            continue
        if path.suffix.lower() in IGNORE_EXTENSIONS and path.suffix.lower() not in image_extensions:
            continue

        relative_path = str(path.relative_to(root)).replace("\\", "/")
        description = descriptions.get(path.name)
        if not description and path.suffix.lower() in image_extensions:
            description = "image input/output asset"
        if not description:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    opening_lines = [line.strip() for line in file.readlines()[:12] if line.strip()]
                description = next(
                    (line.lstrip("# ").strip() for line in opening_lines if line.startswith("#")),
                    f"{path.suffix.lstrip('.').upper() or 'data'} file",
                )
            except OSError:
                description = "file"

        results.append(f"- {relative_path}: {description}")
        files_seen += 1

    return "\n".join(results) if files_seen else "No files found in the repository."


@observable_tool("read_file")
def read_file(relative_path: str, start_line: int = 1, max_lines: int = 200) -> str:
    """Reads lines from a specific code file with line numbers."""
    root = repo_ctx.get_repo_path()
    file_path = (root / relative_path).resolve()

    if not str(file_path).startswith(str(root)):
        return "Error: Cannot access files outside repository."

    if not file_path.exists() or not file_path.is_file():
        return f"Error: File '{relative_path}' not found."

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, start_idx + max_lines)

        slice_lines = lines[start_idx:end_idx]
        formatted = [f"{i+1:4d} | {line.rstrip()}" for i, line in enumerate(slice_lines, start=start_idx+1)]

        header = f"=== File: {relative_path} (Showing lines {start_idx+1}-{end_idx} of {total_lines}) ===\n"
        return header + "\n".join(formatted)
    except Exception as e:
        return f"Error reading file '{relative_path}': {str(e)}"


@observable_tool("search_code")
def search_code(query: str, extension: str = "", is_regex: bool = False) -> str:
    """Searches for pattern or text query across all code files in the repository."""
    root = repo_ctx.get_repo_path()
    matches = []

    pattern = None
    if is_regex:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except Exception as e:
            return f"Invalid regex pattern '{query}': {str(e)}"
    else:
        query_lower = query.lower()

    for path in root.rglob("*"):
        if any(ignored in path.parts for ignored in IGNORE_DIRS):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in IGNORE_EXTENSIONS:
            continue
        if extension and not path.name.endswith(extension):
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    hit = False
                    if is_regex and pattern:
                        if pattern.search(line):
                            hit = True
                    elif query_lower in line.lower():
                        hit = True

                    if hit:
                        rel = path.relative_to(root)
                        matches.append(f"{rel}:{line_num}: {line.strip()[:150]}")
                        if len(matches) >= 100:
                            break
        except Exception:
            pass

        if len(matches) >= 100:
            break

    if not matches:
        return f"No code occurrences matching query '{query}' were found."

    return "\n".join(matches) + ("\n... [truncated at 100 matches]" if len(matches) >= 100 else "")


@observable_tool("analyze_dependencies")
def analyze_dependencies() -> str:
    """Parses project dependency files (requirements.txt, package.json, pyproject.toml, Cargo.toml, go.mod)."""
    root = repo_ctx.get_repo_path()
    deps_found = {}

    # 1. Python requirements.txt
    req_file = root / "requirements.txt"
    if req_file.exists():
        py_deps = []
        with open(req_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    py_deps.append(line)
        deps_found["Python (requirements.txt)"] = py_deps

    # 2. Python pyproject.toml
    pyproject_file = root / "pyproject.toml"
    if pyproject_file.exists():
        with open(pyproject_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            deps_found["Python (pyproject.toml)"] = content[:1000]

    # 3. Node.js package.json
    pkg_file = root / "package.json"
    if pkg_file.exists():
        try:
            with open(pkg_file, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                deps_found["Node.js (dependencies)"] = data.get("dependencies", {})
                deps_found["Node.js (devDependencies)"] = data.get("devDependencies", {})
        except Exception as e:
            deps_found["Node.js (package.json)"] = f"Error parsing package.json: {str(e)}"

    # 4. Rust Cargo.toml
    cargo_file = root / "Cargo.toml"
    if cargo_file.exists():
        with open(cargo_file, "r", encoding="utf-8", errors="ignore") as f:
            deps_found["Rust (Cargo.toml)"] = f.read()[:1000]

    # 5. Go go.mod
    go_file = root / "go.mod"
    if go_file.exists():
        with open(go_file, "r", encoding="utf-8", errors="ignore") as f:
            deps_found["Go (go.mod)"] = f.read()[:1000]

    if not deps_found:
        return "No standard dependency configuration files (requirements.txt, package.json, pyproject.toml, Cargo.toml, go.mod) were found."

    return json.dumps(deps_found, indent=2)


@observable_tool("find_symbol_references")
def find_symbol_references(symbol_name: str) -> str:
    """Finds definitions and usages of a class, function, or symbol name across files."""
    root = repo_ctx.get_repo_path()
    definitions = []
    usages = []

    # Patterns for def, class, function, const, etc.
    def_regex = re.compile(r'\b(def|class|function|const|let|var|type|struct|trait|fn|interface)\s+' + re.escape(symbol_name) + r'\b')
    usage_regex = re.compile(r'\b' + re.escape(symbol_name) + r'\b')

    for path in root.rglob("*"):
        if any(ignored in path.parts for ignored in IGNORE_DIRS):
            continue
        if not path.is_file() or path.suffix.lower() in IGNORE_EXTENSIONS:
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    clean_line = line.strip()
                    if def_regex.search(clean_line):
                        rel = path.relative_to(root)
                        definitions.append(f"[DEF]   {rel}:{line_num}: {clean_line[:120]}")
                    elif usage_regex.search(clean_line):
                        rel = path.relative_to(root)
                        usages.append(f"[USAGE] {rel}:{line_num}: {clean_line[:120]}")
                        if len(usages) >= 50:
                            break
        except Exception:
            pass

    output = []
    output.append(f"=== Symbol Search for '{symbol_name}' ===")
    output.append("\n-- Definitions Found --")
    output.extend(definitions if definitions else ["None found."])
    output.append("\n-- Usages Found (capped at 50) --")
    output.extend(usages if usages else ["None found."])

    return "\n".join(output)
