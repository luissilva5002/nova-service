"""
nova/ingestion/project_scanner.py

Step 1 of the 3-Step Dynamic Ingestion Workflow (see architecture doc):
"Folder Skeleton Inspection". Builds a lightweight directory tree
(top-level manifest files + top 2-3 levels of the directory) that gets
handed to the LLM for classification in llm_filter.py, WITHOUT reading
full file contents yet.
"""
from pathlib import Path

from nova.config import DEFAULT_IGNORE_DIR_HINTS

MAX_DEPTH = 3
MANIFEST_FILENAMES = {"pubspec.yaml", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"}


def is_binary_file(path: Path, sample_size: int = 1024) -> bool:
    """Python's classic is_binary heuristic: look for a NUL byte in a sample."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\0" in chunk
    except OSError:
        return False


def build_directory_tree(project_root: Path, max_depth: int = MAX_DEPTH) -> dict:
    """
    Returns a nested dict skeleton of the project, e.g.:
    {
      "name": "my_flutter_app",
      "manifests": ["pubspec.yaml"],
      "children": [{"name": "lib", "children": [...]}, ...]
    }
    Directories matching DEFAULT_IGNORE_DIR_HINTS are still listed (so the
    LLM can see and confirm them) but not recursed into deeply.
    """
    project_root = Path(project_root)

    def walk(path: Path, depth: int) -> dict:
        node = {"name": path.name or str(path), "children": []}
        if depth >= max_depth:
            return node
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except (PermissionError, FileNotFoundError):
            return node

        manifests = [e.name for e in entries if e.is_file() and e.name in MANIFEST_FILENAMES]
        if manifests:
            node["manifests"] = manifests

        for entry in entries:
            if entry.name.startswith(".") and entry.name not in (".dart_tool", ".gradle"):
                continue
            if entry.is_dir():
                hinted = any(hint.lower() in entry.name.lower() for hint in DEFAULT_IGNORE_DIR_HINTS)
                child_depth = depth + 1 if not hinted else max_depth  # don't recurse deep into obvious build dirs
                node["children"].append(walk(entry, child_depth))
            else:
                node["children"].append({"name": entry.name, "file": True})
        return node

    return walk(project_root, 0)


def flatten_tree_paths(tree: dict, prefix: str = "") -> list:
    """Flattens the nested tree into a list of relative path strings, useful
    for building the compact prompt sent to the LLM classifier."""
    paths = []
    name = tree.get("name", "")
    current = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
    if tree.get("file"):
        paths.append(current)
        return paths
    paths.append(current + "/")
    for child in tree.get("children", []):
        paths.extend(flatten_tree_paths(child, current))
    return paths
