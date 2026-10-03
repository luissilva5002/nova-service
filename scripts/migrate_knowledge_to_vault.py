#!/usr/bin/env python3
"""Copy Markdown notes from Nova's local knowledge directory to WebObsidian."""
import os
import sys
from pathlib import Path, PurePosixPath


def load_nova_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    supported = {"AGENT_API_KEY", "AGENT_API_BASE_URL", "NOVA_VAULT_DIR"}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        if name not in supported or name in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value


load_nova_env()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova.config import PERSISTENT_MEMORY_VAULT_DIR
from nova.memory.knowledge_client import write_knowledge


def main() -> None:
    source_root = PERSISTENT_MEMORY_VAULT_DIR / "knowledge"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Knowledge source directory does not exist: {source_root}")

    notes = sorted(source_root.rglob("*.md"))
    if not notes:
        raise RuntimeError(f"No Markdown notes found under {source_root}")

    for source in notes:
        relative_path = source.relative_to(source_root)
        vault_path = PurePosixPath("knowledge", relative_path.as_posix()).as_posix()
        content = source.read_bytes().decode("utf-8")
        write_knowledge(vault_path, content)
        print(f"Migrated {source} -> {vault_path}")

    print(f"Migrated {len(notes)} Markdown note(s). Source files were left unchanged.")


if __name__ == "__main__":
    main()
