from __future__ import annotations

import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_git_commit_hash(default: str = "unknown") -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return default
    return completed.stdout.strip() or default


def get_git_short_hash(default: str = "unknown") -> str:
    return get_git_commit_hash(default=default)[:8]

