#!/usr/bin/env python3
"""Fetch the official DreaMS source used by the public benchmark loader."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPOSITORY = "https://github.com/pluskal-lab/DreaMS.git"
REVISION = "dbec3a0b514a99e5056cfccde4559fda8cfe8129"


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Benchmark workspace root")
    args = parser.parse_args()
    target = args.root.resolve() / "train/comparison/resources/dreams/DreaMS"
    target.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        if any(target.iterdir()):
            raise RuntimeError(f"DreaMS source directory is not a Git checkout: {target}")
        git("init", "--quiet", cwd=target)
        git("remote", "add", "origin", REPOSITORY, cwd=target)
    if git("remote", "get-url", "origin", cwd=target) != REPOSITORY:
        raise RuntimeError(f"Unexpected DreaMS source remote in {target}")
    if not (target / "dreams/models/dreams/dreams.py").is_file() or git("rev-parse", "HEAD", cwd=target) != REVISION:
        git("fetch", "--depth", "1", "origin", REVISION, cwd=target)
        git("checkout", "--detach", "FETCH_HEAD", cwd=target)
    if git("rev-parse", "HEAD", cwd=target) != REVISION:
        raise RuntimeError(f"DreaMS source revision differs from {REVISION}")
    print(f"ready {target} ({REVISION})")


if __name__ == "__main__":
    main()
