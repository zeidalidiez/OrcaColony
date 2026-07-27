#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z"
)
VERSION_PATHS = (
    "VERSION",
    "README.md",
    "PROGRESS_REPORT.md",
    "pyproject.toml",
    "src/orcacolony/__init__.py",
    "uv.lock",
)


class VersionError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str, label: str) -> Version:
        match = VERSION_PATTERN.fullmatch(value.strip())
        if match is None:
            raise VersionError(
                f"{label} must contain a stable MAJOR.MINOR.PATCH version"
            )
        return cls(*(int(part) for part in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _package_version(source: str, label: str) -> str:
    tree = ast.parse(source, filename=label)
    values: list[str] = []
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "__version__"
                for target in statement.targets
            )
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            values.append(statement.value.value)
    if len(values) != 1:
        raise VersionError(f"{label} must assign __version__ exactly once")
    return values[0]


def _lock_version(source: str, label: str) -> str:
    payload = tomllib.loads(source)
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise VersionError(f"{label} does not contain a package list")
    matches = [
        package.get("version")
        for package in packages
        if isinstance(package, dict) and package.get("name") == "orcacolony"
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise VersionError(
            f"{label} must contain one versioned orcacolony package"
        )
    return matches[0]


def _markdown_version(
    source: str,
    label: str,
    *,
    required: bool,
) -> str | None:
    matches = re.findall(
        r"^\*\*Repository version:\*\* \[`([^`]+)`\]\(VERSION\)$",
        source,
        flags=re.MULTILINE,
    )
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise VersionError(
            f"{label} must show one linked repository version"
        )
    return matches[0]


def _snapshot(
    reader: Callable[[str], str | None],
    label: str,
    *,
    require_version_file: bool,
) -> Version:
    sources = {path: reader(path) for path in VERSION_PATHS}
    always_required = (
        "pyproject.toml",
        "src/orcacolony/__init__.py",
        "uv.lock",
    )
    conditionally_required = (
        "VERSION",
        "README.md",
        "PROGRESS_REPORT.md",
    )
    required_paths = (
        (*always_required, *conditionally_required)
        if require_version_file
        else always_required
    )
    missing_required = [
        path for path in required_paths if sources[path] is None
    ]
    if missing_required:
        raise VersionError(
            f"{label} is missing version sources: "
            + ", ".join(missing_required)
        )

    pyproject = tomllib.loads(sources["pyproject.toml"] or "")
    project = pyproject.get("project")
    project_version = (
        project.get("version") if isinstance(project, dict) else None
    )
    if not isinstance(project_version, str):
        raise VersionError(f"{label} pyproject.toml has no project version")

    declared: dict[str, str] = {
        "pyproject.toml": project_version,
        "src/orcacolony/__init__.py": _package_version(
            sources["src/orcacolony/__init__.py"] or "",
            f"{label}:src/orcacolony/__init__.py",
        ),
        "uv.lock": _lock_version(
            sources["uv.lock"] or "",
            f"{label}:uv.lock",
        ),
    }
    for path in ("README.md", "PROGRESS_REPORT.md"):
        markdown_version = _markdown_version(
            sources[path] or "",
            f"{label}:{path}",
            required=require_version_file,
        )
        if markdown_version is not None:
            declared[path] = markdown_version
    version_file = sources["VERSION"]
    if version_file is None:
        if require_version_file:
            raise VersionError(f"{label} is missing VERSION")
    else:
        declared["VERSION"] = version_file.strip()

    parsed = {
        path: Version.parse(value, f"{label}:{path}")
        for path, value in declared.items()
    }
    unique = set(parsed.values())
    if len(unique) != 1:
        details = ", ".join(
            f"{path}={version}" for path, version in sorted(parsed.items())
        )
        raise VersionError(f"{label} version sources differ: {details}")
    return next(iter(unique))


def _worktree_reader(path: str) -> str | None:
    candidate = ROOT / path
    return candidate.read_text(encoding="utf-8") if candidate.is_file() else None


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _revision_reader(revision: str) -> Callable[[str], str | None]:
    def read(path: str) -> str | None:
        result = _git("show", f"{revision}:{path}", check=False)
        return result.stdout if result.returncode == 0 else None

    return read


def _revision_snapshot(
    revision: str,
    *,
    require_version_file: bool,
) -> Version:
    return _snapshot(
        _revision_reader(revision),
        revision,
        require_version_file=require_version_file,
    )


def check_worktree(base_ref: str) -> None:
    current = _snapshot(
        _worktree_reader,
        "working tree",
        require_version_file=True,
    )
    previous = _revision_snapshot(
        base_ref,
        require_version_file=False,
    )
    if current <= previous:
        raise VersionError(
            f"working tree version {current} must be greater than "
            f"{base_ref} version {previous}"
        )
    print(f"version check passed: {previous} -> {current}")


def check_commit_range(base_ref: str) -> None:
    commits = _git(
        "rev-list",
        "--reverse",
        "--topo-order",
        f"{base_ref}..HEAD",
    ).stdout.splitlines()
    if not commits:
        raise VersionError(f"no commits found after base ref {base_ref}")
    checked: list[tuple[str, Version, Version]] = []
    for commit in commits:
        parent_result = _git("rev-parse", f"{commit}^1", check=False)
        if parent_result.returncode != 0:
            raise VersionError(
                f"cannot resolve first parent for commit {commit}"
            )
        parent = parent_result.stdout.strip()
        previous = _revision_snapshot(
            parent,
            require_version_file=False,
        )
        current = _revision_snapshot(
            commit,
            require_version_file=True,
        )
        if current <= previous:
            raise VersionError(
                f"commit {commit} version {current} must be greater than "
                f"first-parent version {previous}"
            )
        checked.append((commit, previous, current))
    for commit, previous, current in checked:
        print(f"{commit[:12]}: {previous} -> {current}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify synchronized OrcaColony versions and per-commit increases"
        )
    )
    parser.add_argument("--base-ref")
    parser.add_argument(
        "--working-tree",
        action="store_true",
        help="compare current files with --base-ref instead of committed history",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.working_tree:
            check_worktree(args.base_ref or "HEAD")
        elif args.base_ref:
            check_commit_range(args.base_ref)
        else:
            version = _snapshot(
                _worktree_reader,
                "working tree",
                require_version_file=True,
            )
            print(f"version sources agree: {version}")
    except (
        OSError,
        SyntaxError,
        subprocess.CalledProcessError,
        tomllib.TOMLDecodeError,
        VersionError,
    ) as exc:
        print(f"version check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
