#!/usr/bin/env python3
"""Run lesson terminal demos inside disposable containers."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content" / "lessons.json"
FIXTURES = ROOT / "lab" / "fixtures.json"
CONTAINERFILE = ROOT / "lab" / "Containerfile"
ARTIFACT_DIR = ROOT / "artifacts" / "demos"
IMAGE = "localhost/linuxoneliners-lab:latest"

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)


def container_tool() -> str:
    requested = os.environ.get("LOL_CONTAINER_TOOL")
    candidates = [requested] if requested else ["docker", "podman"]
    for candidate in candidates:
        if not candidate:
            continue
        found = run(["bash", "-lc", f"command -v {candidate}"])
        if found.returncode == 0:
            return candidate
    raise RuntimeError("podman or docker is required")


def build_image(tool: str) -> None:
    result = run([tool, "build", "-t", IMAGE, "-f", str(CONTAINERFILE), str(ROOT / "lab")], timeout=600)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def shell_script(fixture_steps: list[str], demo_commands: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "export HOME=/tmp/lol-home",
        "mkdir -p \"$HOME\" /tmp/lol-output",
        "cd /lab",
    ]
    lines.extend(fixture_steps)
    lines.append("echo '::fixture-ready::'")
    for index, command in enumerate(demo_commands, start=1):
        escaped = command.replace("'", "'\"'\"'")
        lines.extend(
            [
                f"echo '$ {escaped}'",
                f"set +e; timeout 4s bash -c {shlex.quote(command)} > /tmp/lol-output/{index}.stdout 2>&1; code=$?; set -e",
                f"cat /tmp/lol-output/{index}.stdout",
                "echo '::exit-code::'${code}",
            ]
        )
    return "\n".join(lines) + "\n"


def assert_no_secrets(text: str, slug: str) -> None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise RuntimeError(f"secret-looking value found in {slug}: {pattern.pattern}")


def run_lesson(tool: str, lesson: dict[str, Any], fixtures: dict[str, list[str]]) -> dict[str, Any]:
    slug = lesson["slug"]
    fixture_name = lesson["terminal_demo"]["fixture"]
    fixture_steps = fixtures.get(fixture_name)
    if not fixture_steps:
        raise RuntimeError(f"missing fixture {fixture_name} for {slug}")

    script = shell_script(fixture_steps, lesson["terminal_demo"]["commands"])
    assert_no_secrets(script, slug)

    with tempfile.TemporaryDirectory(prefix=f"lol-{slug}-") as tmp:
        tmp_path = Path(tmp)
        script_path = tmp_path / "demo.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)

        cmd = [
            tool,
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "all",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "128",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            "--tmpfs",
            "/var:rw,nosuid,nodev,size=64m",
            "-v",
            f"{script_path}:/lab/demo.sh:ro",
            IMAGE,
            "bash",
            "/lab/demo.sh",
        ]
        started = time.time()
        result = run(cmd, timeout=60)
        elapsed_ms = int((time.time() - started) * 1000)

    combined = f"{result.stdout}\n{result.stderr}"
    assert_no_secrets(combined, slug)

    return {
        "slug": slug,
        "title": lesson["title"],
        "fixture": fixture_name,
        "command_count": len(lesson["terminal_demo"]["commands"]),
        "container_tool": tool,
        "image": IMAGE,
        "exit_code": result.returncode,
        "elapsed_ms": elapsed_ms,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
        "security": {
            "network": "none",
            "capabilities": "dropped",
            "no_new_privileges": True,
            "memory": "256m",
            "cpus": "1",
            "pids_limit": 128,
            "tmpfs": ["/tmp", "/var"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-build", action="store_true", help="Reuse the existing lab image")
    parser.add_argument("--slug", help="Run one lesson slug")
    args = parser.parse_args()

    lessons = json.loads(CONTENT.read_text())["lessons"]
    fixtures = json.loads(FIXTURES.read_text())
    if args.slug:
        lessons = [lesson for lesson in lessons if lesson["slug"] == args.slug]
        if not lessons:
            raise SystemExit(f"unknown lesson slug: {args.slug}")

    tool = container_tool()
    if not args.no_build:
        build_image(tool)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_lesson(tool, lesson, fixtures) for lesson in lessons]

    for result in results:
        slug_dir = ARTIFACT_DIR / result["slug"]
        slug_dir.mkdir(parents=True, exist_ok=True)
        (slug_dir / "demo.json").write_text(json.dumps(result, indent=2))
        (slug_dir / "terminal.txt").write_text(result["stdout"] + result["stderr"])

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "container_tool": tool,
        "image": IMAGE,
        "total": len(results),
        "passed": sum(1 for result in results if result["ok"]),
        "failed": [result["slug"] for result in results if not result["ok"]],
        "results": [
            {
                "slug": result["slug"],
                "title": result["title"],
                "ok": result["ok"],
                "elapsed_ms": result["elapsed_ms"],
                "command_count": result["command_count"],
            }
            for result in results
        ],
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
