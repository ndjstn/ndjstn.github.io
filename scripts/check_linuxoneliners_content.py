#!/usr/bin/env python3
"""Validate LinuxOneLiners lesson curation requirements."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONTENT = Path("sites/linuxoneliners.com/content/lessons.json")
REQUIRED_LESSON_FIELDS = [
    "slug",
    "title",
    "series",
    "hook",
    "problem",
    "command",
    "danger",
    "starting_state",
    "what_changed",
    "when_to_use",
    "when_not_to_use",
    "undo",
    "expected_output",
    "terminal_demo",
    "shorts",
    "linkedin",
    "youtube",
    "experiments",
]


def fail(message: str) -> None:
    print(f"content check failed: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    data = json.loads(CONTENT.read_text())
    series_ids = {item["id"] for item in data.get("series", [])}
    slugs: set[str] = set()

    lessons = data.get("lessons", [])
    if not lessons:
        fail("no lessons found")

    for index, lesson in enumerate(lessons, start=1):
        slug = lesson.get("slug", f"lesson_{index}")
        missing = [field for field in REQUIRED_LESSON_FIELDS if not lesson.get(field)]
        if missing:
            fail(f"{slug} missing fields: {', '.join(missing)}")

        if slug in slugs:
            fail(f"duplicate slug: {slug}")
        slugs.add(slug)

        if lesson["series"] not in series_ids:
            fail(f"{slug} uses unknown series {lesson['series']}")

        if lesson["danger"] not in {"safe", "caution", "danger"}:
            fail(f"{slug} has invalid danger rating {lesson['danger']}")

        demo_commands = lesson["terminal_demo"].get("commands", [])
        if len(demo_commands) < 2:
            fail(f"{slug} needs at least two terminal demo commands")

        shorts = lesson["shorts"]
        if shorts.get("format") != "vertical":
            fail(f"{slug} shorts format must be vertical")
        if not 15 <= int(shorts.get("duration_target_seconds", 0)) <= 60:
            fail(f"{slug} shorts duration target must be 15-60 seconds")

        linkedin = lesson["linkedin"]
        if "?" not in linkedin.get("question", ""):
            fail(f"{slug} LinkedIn prompt must be a real question")

        experiments = lesson["experiments"]
        if not isinstance(experiments, list) or not experiments:
            fail(f"{slug} needs at least one experiment")
        for experiment in experiments:
            for field in ["id", "metric", "variant_a", "variant_b"]:
                if not experiment.get(field):
                    fail(f"{slug} experiment missing {field}")

        if lesson["command"].startswith("chmod -R") or lesson["command"].startswith("chown -R"):
            fail(f"{slug} uses recursive permission command as primary command")

    print(f"content check passed: {len(lessons)} curated lessons")


if __name__ == "__main__":
    main()
