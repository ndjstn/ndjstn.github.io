#!/usr/bin/env python3
"""Generate a dashboard-ready registry from private ops site manifests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "ops" / "sites"
OUTPUT_PATH = ROOT / "ops" / "managed-sites.json"


def load_site(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        site = json.load(handle)
    site["manifest_path"] = str(path.relative_to(ROOT))
    site["recommended_actions"] = recommended_actions(site)
    return site


def recommended_actions(site: dict) -> list[dict]:
    domain = site.get("domain", "unknown")
    state = site.get("state", "unknown")
    stage = site.get("stage", "unknown")
    automation = site.get("automation", {})
    actions: list[dict] = []

    if state == "live" and automation.get("health_check") == "automatic":
        actions.append({
            "action": "check_health",
            "mode": "automatic",
            "reason": f"{domain} is live and should be monitored.",
        })

    if automation.get("backup") == "automatic":
        actions.append({
            "action": "verify_backup",
            "mode": "automatic",
            "reason": f"{domain} has automatic backups enabled.",
        })

    if stage == "experiment":
        actions.append({
            "action": "review_experiment_signal",
            "mode": "manual",
            "reason": f"{domain} needs keep, kill, or promote review.",
        })

    if site.get("handoff", {}).get("sale_candidate"):
        actions.append({
            "action": "check_sale_readiness",
            "mode": "manual",
            "reason": f"{domain} is marked as a sale candidate.",
        })

    if automation.get("promote_content") == "manual":
        actions.append({
            "action": "review_distribution",
            "mode": "manual",
            "reason": f"{domain} can be promoted when the content angle is strong.",
        })

    if automation.get("promote_content") == "paused":
        actions.append({
            "action": "skip_distribution",
            "mode": "automatic",
            "reason": f"{domain} promotion is paused.",
        })

    return actions


def main() -> None:
    sites = [load_site(path) for path in sorted(SITE_DIR.glob("*.json"))]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "site_count": len(sites),
        "states": sorted({site.get("state", "unknown") for site in sites}),
        "action_count": sum(len(site["recommended_actions"]) for site in sites),
        "sites": sites,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
