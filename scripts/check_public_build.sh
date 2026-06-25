#!/usr/bin/env bash
set -euo pipefail

python3 -m py_compile scripts/generate_site_registry.py

for manifest in ops/sites/*.json; do
  python3 -m json.tool "$manifest" >/dev/null
done

./scripts/generate_site_registry.py

python3 - <<'PY'
import json
from pathlib import Path

registry = json.loads(Path("ops/managed-sites.json").read_text(encoding="utf-8"))
if registry["site_count"] < 1:
    raise SystemExit("site registry has no sites")

for site in registry["sites"]:
    missing = [key for key in ("domain", "state", "stage", "recommended_actions") if key not in site]
    if missing:
        raise SystemExit(f"{site.get('domain', '<unknown>')} missing {missing}")

print(f"registry ok: {registry['site_count']} sites, {registry['action_count']} actions")
PY

JEKYLL_ENV=production bundle exec jekyll build

leaks="$(find _site -maxdepth 2 \( -path '_site/ops' -o -path '_site/deploy' -o -name 'flake.nix' -o -name 'managed-sites.json' \) -print)"
if [[ -n "$leaks" ]]; then
  printf 'private deployment files leaked into _site:\n%s\n' "$leaks" >&2
  exit 1
fi

echo "public build ok"
