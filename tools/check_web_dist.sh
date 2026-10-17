#!/usr/bin/env bash
# Is src/inframe/web/dist what `npm --prefix web run build` makes from web/ right now? (docs/T9A_DESIGN.md 2.2)
# Builds into a fresh temp folder with the pinned Node and diffs it against the committed dist.
#   exit 0: identical   exit 1: dist differs (rebuild and commit it)   exit 2: wrong Node version, or the build failed to start
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root" || exit 2

want="$(tr -d '[:space:]' < web/.nvmrc)"
have="$(node --version 2>/dev/null | tr -d '[:space:]')"
if [ "${have#v}" != "${want#v}" ]; then
  echo "check_web_dist: Node ${have:-'(not found)'} is not the pinned ${want} (web/.nvmrc). Switch with: nvm use (in web/), then run this again." >&2
  exit 2
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/inframe-dist.XXXXXX")" || exit 2
trap 'rm -rf "$tmp"' EXIT

if ! (cd web && npm exec -- vite build --outDir "$tmp/dist" --emptyOutDir --logLevel error); then
  echo "check_web_dist: the build failed" >&2
  exit 2
fi

if diff -r "$tmp/dist" src/inframe/web/dist; then
  echo "check_web_dist: src/inframe/web/dist is up to date"
  exit 0
fi
echo "check_web_dist: src/inframe/web/dist is stale. Run: npm --prefix web ci && npm --prefix web run build, then commit it." >&2
exit 1
