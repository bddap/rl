#!/usr/bin/env bash
# Publish the web bundle (rl#411 stage 6): assemble dist from a PUBLISHED release
# bundle — the exact assets+weights the decks pull, one artifact source — and
# force-push it as a single orphan commit to the hosting repo (GitHub Pages serves
# that branch). Single-commit-force keeps the hosting repo one bundle deep: history
# there is worthless (the rl repo is the history) and ~100 MB per push must not
# accrete. Called by rl-release-build (the ONE release pipeline); runnable by hand.
#
#   deploy.sh --bundle <release-version-dir> [--repo <git-url>] [--branch main]
#
# On success the last stdout line is `DEPLOYED <pushed-commit> <rl-sha>` — the
# builder's stamp contract.
set -euo pipefail

# Args before the cd, so a hand-run's relative --bundle resolves against the
# caller's cwd, not game-web/.
BUNDLE='' REPO=${RL_WEB_REPO:-git@github.com:bddap-bot/rl-web.git} BRANCH=main
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) BUNDLE=$(realpath "$2"); shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --branch) BRANCH=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
cd "$(dirname "$0")"   # game-web/ of the checkout being deployed
[ -n "$BUNDLE" ] && [ -d "$BUNDLE/assets" ] && [ -d "$BUNDLE/checkpoints" ] || {
  echo "usage: deploy.sh --bundle <release-version-dir> (needs assets/ + checkpoints/)" >&2
  exit 1
}

STAGE=$(mktemp -d) PUSHDIR=$(mktemp -d)
trap 'rm -rf "$STAGE" "$PUSHDIR"' EXIT

# The web asset tree = the bundle's assets plus its checkpoints as weights/ (the
# shipped game's asset-tree-relative default checkpoint dir). Plain copies — the
# staging dir may sit on a different fs (tmpfs) than the store, so no hardlinks.
mkdir -p "$STAGE/assets"
cp -a "$BUNDLE/assets/." "$STAGE/assets/"
cp -a "$BUNDLE/checkpoints/." "$STAGE/assets/weights/"

BEVY_ASSET_ROOT="$STAGE" bash run.sh dist

RL_SHA=$(git -C .. rev-parse --short=12 HEAD)
cp -a dist/. "$PUSHDIR/"
git -C "$PUSHDIR" init -q -b "$BRANCH"
git -C "$PUSHDIR" -c user.name=bddap-bot -c user.email=bddap.bot@gmail.com \
  add -A
git -C "$PUSHDIR" -c user.name=bddap-bot -c user.email=bddap.bot@gmail.com \
  commit -qm "gcr-web $RL_SHA ($(date -u '+%F %T') UTC)"
git -C "$PUSHDIR" push --force "$REPO" "HEAD:$BRANCH"
echo "DEPLOYED $(git -C "$PUSHDIR" rev-parse HEAD) $RL_SHA"
