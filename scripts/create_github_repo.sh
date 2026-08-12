#!/usr/bin/env bash
# Create the GitHub repository and push (requires: gh auth login).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPO="${1:-lucadagati/XAIR_AI_Native_Action_Intents}"
DESC="AI-native Action Intents: VLM producer, XAIR validation harness, Phase P/G evaluation (no paper)"

if ! gh auth status >/dev/null 2>&1; then
  echo "Run: gh auth login" >&2
  exit 1
fi

if gh repo view "$REPO" >/dev/null 2>&1; then
  echo "Repository already exists: https://github.com/${REPO#*/}"
elif git remote get-url origin >/dev/null 2>&1; then
  # Remote was set locally before the GitHub repo existed — create it, then push.
  gh repo create "$REPO" --public --description "$DESC"
else
  gh repo create "$REPO" --public --description "$DESC" --source=. --remote=origin
fi

git push -u origin main
echo "Published: https://github.com/${REPO#*/}"
