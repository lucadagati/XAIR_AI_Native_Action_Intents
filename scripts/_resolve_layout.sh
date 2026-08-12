#!/usr/bin/env bash
# Resolve layout for Adaptix monorepo, standalone XAIR repo, or paper artifact bundle.
# Sets: REPO_ROOT, SCRIPTS, XAIR_ROOT (ADAPTIX_ROOT alias)
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$_here/../pyproject.toml" ] && [ -d "$_here/../xair" ]; then
  REPO_ROOT="$(cd "$_here/.." && pwd)"
  SCRIPTS="$_here"
  XAIR_ROOT="$REPO_ROOT"
elif [ -d "$_here/../XAIR_Runtime" ]; then
  REPO_ROOT="$(cd "$_here/.." && pwd)"
  SCRIPTS="$_here"
  XAIR_ROOT="$REPO_ROOT/XAIR_Runtime"
elif [ -d "$_here/../xair_runtime" ]; then
  REPO_ROOT="$(cd "$_here/.." && pwd)"
  SCRIPTS="$_here"
  XAIR_ROOT="$REPO_ROOT/xair_runtime"
else
  echo "ERROR: cannot locate XAIR runtime next to scripts/" >&2
  exit 1
fi

ADAPTIX_ROOT="$REPO_ROOT"
export REPO_ROOT SCRIPTS XAIR_ROOT ADAPTIX_ROOT
unset _here
