#!/usr/bin/env bash
# SSH to Tailscale GPU node (L40). Requires config/compute-gpu.env (gitignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="$ROOT/config/compute-gpu.env"

if [ -f "$ENV_FILE" ]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
fi

: "${GPU_HOST:?Set GPU_HOST in config/compute-gpu.env}"
: "${GPU_USER:=ml}"

REMOTE="${GPU_USER}@${GPU_HOST}"

if [ $# -eq 0 ]; then
  set -- bash -l
fi

if [ -n "${GPU_SSH_IDENTITY_FILE:-}" ] && [ -f "${GPU_SSH_IDENTITY_FILE}" ]; then
  exec ssh -i "${GPU_SSH_IDENTITY_FILE}" -o StrictHostKeyChecking=accept-new "$REMOTE" "$@"
fi

if [ -n "${GPU_SSH_PASSWORD:-}" ] && command -v sshpass >/dev/null; then
  # sshpass -e reads the password from SSHPASS, not from our own variable name.
  SSHPASS="$GPU_SSH_PASSWORD" exec sshpass -e ssh -o StrictHostKeyChecking=accept-new "$REMOTE" "$@"
fi

exec ssh -o StrictHostKeyChecking=accept-new "$REMOTE" "$@"
