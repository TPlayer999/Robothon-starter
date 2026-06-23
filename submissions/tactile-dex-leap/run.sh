#!/usr/bin/env bash
# TACTILE-DEX one-command runner.
#
# Creates an isolated Python 3.12 virtualenv (via uv if available, else venv),
# installs dependencies, and runs the requested pipeline stage. Designed so a
# reviewer can reproduce the whole submission from a clean checkout:
#
#   ./run.sh setup     # create venv + install deps
#   ./run.sh train     # train PPO (default 2M steps; override with TSTEPS=...)
#   ./run.sh eval      # evaluate the trained policy
#   ./run.sh demo      # render demo.mp4
#   ./run.sh data      # record a dataset.npz
#   ./run.sh all       # setup -> train -> eval -> demo
#
# The venv lives in ./.venv and never touches the system Python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

VENV="$ROOT/.venv"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

setup_venv() {
    if [ -x "${PY}" ]; then
        echo "[setup] venv already exists at ${VENV}"
    elif command -v uv >/dev/null 2>&1; then
        echo "[setup] creating venv with uv (Python 3.12)"
        uv venv --python 3.12 "${VENV}"
    else
        echo "[setup] creating venv with python3.12"
        python3.12 -m venv "${VENV}"
    fi
    echo "[setup] installing dependencies"
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "${PY}" -r requirements.txt
        uv pip install --python "${PY}" torch --index-url https://download.pytorch.org/whl/cpu
    else
        "${PIP}" install -r requirements.txt
        "${PIP}" install torch --index-url https://download.pytorch.org/whl/cpu
    fi
    echo "[setup] done."
}

cmd="${1:-help}"
case "${cmd}" in
    setup)
        setup_venv
        ;;
    train)
        [ -x "${PY}" ] || setup_venv
        TSTEPS="${TSTEPS:-3000000}"; NENV="${NENV:-8}"
        "${PY}" train.py --timesteps "${TSTEPS}" --n-envs "${NENV}"
        ;;
    eval)
        [ -x "${PY}" ] || setup_venv
        "${PY}" eval_policy.py --model models/ppo_leap_dex.zip --episodes 20
        ;;
    demo)
        [ -x "${PY}" ] || setup_venv
        "${PY}" demo.py --model models/ppo_leap_dex.zip
        ;;
    data)
        [ -x "${PY}" ] || setup_venv
        "${PY}" data_recorder.py --policy model --episodes 30 --out dataset.npz
        ;;
    all)
        setup_venv
        TSTEPS="${TSTEPS:-2000000}" "${PY}" train.py --timesteps "${TSTEPS}" --n-envs "${NENV:-6}"
        "${PY}" eval_policy.py --model models/ppo_leap_dex.zip --episodes 20
        "${PY}" demo.py --model models/ppo_leap_dex.zip
        ;;
    *)
        echo "Usage: $0 {setup|train|eval|demo|data|all}"
        echo "Env vars: TSTEPS (train steps), NENV (parallel envs)"
        exit 1
        ;;
esac
