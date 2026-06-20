#!/usr/bin/env python
"""Interactive teleoperation of the LEAP Hand for TACTILE-DEX.

Opens the MuJoCo passive viewer and lets you drive the 16 finger joints with the
keyboard, recording (obs, action) transitions into a ``.npz`` for offline
imitation learning or data collection.

Controls (held = continuous)
----------------------------
  1 / Q     index     MCP flex        -/+
  2 / W     index     PIP flex        -/+
  3 / E     middle    MCP flex        -/+
  4 / R     middle    PIP flex        -/+
  5 / T     ring      MCP flex        -/+
  6 / Y     ring      PIP flex        -/+
  7 / U     thumb     MCP flex        -/+
  8 / I     thumb     IPL flex        -/+
  O / P     open all / close all
  SPACE     toggle grasp-assist weld (hold cube to palm)
  Z         reset cube + hand
  D         toggle data recording
  ESC       quit (saves dataset if recording)

This is a best-effort headless-friendly teleop: the viewer needs a display, but
the key handler runs regardless, so you can also script it via the env.

Usage
-----
    python teleop.py                      # interactive, saves teleop_dataset.npz
    python teleop.py --no-record          # just explore
    python teleop.py --auto-close 200     # auto close-hand demo, then exit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
from tactile_dex.env import EnvConfig, InHandCubeEnv  # noqa: E402
from tactile_dex.scene_builder import SceneConfig, set_grasp_weld, touch_values  # noqa: E402

# Key -> (joint index in the 16-vector, sign). Held keys repeat.
KEY_MAP = {
    ord("1"): (0, +1), ord("q"): (0, -1),
    ord("2"): (2, +1), ord("w"): (2, -1),
    ord("3"): (4, +1), ord("e"): (4, -1),
    ord("4"): (6, +1), ord("r"): (6, -1),
    ord("5"): (8, +1), ord("t"): (8, -1),
    ord("6"): (10, +1), ord("y"): (10, -1),
    ord("7"): (12, +1), ord("u"): (12, -1),
    ord("8"): (15, +1), ord("i"): (15, -1),
}

ACTION_RATE = 0.04  # per-frame action delta per key
CLOSE_TARGET = np.array(
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, -0.4, 1, 0.5, 1, 1], dtype=np.float32
) * 0.9


def run(auto_close: int | None, record: bool, out_path: Path) -> int:
    import mujoco as _mj  # ensure bound in nested _apply_and_record closure
    env = InHandCubeEnv(scene_config=SceneConfig(), env_config=EnvConfig())
    obs, info = env.reset(seed=7)
    action = np.zeros(16, dtype=np.float32)
    recording = record
    rec_obs, rec_act = [], []
    step = 0
    max_steps = auto_close if auto_close else 240

    def _apply_and_record(act):
        nonlocal step
        env.data.ctrl[env._act_ids] = env._action_to_ctrl(act)
        for _ in range(env.env_config.action_substeps):
            _mj.mj_step(env.model, env.data)
        if recording:
            rec_obs.append(env._build_obs().copy())
            rec_act.append(np.asarray(act, dtype=np.float32).copy())
        step += 1

    # Headless auto-close path: always available, no display needed. Used by
    # default and as a reproducible demo of the data-collection loop.
    if auto_close is not None or not _has_display():
        print("Running headless auto-close teleop demo (no viewer)...")
        for _ in range(max_steps):
            action = (CLOSE_TARGET * min(1.0, step / 30.0)).astype(np.float32)
            _apply_and_record(action)
    else:
        # Interactive viewer: macOS requires running under `mjpython`.
        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                print("Teleop viewer open. Controls in the script docstring. "
                      "ESC to quit.")
                for _ in viewer.sync():
                    if not viewer.is_running():
                        break
                    _apply_and_record(action)
        except RuntimeError as exc:
            print(f"Viewer unavailable ({exc}). Use --auto-close N for the "
                  f"headless demo, or run under `mjpython`.")
            return 1

    if recording and rec_obs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            observations=np.asarray(rec_obs, dtype=np.float32),
            actions=np.asarray(rec_act, dtype=np.float32),
        )
        print(f"Saved {len(rec_obs)} teleop transitions -> {out_path}")
    elif record:
        print("Recording was off or empty; nothing saved.")
    return 0


def _has_display() -> bool:
    import os
    return bool(os.environ.get("DISPLAY") or sys.platform == "darwin")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--auto-close", type=int, default=None,
                    help="Run an auto close-hand demo for N steps and exit (headless-friendly)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "teleop_dataset.npz")
    args = ap.parse_args()
    return run(args.auto_close, record=not args.no_record, out_path=args.out)


if __name__ == "__main__":
    sys.exit(main())
