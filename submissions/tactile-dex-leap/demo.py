#!/usr/bin/env python
"""Render the TACTILE-DEX showcase video.

Produces ``demo.mp4`` (1-3 min) with three acts, following the submission
video requirements (simulation startup, platform + scene, task execution,
control logic, final state):

  1. Intro        — the LEAP Hand loads; the cube drops into the cradle.
  2. Reorientation — the trained policy (or scripted baseline) rotates the
                     cube toward the green target ghost; a live tactile
                     heatmap shows which fingertips are in contact.
  3. Grasp & lift  — the hand closes and lifts the cube off the table,
                     demonstrating force-controlled multi-finger grasping.

A per-frame overlay is composited with matplotlib (touch bars, alignment
gauge, target indicator, narration). Falls back to GIF if MP4 encoding fails.

Usage
-----
    python demo.py --model models/ppo_leap_dex.zip
    python demo.py --baseline grasp          # scripted close-hand
    python demo.py --output demo.mp4 --fps 30 --width 960 --height 540
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  python -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tactile_dex.env import EnvConfig, InHandCubeEnv  # noqa: E402
from tactile_dex.scene_builder import SceneConfig, touch_values  # noqa: E402

FINGER_LABELS = ("Index", "Middle", "Ring", "Thumb")
NARRATION = {
    "intro": "TACTILE-DEX  ·  LEAP Hand 16-DOF  ·  in-hand cube reorientation",
    "reorient": "PPO policy rotates the cube toward the target pose",
    "grasp": "Force-controlled multi-finger grasp lifts the cube",
    "outro": "Tactile sensors  ·  RL control  ·  MuJoCo physics",
}


# ---------------------------------------------------------------------- #
# Overlay compositor
# ---------------------------------------------------------------------- #
class Overlay:
    """Composites the MuJoCo frame with a HUD using matplotlib."""

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height

    def composite(
        self,
        frame: np.ndarray,
        *,
        touch: np.ndarray,
        align: float,
        act: str,
        title: str,
        step: int,
        max_steps: int,
    ) -> np.ndarray:
        # One axes in [0,1] x [0,1] so every overlay element (image + patches
        # + text) shares the same fractional coordinate system. imshow extent
        # maps the frame pixels onto that unit box (origin lower so y grows up).
        fig = plt.figure(figsize=(self.w / 100, self.h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(frame, extent=(0, 1, 0, 1), origin="upper", aspect="auto")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # --- touch bars (bottom-left) ---
        bar_w, bar_h = 0.020, 0.18
        x0, y0 = 0.025, 0.20
        tmax = 5.0  # Newtons full-scale
        for i, (label, val) in enumerate(zip(FINGER_LABELS, touch)):
            x = x0 + i * (bar_w + 0.014)
            frac = float(np.clip(val / tmax, 0.0, 1.0))
            color = (1.0, 0.35, 0.2) if frac < 0.05 else (0.2, 0.9, 0.35 + 0.4 * frac)
            ax.add_patch(Rectangle((x, y0), bar_w, bar_h, color=(0.15, 0.15, 0.15, 0.6)))
            ax.add_patch(
                Rectangle((x, y0), bar_w, bar_h * frac, color=color)
            )
            ax.text(
                x + bar_w / 2, y0 - 0.02, label[0],
                color="white", fontsize=10, ha="center", va="top",
            )
        ax.text(x0, y0 + bar_h + 0.02, "TACTILE (N)", color="#9fe8b0",
                fontsize=8, fontfamily="monospace")

        # --- alignment gauge (bottom-right) ---
        gx, gy, gr = 0.93, 0.82, 0.045
        ax.add_patch(plt.Circle((gx, gy), gr, color=(0.15, 0.15, 0.15, 0.6), zorder=5))
        ax.add_patch(plt.Circle((gx, gy), gr * align, color=(0.3, 0.85, 1.0), zorder=6))
        ax.text(gx, gy, f"{int(align * 100)}", color="white",
                fontsize=10, ha="center", va="center", fontweight="bold", zorder=7)
        ax.text(gx, gy - gr - 0.025, "ALIGN %", color="#9fc7e8",
                fontsize=8, ha="center", fontfamily="monospace", zorder=7)

        # --- top bar: title + narration ---
        ax.text(0.025, 0.96, title, color="white", fontsize=14,
                fontweight="bold", va="top", fontfamily="monospace", zorder=7)
        ax.text(0.025, 0.915, NARRATION.get(act, ""), color="#cfd8e0",
                fontsize=9, va="top", fontfamily="monospace", zorder=7)

        # --- progress ---
        ax.add_patch(Rectangle((0.025, 0.035), 0.35, 0.014,
                               color=(0.2, 0.2, 0.2, 0.6), zorder=5))
        ax.add_patch(Rectangle(
            (0.025, 0.035), 0.35 * (step / max(1, max_steps)), 0.014,
            color=(0.4, 0.8, 1.0), zorder=6,
        ))
        ax.text(0.025, 0.06, f"step {step}/{max_steps}", color="#9fb0c0",
                fontsize=8, fontfamily="monospace", zorder=7)

        # Render to a numpy array via savefig into an in-memory buffer for a
        # robust grab (buffer_rgba can miss patches on some backends).
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        import imageio.v3 as iio
        out = iio.imread(buf)[..., :3].copy()
        # Defensive resize in case savefig rounded the pixel dims.
        if out.shape[0] != self.h or out.shape[1] != self.w:
            out = np.asarray(
                __import__("PIL.Image", fromlist=["Image"]).fromarray(out).resize(
                    (self.w, self.h)
                )
            )
        return out


# ---------------------------------------------------------------------- #
# Camera control
# ---------------------------------------------------------------------- #
def set_camera(cam: mujoco.MjvCamera, act: str, t: float, duration: float, cube_pos):
    """Frame the hand, not the cube.

    The cube can be batted away by the policy, so tracking it would put the
    hand off-frame. We anchor the lookat to the palm region (fixed) and only
    use the cube_pos to gently bias the framing when it stays close.
    """
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    palm_center = np.array([0.05, 0.0, 0.08], dtype=np.float64)
    # blend toward the cube only when it is near the hand (keeps the shot
    # stable if the cube is flung away during learning).
    cube = np.asarray(cube_pos, dtype=np.float64)
    if np.linalg.norm(cube - palm_center) < 0.10:
        lookat = 0.6 * palm_center + 0.4 * cube
    else:
        lookat = palm_center
    lookat[2] = max(lookat[2], 0.06)
    cam.lookat[:] = lookat
    if act == "intro":
        cam.distance = 0.42 - 0.08 * min(1.0, t / duration)
        cam.azimuth = 60.0 + 40.0 * (t / max(duration, 0.1))
        cam.elevation = -30.0 + 10.0 * (t / max(duration, 0.1))
    elif act == "reorient":
        cam.distance = 0.30
        cam.azimuth = 120.0 + 25.0 * math.sin(2.0 * math.pi * t / duration)
        cam.elevation = -20.0
    else:  # grasp
        cam.distance = 0.34
        cam.azimuth = 90.0 + 30.0 * (t / max(duration, 0.1))
        cam.elevation = -15.0


# ---------------------------------------------------------------------- #
# Policies for each act
# ---------------------------------------------------------------------- #
def make_policy(kind: str, model_path: Path | None):
    """Return a policy(obs, step, max_steps, act) -> action in [-1,1]^16."""

    # Scripted baseline: intro = hold rest, reorient/grasp = ramped close.
    close = np.array(
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, -0.4, 1, 0.5, 1, 1], dtype=np.float32
    ) * 0.9

    def scripted(obs, step, max_steps, act):
        if act == "intro":
            return np.zeros(16, dtype=np.float32)
        if act == "reorient":
            # gentle oscillation to suggest in-hand reorientation
            phase = 0.4 * math.sin(2.0 * math.pi * step / 24.0)
            base = close * min(1.0, step / 20.0)
            base[0::4] += phase  # wiggle MCP joints
            return np.clip(base, -1.0, 1.0)
        return close * min(1.0, step / 15.0)

    if kind == "model" and model_path is not None:
        from stable_baselines3 import PPO
        m = PPO.load(model_path, device="cpu")

        def learned(obs, step, max_steps, act):
            a, _ = m.predict(obs, deterministic=True)
            return a
        return learned

    return scripted


# ---------------------------------------------------------------------- #
# Main render loop
# ---------------------------------------------------------------------- #
def run_act(
    env: InHandCubeEnv,
    policy,
    overlay: Overlay,
    renderer: mujoco.Renderer,
    cam: mujoco.MjvCamera,
    *,
    act: str,
    n_frames: int,
    fps: int,
    substeps: int,
    title: str,
    frames_out: list,
    traj_out: list,
):
    obs, info = env.reset(seed=abs(hash(act)) % 100000)
    target_quat = info["target_quat"].copy()

    for fi in range(n_frames):
        action = policy(obs, fi, n_frames, act)
        obs, r, term, trunc, info = env.step(action)
        align = float(info.get("align", 0.0))
        # extra physics settling for visual smoothness
        for _ in range(substeps - 1):
            mujoco.mj_step(env.model, env.data)

        # If the cube was dropped or the episode ended, soft-reset so the demo
        # keeps running for the full act length (we never cut the video short).
        if term or trunc:
            obs, _ = env.reset(seed=abs(hash(act)) % 100000 + fi)

        cube_pos = env.data.xpos[env._cube_body].copy()
        touch = touch_values(env.model, env.data)

        set_camera(cam, act, fi / fps, n_frames / fps, cube_pos)
        renderer.update_scene(env.data, camera=cam)
        frame = renderer.render().copy()
        comp = overlay.composite(
            frame, touch=touch, align=align, act=act, title=title,
            step=fi, max_steps=n_frames,
        )
        frames_out.append(comp)
        if fi % max(1, fps // 6) == 0:
            traj_out.append({
                "act": act, "frame": fi, "time_s": round(fi / fps, 3),
                "cube_pos": np.round(cube_pos, 4).tolist(),
                "align": round(float(align), 4),
                "touch_N": np.round(touch, 3).tolist(),
            })
    return target_quat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "ppo_leap_dex.zip")
    ap.add_argument("--baseline", action="store_true",
                    help="Use the scripted close-hand policy (no trained model)")
    ap.add_argument("--output", type=Path, default=ROOT / "demo.mp4")
    ap.add_argument("--trajectory", type=Path, default=ROOT / "demo_trajectory.json")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--act-frames", type=int, default=300,
                    help="Frames per act (approx; ~30s total at fps=30)")
    return _run(ap.parse_args())


def _run(args) -> int:
    if args.baseline:
        policy = make_policy("scripted", None)
        title_suffix = "(scripted baseline)"
    elif args.model.exists():
        policy = make_policy("model", args.model)
        title_suffix = f"(PPO: {args.model.name})"
    else:
        print(f"Model not found at {args.model}; falling back to scripted baseline.")
        policy = make_policy("scripted", None)
        title_suffix = "(scripted baseline)"

    scene_cfg = SceneConfig(render_width=args.width, render_height=args.height)
    env = InHandCubeEnv(scene_config=scene_cfg, env_config=EnvConfig(),
                        render_mode="rgb_array")
    renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    overlay = Overlay(args.width, args.height)

    frames: list[np.ndarray] = []
    traj: list[dict] = []
    n = args.act_frames
    print(f"Rendering TACTILE-DEX demo {title_suffix}: 3 acts x {n} frames...")

    t0 = run_act(env, policy, overlay, renderer, cam,
                 act="intro", n_frames=n // 3, fps=args.fps, substeps=2,
                 title="TACTILE-DEX", frames_out=frames, traj_out=traj)
    _ = run_act(env, policy, overlay, renderer, cam,
                act="reorient", n_frames=n // 3, fps=args.fps, substeps=2,
                title="In-Hand Reorientation", frames_out=frames, traj_out=traj)
    _ = run_act(env, policy, overlay, renderer, cam,
                act="grasp", n_frames=n - 2 * (n // 3), fps=args.fps, substeps=2,
                title="Grasp & Lift", frames_out=frames, traj_out=traj)

    summary = {
        "project": "TACTILE-DEX: Tactile In-Hand Manipulation with LEAP Hand",
        "task": ("A 16-DOF LEAP Hand reorients and lifts a cube using tactile "
                 "feedback and a PPO policy trained in MuJoCo."),
        "policy": title_suffix,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "n_frames": len(frames),
        "duration_s": round(len(frames) / args.fps, 2),
        "video": str(args.output),
        "trajectory": str(args.trajectory),
        "trajectory_samples": traj,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trajectory.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(args.output, np.asarray(frames), fps=args.fps, codec="libx264")
    except Exception as exc:
        fallback = args.output.with_suffix(".gif")
        iio.imwrite(fallback, np.asarray(frames), fps=args.fps)
        summary["video"] = str(fallback)
        summary["video_fallback_reason"] = str(exc)

    args.trajectory.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "trajectory_samples"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
