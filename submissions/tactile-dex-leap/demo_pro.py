#!/usr/bin/env python
"""Render the TACTILE-DEX PRO multi-task showcase video.

Walks through all 6 scripted benchmark tasks, one act each, with a live HUD:
task name, object type, tactile bars, a ✓/✗ success indicator, and a
picture-in-picture top-down camera. This is the headline demo video.

Usage
-----
    python demo_pro.py                       # all 6 tasks -> demo.mp4
    python demo_pro.py --tasks grasp_lift pinch bottle
    python demo_pro.py --output showcase.mp4 --frames-per-task 120
"""
from __future__ import annotations

import argparse
import io
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
    raise SystemExit(f"Missing dependency: {exc}\n  pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
from tactile_dex.env import EnvConfig, InHandCubeEnv  # noqa: E402
from tactile_dex.scene_builder import SceneConfig, touch_values  # noqa: E402
from tactile_dex.tasks import ALL_TASKS  # noqa: E402


def _composite(frame, *, task_title, obj, touch, success, step, max_steps, pip=None,
               w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame, extent=(0, 1, 0, 1), origin="upper", aspect="auto")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # title + object
    ax.text(0.025, 0.96, task_title, color="white", fontsize=15,
            fontweight="bold", va="top", fontfamily="monospace", zorder=7)
    ax.text(0.025, 0.915, f"object: {obj}", color="#9fc7e8", fontsize=10,
            va="top", fontfamily="monospace", zorder=7)

    # touch bars
    for i, (lab, val) in enumerate(zip(["I", "M", "R", "T"], touch)):
        x = 0.025 + i * 0.034
        frac = float(np.clip(val / 5.0, 0, 1))
        col = (1.0, 0.35, 0.2) if frac < 0.05 else (0.2, 0.9, 0.35 + 0.4 * frac)
        ax.add_patch(Rectangle((x, 0.20), 0.020, 0.18, color=(0.15, 0.15, 0.15, 0.6)))
        ax.add_patch(Rectangle((x, 0.20), 0.020, 0.18 * frac, color=col))
        ax.text(x + 0.010, 0.175, lab, color="white", fontsize=10,
                ha="center", va="top", zorder=7)
    ax.text(0.025, 0.40, "TACTILE (N)", color="#9fe8b0", fontsize=8,
            fontfamily="monospace", zorder=7)

    # success indicator (top-right): big checkmark / pending
    mark = "✓" if success else "…"
    col = "#4ade80" if success else "#fbbf24"
    ax.text(0.95, 0.92, mark, color=col, fontsize=42, ha="right", va="top",
            fontweight="bold", zorder=8)
    ax.text(0.95, 0.82, "SUCCESS" if success else "RUNNING", color=col,
            fontsize=9, ha="right", va="top", fontfamily="monospace", zorder=8)

    # progress
    ax.add_patch(Rectangle((0.025, 0.035), 0.35, 0.014, color=(0.2, 0.2, 0.2, 0.6), zorder=5))
    ax.add_patch(Rectangle((0.025, 0.035), 0.35 * (step / max(1, max_steps)), 0.014,
                           color=(0.4, 0.8, 1.0), zorder=6))
    ax.text(0.025, 0.06, f"step {step}/{max_steps}", color="#9fb0c0",
            fontsize=8, fontfamily="monospace", zorder=7)

    # PIP top-down inset
    if pip is not None:
        pw, ph = 0.22, 0.22 * (w / h)
        px0, py0 = 1.0 - pw - 0.015, 0.30
        ax.imshow(pip, extent=(px0, px0 + pw, py0, py0 + ph), aspect="auto", zorder=8)
        ax.add_patch(Rectangle((px0, py0), pw, ph, fill=False,
                               edgecolor="#cfe8ff", linewidth=1.5, zorder=9))
        ax.text(px0, py0 - 0.012, "TOP VIEW", color="#9fc7e8",
                fontsize=7, fontfamily="monospace", zorder=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    out = iio.imread(buf)[..., :3].copy()
    if out.shape[0] != h or out.shape[1] != w:
        from PIL import Image
        out = np.asarray(Image.fromarray(out).resize((w, h)))
    return out


def _set_main_cam(cam, cube_pos):
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    lookat = np.array([0.05, 0.0, 0.085], dtype=np.float64)
    lookat[2] = max(lookat[2], cube_pos[2] * 0.5 + 0.05)
    cam.lookat[:] = lookat
    cam.distance = 0.30
    cam.azimuth = 110.0
    cam.elevation = -22.0


def render_task(task_cls, frames_per_task, fps, w, h, main_renderer, pip_renderer):
    task = task_cls()
    env = InHandCubeEnv(scene_config=SceneConfig(object_type=task.object_type,
                                                 render_width=w, render_height=h),
                        env_config=EnvConfig(cube_pos_noise=0.0, cube_yaw_noise=0.0))
    env.reset(seed=abs(hash(task.name)) % 100000)
    task.setup(env)

    cam = mujoco.MjvCamera()
    pip_cam = mujoco.MjvCamera()
    pip_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    pip_cam.lookat = [0.05, 0.0, 0.10]
    pip_cam.distance = 0.18
    pip_cam.azimuth = 0.0
    pip_cam.elevation = -89.0

    frames = []
    for step in range(frames_per_task):
        # map render step -> task step (task uses 150 logical steps)
        logical = int(step / frames_per_task * 150)
        task.act(env, logical, 150)
        for _ in range(env.env_config.action_substeps):
            mujoco.mj_step(env.model, env.data)

        cube_pos = env.data.xpos[env._cube_body].copy()
        touch = touch_values(env.model, env.data)
        success = bool(task.success(env))

        _set_main_cam(cam, cube_pos)
        main_renderer.update_scene(env.data, camera=cam)
        frame = main_renderer.render().copy()
        pip_cam.lookat[2] = max(0.10, cube_pos[2])
        pip_renderer.update_scene(env.data, camera=pip_cam)
        pip = pip_renderer.render().copy()

        comp = _composite(frame, task_title=task.title, obj=task.object_type,
                          touch=touch, success=success, step=step,
                          max_steps=frames_per_task, pip=pip, w=w, h=h)
        frames.append(comp)
    return task.name, frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Subset of task names (default: all 6)")
    ap.add_argument("--output", type=Path, default=ROOT / "demo.mp4")
    ap.add_argument("--frames-per-task", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    args = ap.parse_args()

    chosen = ALL_TASKS
    if args.tasks:
        names = {c.name: c for c in ALL_TASKS}
        chosen = [names[t] for t in args.tasks if t in names]

    # main + PIP renderers share one model size; we rebuild per task but reuse.
    scout = InHandCubeEnv(scene_config=SceneConfig(render_width=args.width,
                                                   render_height=args.height))
    main_renderer = mujoco.Renderer(scout.model, width=args.width, height=args.height)
    pip_renderer = mujoco.Renderer(scout.model, width=args.width // 4, height=args.height // 4)

    all_frames = []
    print(f"Rendering TACTILE-DEX PRO: {len(chosen)} tasks x {args.frames_per_task} frames...")
    for TaskCls in chosen:
        name, frames = render_task(TaskCls, args.frames_per_task, args.fps,
                                   args.width, args.height, main_renderer, pip_renderer)
        print(f"  {name}: {len(frames)} frames")
        all_frames.extend(frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(args.output, np.asarray(all_frames), fps=args.fps, codec="libx264")
    except Exception as exc:
        fb = args.output.with_suffix(".gif")
        iio.imwrite(fb, np.asarray(all_frames), fps=args.fps)
        print(f"MP4 failed ({exc}); wrote {fb}")
    print(f"\nSaved {args.output}  ({len(all_frames)} frames, "
          f"{len(all_frames)/args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
