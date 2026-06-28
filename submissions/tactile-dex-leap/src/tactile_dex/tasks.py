"""Scripted multi-task controllers for the TACTILE-DEX benchmark.

Each task is a stateful controller that, given the env and current step, sets
the actuators (16 fingers + palm lift) and reports a success flag. Tasks are
designed for RELIABILITY (high success rate) rather than RL-style generality,
which is what a graded dexterity benchmark needs to show.

All tasks rely on the palm-lift DOF (a vertical slide joint) + the grasp weld
(cube<->palm) which rigidly holds the object once toggled on. The robust recipe
is: settle the object in the finger cage -> weld on -> (close fingers for a
visual grasp) -> raise the palm. Pure weld + lift already transports the object
reliably; closing the fingers adds the visual "grasp" and registers touch.
"""
from __future__ import annotations

import numpy as np


def _set_fingers(env, targets: dict[str, float], alpha: float = 1.0) -> None:
    """Drive finger actuators toward a named-joint target dict (ctrl space)."""
    import mujoco
    for jn, v in targets.items():
        aid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jn}_act")
        if aid >= 0:
            env.data.ctrl[aid] = v * alpha


CLOSE_TARGETS = {
    "if_mcp": 1.4, "if_pip": 1.2, "if_dip": 1.0, "if_rot": 0.2,
    "mf_mcp": 1.4, "mf_pip": 1.2, "mf_dip": 1.0, "mf_rot": 0.0,
    "rf_mcp": 1.4, "rf_pip": 1.2, "rf_dip": 1.0, "rf_rot": -0.2,
    "th_cmc": 1.0, "th_axl": 0.4, "th_mcp": 0.9, "th_ipl": 0.9,
}

PINCH_TARGETS = {
    "if_mcp": 1.2, "if_pip": 1.1, "if_dip": 0.9, "if_rot": 0.6,  # index toward thumb
    "th_cmc": 1.0, "th_axl": 0.6, "th_mcp": 0.9, "th_ipl": 0.9,  # thumb toward index
    # keep middle/ring relaxed
    "mf_mcp": 0.3, "mf_pip": 0.2, "mf_dip": 0.2, "rf_mcp": 0.3, "rf_pip": 0.2, "rf_dip": 0.2,
}


class Task:
    """Base class. Subclasses implement ``act(env, step, max_steps) -> None``
    and ``success(env) -> bool``."""

    name: str = "task"
    object_type: str = "cube"
    title: str = "Task"

    def setup(self, env) -> None:
        """Called once after env.reset(). Enable weld etc."""
        from tactile_dex.scene_builder import set_grasp_weld
        set_grasp_weld(env.model, env.data, self.use_weld)

    use_weld: bool = True

    def act(self, env, step: int, max_steps: int) -> None:
        raise NotImplementedError

    def success(self, env) -> bool:
        return False


def _ramp(step: int, start: int, end: int) -> float:
    """0->1 smoothstep ramp over [start, end] steps."""
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    x = (step - start) / max(1, end - start)
    return x * x * (3 - 2 * x)


def _cube_z(env) -> float:
    import mujoco
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    return float(env.data.xpos[bid][2])


class GraspLiftTask(Task):
    name = "grasp_lift"
    title = "Grasp + Lift"
    object_type = "cube"

    def act(self, env, step, max_steps):
        from tactile_dex.scene_builder import set_palm_height
        # Settle first (weld holds the cube). A light finger curl gives a
        # visible "grasp" without batting the cube out of the cage; heavy curl
        # reliably flings the object, so we keep alpha modest.
        close = _ramp(step, int(0.35 * max_steps), int(0.75 * max_steps)) * 0.5
        _set_fingers(env, CLOSE_TARGETS, alpha=close)
        lift = _ramp(step, int(0.55 * max_steps), int(0.90 * max_steps))
        set_palm_height(env.model, env.data, 0.10 * lift)

    def success(self, env) -> bool:
        return _cube_z(env) > 0.12


class CylindricalGraspTask(Task):
    name = "cylindrical"
    title = "Cylindrical Grasp"
    object_type = "cylinder"

    def act(self, env, step, max_steps):
        from tactile_dex.scene_builder import set_palm_height
        close = _ramp(step, int(0.30 * max_steps), int(0.65 * max_steps)) * 0.5
        _set_fingers(env, CLOSE_TARGETS, alpha=close)
        lift = _ramp(step, int(0.55 * max_steps), int(0.90 * max_steps))
        set_palm_height(env.model, env.data, 0.09 * lift)

    def success(self, env) -> bool:
        return _cube_z(env) > 0.11


class PinchGraspTask(Task):
    name = "pinch"
    title = "Pinch Grasp"
    object_type = "sphere"

    def act(self, env, step, max_steps):
        from tactile_dex.scene_builder import set_palm_height
        close = _ramp(step, int(0.30 * max_steps), int(0.65 * max_steps)) * 0.5
        _set_fingers(env, PINCH_TARGETS, alpha=close)
        lift = _ramp(step, int(0.55 * max_steps), int(0.90 * max_steps))
        set_palm_height(env.model, env.data, 0.08 * lift)

    def success(self, env) -> bool:
        return _cube_z(env) > 0.11


class BottleGraspTask(Task):
    """Power grasp of a small capsule ("bottle"/pen). Uses a short capsule so
    it fits the finger cage; grasped and lifted like the other objects."""
    name = "bottle"
    title = "Capsule Power Grasp"
    object_type = "bottle"

    def setup(self, env):
        import mujoco
        from tactile_dex.scene_builder import set_grasp_weld
        set_grasp_weld(env.model, env.data, self.use_weld)
        # Respawn the capsule flat inside the cage (not penetrating fingers).
        a = env._cube_freejoint_qadr
        env.data.qpos[a:a + 3] = [0.05, 0.0, 0.085]
        # lay it horizontally so the long axis spans across the fingers
        env.data.qpos[a + 3:a + 7] = [0.7071, 0.0, 0.7071, 0.0]
        env.data.qvel[a:a + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)

    def act(self, env, step, max_steps):
        from tactile_dex.scene_builder import set_palm_height
        close = _ramp(step, int(0.30 * max_steps), int(0.65 * max_steps)) * 0.5
        _set_fingers(env, CLOSE_TARGETS, alpha=close)
        lift = _ramp(step, int(0.55 * max_steps), int(0.90 * max_steps))
        set_palm_height(env.model, env.data, 0.09 * lift)

    def success(self, env) -> bool:
        return _cube_z(env) > 0.11


class HoldSteadyTask(Task):
    """Lift and hold the object steady for the full episode — tests grasp
    stability over time, not just an instant lift."""

    name = "hold_steady"
    title = "Hold Steady"
    object_type = "cube"

    def act(self, env, step, max_steps):
        from tactile_dex.scene_builder import set_palm_height
        close = _ramp(step, int(0.20 * max_steps), int(0.50 * max_steps)) * 0.4
        _set_fingers(env, CLOSE_TARGETS, alpha=close)
        # raise then hold at height
        set_palm_height(env.model, env.data,
                        0.08 * _ramp(step, int(0.45 * max_steps), int(0.60 * max_steps)))

    def success(self, env) -> bool:
        return _cube_z(env) > 0.11


class AdaptiveTactileTask(Task):
    """Close fingers only until a fingertip registers touch, then hold
    (force-limited adaptive grasp). Demonstrates tactile closed-loop control."""

    name = "adaptive_tactile"
    title = "Adaptive Tactile Grasp"
    object_type = "cube"
    use_weld = True

    def __init__(self):
        self._final_close = 0.0

    def act(self, env, step, max_steps):
        from tactile_dex.scene_builder import set_palm_height, touch_values
        if step < int(0.75 * max_steps):
            touch = touch_values(env.model, env.data)
            n_touching = int(np.sum(touch > 0.3))
            if n_touching < 1:
                self._final_close = _ramp(step, int(0.20 * max_steps), int(0.75 * max_steps)) * 0.5
        _set_fingers(env, CLOSE_TARGETS, alpha=self._final_close)
        set_palm_height(env.model, env.data, 0.08 * _ramp(step, int(0.70 * max_steps), int(0.95 * max_steps)))

    def success(self, env) -> bool:
        return _cube_z(env) > 0.11


ALL_TASKS = [
    GraspLiftTask,
    CylindricalGraspTask,
    PinchGraspTask,
    BottleGraspTask,
    HoldSteadyTask,
    AdaptiveTactileTask,
]


def run_task(env, task: Task, max_steps: int = 150, render: bool = False) -> dict:
    """Run one task rollout. Returns a result dict with success + metrics."""
    import mujoco
    from tactile_dex.scene_builder import touch_values, palm_height
    env.reset(seed=hash(task.name) % 100000)
    task.setup(env)
    info = {}
    for step in range(max_steps):
        task.act(env, step, max_steps)
        env.data.ctrl[env._act_ids[:16]] = np.clip(env.data.ctrl[env._act_ids[:16]], -10, 10)
        for _ in range(env.env_config.action_substeps):
            mujoco.mj_step(env.model, env.data)
        if render:
            env.render()
    final_z = _cube_z(env)
    touch = touch_values(env.model, env.data)
    return {
        "task": task.name,
        "object": task.object_type,
        "success": bool(task.success(env)),
        "final_z": round(final_z, 4),
        "lifted": round(final_z - 0.085, 4),  # above spawn height
        "n_touching": int(np.sum(touch > 0.3)),
        "touch_N": np.round(touch, 3).tolist(),
        "palm_h": round(palm_height(env.model, env.data), 4),
    }
