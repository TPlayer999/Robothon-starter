"""Gymnasium environment for in-hand cube reorientation with the LEAP Hand.

Task
----
A 16-DOF LEAP Hand, palm up, must reorient a free-floating cube to a random
target quaternion while keeping it grasped (not dropped). The agent commands
position targets for the 16 finger actuators; physics does the rest.

Reward (per step)
-----------------
A multi-term shaping signal that is dense enough for PPO to make progress on
a CPU budget:

* ``align``  — bonus proportional to how close the cube orientation is to the
  target quaternion (uses the log-quaternion alignment, in [0, 1]).
* ``grasp``  — bonus when >=2 fingertips register touch (encourages a real
  multi-contact grasp rather than balancing the cube on one finger).
* ``alive``  — small constant while the cube is held above the drop height.
* ``ctrl``   — small L2 penalty on the action delta (smoothness).
* ``drop``   — large negative terminal reward if the cube falls.

Shaping aid
-----------
Optionally a soft weld between cube and palm ("grasp assist") is active for the
first ``weld_steps`` of the episode and then disabled, so the policy bootstraps
a stable grasp before having to hold the cube by contact alone. This mirrors
common curriculum tricks in dexterous-manipulation RL and is toggleable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .scene_builder import (
    FINGER_NAMES,
    SceneConfig,
    build_scene,
    set_grasp_weld,
    touch_values,
)


@dataclass
class EnvConfig:
    """Reward weights and episode constants (mirror defaults.yaml)."""

    # Episode.
    max_episode_steps: int = 120
    action_substeps: int = 4  # physics steps per env step (dt=0.002 -> 8ms)

    # Reward weights.
    w_align: float = 1.0       # quadratic alignment (align**2 for steeper gradient)
    w_grasp: float = 0.4       # contact count bonus (saturates at 4 fingers)
    w_alive: float = 0.02
    w_ctrl: float = 0.005      # action-delta smoothness
    w_drop: float = -2.0       # terminal drop penalty (now actually applied)
    w_reach_align: float = 5.0  # terminal bonus when target reached
    w_pbrs: float = 0.5        # potential-based shaping (fingertip-to-cube distance)
    w_height: float = 0.3      # keep cube near a target height (height stability)
    w_angvel: float = 0.01     # penalize flinging the cube (cube angular velocity)
    w_force: float = 0.05      # reward total contact force magnitude (not just count)

    # Thresholds.
    touch_threshold: float = 0.5  # Newtons; below this a fingertip is "off"
    drop_z: float = 0.02  # cube centre below this = dropped
    target_z: float = 0.085  # nominal held cube height (for height-stability reward)
    align_success: float = 0.95  # alignment (in [0,1]) considered "solved"

    # Grasp-assist curriculum.
    use_weld_curriculum: bool = True
    weld_steps: int = 40  # first N steps of each episode keep the weld on

    # Domain randomization (sim2real relevance + robustness).
    friction_noise: float = 0.0   # +/- fraction applied to cube/tip friction each reset
    mass_noise: float = 0.0       # +/- fraction applied to cube mass each reset
    touch_noise: float = 0.0      # Gaussian std (N) added to touch readings

    # Randomisation on reset.
    cube_pos_noise: float = 0.012
    cube_yaw_noise: float = 0.6  # rad
    target_is_random: bool = True


def _quat_align(q_curr: np.ndarray, q_target: np.ndarray) -> float:
    """Alignment in [0, 1]: 1 = same orientation, 0 = maximally different.

    Uses ``|q_curr . q_target|`` (handles double-cover) mapped through a
    smooth shaping so the reward gradient is informative near the target.
    """
    dot = abs(float(np.dot(q_curr, q_target)))
    dot = min(1.0, max(0.0, dot))
    # cos(theta/2) = |dot|; theta in [0, pi]. Shape to [0,1] with theta/pi.
    theta = 2.0 * math.acos(dot)
    return max(0.0, 1.0 - theta / math.pi)


class InHandCubeEnv(gym.Env):
    """LEAP Hand in-hand cube reorientation."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_config: SceneConfig | None = None,
        env_config: EnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.scene_config = scene_config or SceneConfig()
        self.env_config = env_config or EnvConfig()
        self.render_mode = render_mode

        self.model, self.info = build_scene(self.scene_config)
        self.data = mujoco.MjData(self.model)

        # Precompute frequently-used ids.
        self._act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
             for n in self.info["actuator_names"]],
            dtype=np.int32,
        )
        # The 16 finger actuators (everything except the palm-lift actuator).
        self._finger_act_ids = self._act_ids[
            [i for i, n in enumerate(self.info["actuator_names"]) if n != "palm_lift_act"]
        ]
        self._lift_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "palm_lift_act"
        )
        self._has_palm_lift = self._lift_act_id >= 0
        self._joint_qadrs = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in self.info["joint_names"]],
            dtype=np.int32,
        )
        self._joint_dadrs = np.array(
            [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in self.info["joint_names"]],
            dtype=np.int32,
        )
        self._cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_freejoint_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        ]
        self._cube_dofadr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        ]
        self._eq_weld = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        self._fingertip_bodies = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("if_ds", "mf_ds", "rf_ds", "th_ds")
        ]
        # nominal cube mass/friction for domain randomization (re-applied per reset).
        self._nominal_cube_mass = float(self.model.body_mass[self._cube_body])
        self._nominal_cube_friction = np.asarray(
            self.model.geom_friction[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
            ]
        ).copy()

        # Actuator ctrl ranges -> action space in [-1, 1] (rescaled before apply).
        ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self._ctrl_low = ctrl_low
        self._ctrl_high = ctrl_high
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        # Observation (rich, 57-dim):
        #   joint pos(16) + joint vel(16) + cube pos(3) + cube quat(4)
        #   + relative quat err q_target⊗q_cube⁻¹ (4) + cube linvel(3)
        #   + cube angvel(3) + touch(4) + fingertip-to-cube dist(4) = 57
        obs_dim = 16 + 16 + 3 + 4 + 4 + 3 + 3 + 4 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._step_count = 0
        self._last_action = np.zeros(16, dtype=np.float32)
        self._last_potential = 0.0  # for potential-based reward shaping (PBRS)
        self._renderer = None

        # Gymnasium boilerplate.
        self.spec = None  # set by gym.register if used

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _joint_qpos(self) -> np.ndarray:
        return self.data.qpos[self._joint_qadrs]

    def _joint_qvel(self) -> np.ndarray:
        return self.data.qvel[self._joint_dadrs]

    def _cube_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self._cube_body].copy()
        quat = self.data.xquat[self._cube_body].copy()
        return pos, quat

    def _set_cube_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        a = self._cube_freejoint_qadr
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = quat
        self.data.qvel[a:a + 6] = 0.0

    def _set_weld(self, active: bool) -> None:
        if self._eq_weld >= 0:
            set_grasp_weld(self.model, self.data, active)

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        # action in [-1,1] -> ctrl range
        return (
            self._ctrl_low + (action.astype(np.float32) + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)
        )

    def _apply_domain_randomization(self) -> None:
        """Randomize cube friction/mass each reset if configured (sim2real)."""
        ec = self.env_config
        if ec.friction_noise > 0.0 or ec.mass_noise > 0.0:
            rng = self.np_random
            cube_g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
            if ec.mass_noise > 0.0:
                m = self._nominal_cube_mass * (1.0 + rng.uniform(-ec.mass_noise, ec.mass_noise))
                self.model.body_mass[self._cube_body] = m
            if ec.friction_noise > 0.0:
                f = self._nominal_cube_friction * (1.0 + rng.uniform(-ec.friction_noise, ec.friction_noise, size=3))
                self.model.geom_friction[cube_g] = f

    def _cube_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Cube linear (world) and angular (world) velocity from the free joint."""
        a = self._cube_dofadr
        linvel = self.data.qvel[a:a + 3].copy()
        angvel = self.data.qvel[a + 3:a + 6].copy()
        return linvel, angvel

    def _fingertip_to_cube_dist(self) -> np.ndarray:
        """Euclidean distance from each fingertip body origin to the cube centre."""
        cube_pos = self.data.xpos[self._cube_body]
        return np.array(
            [np.linalg.norm(self.data.xpos[bid] - cube_pos) for bid in self._fingertip_bodies],
            dtype=np.float32,
        )

    def _potential(self) -> float:
        """Potential function for PBRS: high when fingers are close to the cube
        and the cube is held near the target height."""
        ec = self.env_config
        mean_dist = float(np.mean(self._fingertip_to_cube_dist()))
        cube_pos = self.data.xpos[self._cube_body]
        height_err = abs(float(cube_pos[2]) - ec.target_z)
        return -(mean_dist + 0.5 * height_err)

    def _build_obs(self) -> np.ndarray:
        pos, quat = self._cube_pose()
        linvel, angvel = self._cube_velocity()
        touch = touch_values(self.model, self.data).astype(np.float32)
        if self.env_config.touch_noise > 0.0:
            touch = touch + self.np_random.normal(0.0, self.env_config.touch_noise, size=4).astype(np.float32)
        # relative quaternion error q_err = q_target ⊗ q_cube⁻¹ (canonical form)
        q_err = _quat_mul(self._target_quat.astype(np.float64), _quat_conj(quat.astype(np.float64)))
        if q_err[0] < 0:
            q_err = -q_err  # canonical positive-w
        ftip_dist = self._fingertip_to_cube_dist()
        obs = np.concatenate(
            [
                self._joint_qpos().astype(np.float32),
                self._joint_qvel().astype(np.float32),
                pos.astype(np.float32),
                quat.astype(np.float32),
                q_err.astype(np.float32),
                linvel.astype(np.float32),
                angvel.astype(np.float32),
                touch,
                ftip_dist,
            ]
        )
        return obs

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        ec = self.env_config
        rng = self.np_random

        # Rest pose: fingers slightly curled so they already cage the cube.
        rest = {
            "if_mcp": 0.6, "if_pip": 0.5, "if_dip": 0.4, "if_rot": 0.2,
            "mf_mcp": 0.6, "mf_pip": 0.5, "mf_dip": 0.4, "mf_rot": 0.0,
            "rf_mcp": 0.6, "rf_pip": 0.5, "rf_dip": 0.4, "rf_rot": -0.2,
            "th_cmc": 0.6, "th_axl": 0.3, "th_mcp": 0.4, "th_ipl": 0.4,
        }
        for jn, adr in zip(self.info["joint_names"], self._joint_qadrs):
            if jn not in rest:
                continue  # skip palm_lift (set via its own actuator)
            self.data.qpos[adr] = rest[jn]
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn + "_act")
            self.data.ctrl[aid] = rest[jn]
        # zero the palm-lift ctrl (hand starts at rest height)
        if self._has_palm_lift:
            self.data.ctrl[self._lift_act_id] = 0.0

        # Spawn the cube with light noise around the nominal cradle pose.
        spawn = np.array(self.scene_config.cube_spawn_pos, dtype=np.float64)
        noise = rng.uniform(-ec.cube_pos_noise, ec.cube_pos_noise, size=3)
        noise[2] *= 0.5  # less vertical jitter
        yaw = rng.uniform(-ec.cube_yaw_noise, ec.cube_yaw_noise)
        self._set_cube_pose(spawn + noise, _yaw_quat(yaw))

        # Random target orientation (one of a few axis-aligned flips + noise).
        if ec.target_is_random:
            self._target_quat = _random_target_quat(rng)
        else:
            self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Grasp-assist curriculum: weld on for the first few steps.
        self._apply_domain_randomization()
        self._set_weld(ec.use_weld_curriculum)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._last_action = np.zeros(16, dtype=np.float32)
        self._last_potential = self._potential()  # init PBRS baseline
        return self._build_obs(), {"target_quat": self._target_quat.copy()}

    def step(self, action: np.ndarray):
        ec = self.env_config
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        ctrl = self._action_to_ctrl(action)
        self.data.ctrl[self._act_ids] = ctrl

        # Disable weld once the curriculum window elapses.
        if ec.use_weld_curriculum and self._step_count >= ec.weld_steps:
            self._set_weld(False)

        for _ in range(ec.action_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # ---- reward (dense, multi-term) ----
        cube_pos, cube_quat = self._cube_pose()
        align = _quat_align(cube_quat, self._target_quat)
        touch = touch_values(self.model, self.data)
        n_touching = int(np.sum(touch > ec.touch_threshold))
        total_force = float(np.sum(np.clip(touch, 0.0, None)))
        linvel, angvel = self._cube_velocity()

        # Quadratic alignment: steeper gradient near the target (the hard part).
        align_bonus = ec.w_align * (align ** 2)
        # Grasp: reward scales up to 4 contacting fingers (cages prevent drops).
        grasp_bonus = ec.w_grasp * min(1.0, n_touching / 4.0)
        force_bonus = ec.w_force * min(1.0, total_force / 4.0)
        alive_bonus = ec.w_alive
        ctrl_cost = ec.w_ctrl * float(np.sum((action - self._last_action) ** 2))
        # Penalize flinging: cube angular velocity magnitude.
        angvel_cost = ec.w_angvel * float(np.linalg.norm(angvel))
        # Height stability: reward keeping the cube near the held target height.
        height_bonus = ec.w_height * max(0.0, 1.0 - abs(float(cube_pos[2]) - ec.target_z) / 0.05)
        # Potential-based reward shaping (PBRS) — keeps optimality, makes progress visible.
        new_potential = self._potential()
        pbrs = ec.w_pbrs * (new_potential - self._last_potential)
        self._last_potential = new_potential

        reward = (align_bonus + grasp_bonus + force_bonus + alive_bonus
                  + height_bonus + pbrs - ctrl_cost - angvel_cost)
        self._last_action = action.copy()

        # ---- termination ----
        dropped = bool(cube_pos[2] < ec.drop_z)
        solved = bool(align >= ec.align_success)
        terminated = dropped
        if dropped:
            reward += ec.w_drop  # FIX dead code: now the drop penalty actually applies.
        if solved:
            reward += ec.w_reach_align
            terminated = True
        truncated = self._step_count >= ec.max_episode_steps

        info = {
            "align": float(align),
            "n_touching": n_touching,
            "touch": touch.tolist(),
            "cube_z": float(cube_pos[2]),
            "dropped": dropped,
            "solved": solved,
        }
        return self._build_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, width=640, height=480
            )
            self._camera = mujoco.MjvCamera()
            self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._camera.lookat = [0.05, 0.0, 0.08]
            self._camera.distance = 0.20
            self._camera.azimuth = 120
            self._camera.elevation = -25
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render().copy()


# ---------------------------------------------------------------------- #
# Quaternion helpers
# ---------------------------------------------------------------------- #
def _yaw_quat(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (== inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b (w,x,y,z convention)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _random_target_quat(rng: np.random.Generator) -> np.ndarray:
    """A target that is a 90-degree flip about a random axis (reachable)."""
    axes = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    axis = axes[rng.integers(0, 3)]
    angle = rng.choice([math.pi / 2, math.pi, -math.pi / 2])
    half = angle * 0.5
    q = np.array([math.cos(half), *(axis * math.sin(half))])
    # small noise so the target isn't always a perfect axis flip
    q = q + rng.normal(0.0, 0.02, size=4)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q
