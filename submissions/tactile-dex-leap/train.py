#!/usr/bin/env python
"""Train a PPO policy for TACTILE-DEX in-hand cube reorientation.

Designed for Apple Silicon (M1/M2/M3): runs CPU-only via ``device="cpu"`` and
uses a small vectorised env so the ~2M-timestep budget finishes in a few hours.

Usage
-----
    python train.py [--config configs/default.yaml] [--timesteps N] [--n-envs N]

Checkpoints are written to ``<model_dir>/<model_name>_<steps>steps.zip`` and the
final model to ``<model_dir>/<model_name>.zip``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import LinearSchedule

# Make ``tactile_dex`` importable when running the script from the submission
# root without installing the package.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tactile_dex.env import EnvConfig, InHandCubeEnv  # noqa: E402
from tactile_dex.scene_builder import SceneConfig  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def env_config_from_yaml(cfg: dict) -> EnvConfig:
    e = cfg.get("env", {})
    r = cfg.get("reward", {})
    return EnvConfig(
        max_episode_steps=e.get("max_episode_steps", 120),
        action_substeps=e.get("action_substeps", 4),
        use_weld_curriculum=e.get("use_weld_curriculum", True),
        weld_steps=e.get("weld_steps", 40),
        target_is_random=e.get("target_is_random", True),
        w_align=r.get("w_align", 1.0),
        w_grasp=r.get("w_grasp", 0.4),
        w_alive=r.get("w_alive", 0.02),
        w_ctrl=r.get("w_ctrl", 0.005),
        w_drop=r.get("w_drop", -2.0),
        w_reach_align=r.get("w_reach_align", 5.0),
        w_pbrs=r.get("w_pbrs", 0.5),
        w_height=r.get("w_height", 0.3),
        w_angvel=r.get("w_angvel", 0.01),
        w_force=r.get("w_force", 0.05),
        touch_threshold=r.get("touch_threshold", 0.5),
        drop_z=r.get("drop_z", 0.02),
        align_success=r.get("align_success", 0.95),
        friction_noise=r.get("friction_noise", 0.0),
        mass_noise=r.get("mass_noise", 0.0),
        touch_noise=r.get("touch_noise", 0.0),
    )


def make_env(env_cfg: EnvConfig, rank: int, seed: int):
    def _init():
        env = InHandCubeEnv(scene_config=SceneConfig(), env_config=env_cfg)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


class MetricLogCallback(BaseCallback):
    """Logs real task metrics every N steps so training is never blind.

    Aggregates ``align / n_touching / solved / dropped`` from the per-env
    ``info`` dicts (always printed, unlike the old buffer-gated logger).
    """

    def __init__(self, log_interval: int = 8192, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_print = 0
        self._reset_accumulator()

    def _reset_accumulator(self) -> None:
        self._acc_align = []
        self._acc_touch = []
        self._n_solved = 0
        self._n_dropped = 0
        self._n_term = 0

    def _on_step(self) -> bool:
        # SB3 collects per-env terminal infos in self.locals["infos"].
        for info in self.locals.get("infos", []):
            if "align" not in info:
                continue
            self._acc_align.append(info["align"])
            self._acc_touch.append(info["n_touching"])
            self._n_term += 1
            if info.get("solved"):
                self._n_solved += 1
            if info.get("dropped"):
                self._n_dropped += 1

        if self.num_timesteps - self._last_print < self.log_interval:
            return True
        self._last_print = self.num_timesteps
        if self._acc_align:
            n = len(self._acc_align)
            mean_al = float(np.mean(self._acc_align))
            max_al = float(np.max(self._acc_align))
            mean_t = float(np.mean(self._acc_touch))
            solve = self._n_solved / n
            drop = self._n_dropped / n
            ep_r = (float(np.mean([e["r"] for e in self.model.ep_info_buffer]))
                    if self.model.ep_info_buffer else float("nan"))
            print(f"[{self.num_timesteps:>9d}] ep_r={ep_r:+.2f} align={mean_al:.3f} "
                  f"max={max_al:.3f} touch={mean_t:.2f} solve={solve:.0%} drop={drop:.0%}",
                  flush=True)
        else:
            print(f"[{self.num_timesteps:>9d}] (no episodes terminated yet)", flush=True)
        self._reset_accumulator()
        return True


class CurriculumCallback(BaseCallback):
    """Anneal the grasp-assist weld length across training.

    Early on the cube is held by the weld for most of the episode (easy: learn
    to rotate while glued). Over the first ``anneal_steps`` we shrink
    ``weld_steps`` from ``start`` to ``end`` so the policy must hold the cube by
    real contacts. Operates on the underlying envs through ``vec_env.envs``.
    """

    def __init__(self, start_steps: int, end_steps: int, anneal_steps: int,
                 verbose: int = 1):
        super().__init__(verbose)
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.anneal_steps = anneal_steps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(1, self.anneal_steps))
        cur = int(round(self.start_steps + (self.end_steps - self.start_steps) * frac))
        # VecNormalize wraps SubprocVecEnv; unwrap to reach the envs.
        venv = self.model.get_vec_normalize_env() or self.model.env
        try:
            envs = venv.envs
        except AttributeError:
            return True
        for e in envs:
            base = getattr(e, "env", e)
            base.env_config.weld_steps = cur
        return True


class TimestepCheckpointCallback(BaseCallback):
    """Save a checkpoint every ``every_total_steps`` *total* environment steps.

    SB3's ``CheckpointCallback.save_freq`` is counted per-rollout and per-env,
    which is easy to mis-tune and produced checkpoint spam here. This callback
    keys off the true global ``num_timesteps`` so we get exactly one file per
    ``every_total_steps`` window regardless of n_envs / n_steps.
    """

    def __init__(self, every_total_steps: int, save_dir: str, name_prefix: str,
                 verbose: int = 1):
        super().__init__(verbose)
        self.every_total_steps = every_total_steps
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        # Snap to the next checkpoint boundary at training start (see below) so
        # resuming from a checkpoint doesn't immediately fire a burst of saves.
        self._next_save: int | None = None

    def _on_training_start(self) -> None:
        import math as _math
        # First boundary strictly after the current step count.
        self._next_save = (
            _math.ceil(self.num_timesteps / self.every_total_steps) * self.every_total_steps
        )
        if self._next_save <= self.num_timesteps:
            self._next_save += self.every_total_steps

    def _on_step(self) -> bool:
        if self._next_save is None:
            self._on_training_start()
        if self.num_timesteps >= self._next_save:
            path = Path(self.save_dir) / f"{self.name_prefix}_{self.num_timesteps}steps.zip"
            self.model.save(path)
            if self.verbose:
                print(f"[checkpoint] {path.name} @ {self.num_timesteps} steps", flush=True)
            self._next_save += self.every_total_steps
        return True


def build_vec_env(cfg: dict, n_envs: int, seed: int) -> VecEnv:
    env_cfg = env_config_from_yaml(cfg)
    fns = [make_env(env_cfg, i, seed) for i in range(n_envs)]
    vec = SubprocVecEnv(fns)
    # VecNormalize: running mean/std on obs + reward. Critical when the obs
    # mixes rad, Newtons and quaternion components of very different scales.
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=cfg["ppo"]["gamma"])
    return vec


def _as_schedule(value):
    """Turn a scalar into a LinearSchedule decaying to 0; pass schedules through."""
    if isinstance(value, str) and value.startswith("linear"):
        start = float(value.split(":")[1])
        # LinearSchedule(start, end, end_fraction): value goes start -> end over
        # end_fraction of training. Decay to 0 at the very end.
        return LinearSchedule(start, 0.0, 1.0)
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", type=Path, default=None, help="Resume from a .zip checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    ppo_cfg = cfg["ppo"]

    n_envs = args.n_envs or train_cfg["n_envs"]
    total = args.timesteps or train_cfg["total_timesteps"]
    seed = args.seed if args.seed is not None else train_cfg["seed"]

    model_dir = ROOT / train_cfg["model_dir"]
    log_dir = ROOT / train_cfg["log_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_name = train_cfg["model_name"]

    print(f"TACTILE-DEX PPO training: {n_envs} envs, {total} timesteps, device={ppo_cfg['device']}")
    print(f"  model_dir = {model_dir}")
    print(f"  checkpoint every {train_cfg['checkpoint_every']} steps")

    vec_env = build_vec_env(cfg, n_envs, seed)

    policy_kwargs = ppo_cfg.get("policy_kwargs", {})
    if "net_arch" in policy_kwargs:
        policy_kwargs["net_arch"] = list(policy_kwargs["net_arch"])

    if args.resume is not None:
        print(f"  resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=ppo_cfg["device"])
    else:
        model = PPO(
            policy=ppo_cfg["policy"],
            env=vec_env,
            learning_rate=_as_schedule(ppo_cfg["learning_rate"]),
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg.get("n_epochs", 4),
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=_as_schedule(ppo_cfg["clip_range"]),
            ent_coef=ppo_cfg.get("ent_coef", 0.0),
            target_kl=ppo_cfg.get("target_kl", 0.03),
            use_sde=ppo_cfg.get("use_sde", False),
            sde_sample_freq=ppo_cfg.get("sde_sample_freq", 4),
            policy_kwargs=policy_kwargs,
            device=ppo_cfg["device"],
            seed=seed,
            verbose=0,
            tensorboard_log=str(log_dir),
        )

    checkpoint_cb = TimestepCheckpointCallback(
        every_total_steps=train_cfg["checkpoint_every"],
        save_dir=str(model_dir),
        name_prefix=model_name,
    )
    metric_cb = MetricLogCallback(log_interval=8192)
    curr_cfg = cfg.get("curriculum", {})
    curriculum_cb = CurriculumCallback(
        start_steps=curr_cfg.get("weld_start_steps", env_config_from_yaml(cfg).max_episode_steps),
        end_steps=curr_cfg.get("weld_end_steps", 0),
        anneal_steps=curr_cfg.get("weld_anneal_steps", 1_000_000),
    )

    try:
        model.learn(
            total_timesteps=total,
            callback=[checkpoint_cb, metric_cb, curriculum_cb],
            reset_num_timesteps=args.resume is None,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving partial model.")

    final_path = model_dir / f"{model_name}.zip"
    model.save(final_path)
    # Save the VecNormalize stats alongside the model (needed to reload).
    if hasattr(vec_env, "save"):
        vec_env.save(str(model_dir / f"{model_name}_vecnorm.pkl"))
    vec_env.close()
    print(f"Saved final model -> {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
