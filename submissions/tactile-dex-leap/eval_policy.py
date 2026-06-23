#!/usr/bin/env python
"""Evaluate a trained PPO policy (or a scripted baseline) on TACTILE-DEX.

Reports mean episode reward, alignment, grasp stability and solve/drop rates
across N rollouts. Used to pick the best checkpoint before rendering the demo.

Usage
-----
    python eval_policy.py --model models/ppo_leap_dex.zip --episodes 20
    python eval_policy.py --baseline grasp   # scripted close-hand baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tactile_dex.env import EnvConfig, InHandCubeEnv  # noqa: E402
from tactile_dex.scene_builder import SceneConfig  # noqa: E402


def scripted_close_hand(_obs, step, max_steps):
    """A simple scripted policy: ramp fingers toward a strong flex over time."""
    alpha = min(1.0, step / 30.0)
    target = np.array(
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, -0.5, 1, 0.5, 1, 1],
        dtype=np.float32,
    ) * 0.9
    return target * alpha


def run(env, policy_fn, episodes: int, deterministic: bool = True) -> dict:
    rewards, aligns, touches, solves, drops, lengths = [], [], [], 0, 0, []
    for ep in range(episodes):
        obs, info = env.reset(seed=1000 + ep)
        ep_r = 0.0
        step = 0
        done = False
        while not done:
            action = policy_fn(obs, step, env.env_config.max_episode_steps)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            step += 1
            done = term or trunc
        rewards.append(ep_r)
        aligns.append(info["align"])
        touches.append(info["n_touching"])
        lengths.append(step)
        if info["solved"]:
            solves += 1
        if info["dropped"]:
            drops += 1
    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_align": float(np.mean(aligns)),
        "max_align": float(np.max(aligns)),
        "mean_touch": float(np.mean(touches)),
        "solve_rate": solves / episodes,
        "drop_rate": drops / episodes,
        "mean_len": float(np.mean(lengths)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None, help="Trained PPO .zip")
    ap.add_argument("--baseline", choices=["grasp"], default=None,
                    help="Use a scripted baseline instead of a model")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--deterministic", action="store_true", default=True)
    args = ap.parse_args()

    env = InHandCubeEnv(scene_config=SceneConfig(), env_config=EnvConfig())

    if args.baseline:
        print(f"Evaluating scripted baseline '{args.baseline}' ...")
        policy_fn = scripted_close_hand
        results = run(env, policy_fn, args.episodes, args.deterministic)
    elif args.model:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        print(f"Loading policy {args.model} ...")
        model = PPO.load(args.model, device="cpu")
        # The trained model expects VecNormalize-normalized observations; wrap
        # the env the same way and load the saved running stats if present.
        vecnorm_path = args.model.parent / "ppo_leap_dex_vecnorm.pkl"
        vec_env = VecNormalize(DummyVecEnv([lambda: env]), training=False, norm_reward=False)
        if vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env.venv)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"  loaded VecNormalize stats from {vecnorm_path.name}")
        else:
            print("  WARNING: no VecNormalize stats found; obs may be mis-scaled.")

        def policy_fn(obs, step, max_steps):
            a, _ = model.predict(obs, deterministic=args.deterministic)
            return a

        # Custom rollout that normalizes obs before predict (VecNormalize wraps
        # the env; we drive it step-by-step through the wrapper).
        rewards, aligns, touches, solves, drops, lengths = [], [], [], 0, 0, []
        for ep in range(args.episodes):
            obs_raw, info = env.reset(seed=1000 + ep)
            obs = vec_env.normalize_obs(np.asarray(obs_raw)[None, :])[0]
            ep_reward = 0.0
            step = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                # action from VecNormalize-normalized policy is in normalized space already
                obs_raw, r, term, trunc, info = env.step(np.asarray(action))
                obs = vec_env.normalize_obs(np.asarray(obs_raw)[None, :])[0]
                ep_reward += r
                step += 1
                done = term or trunc
            rewards.append(ep_reward)
            aligns.append(info["align"])
            touches.append(info["n_touching"])
            lengths.append(step)
            if info["solved"]:
                solves += 1
            if info["dropped"]:
                drops += 1
        results = {
            "episodes": args.episodes,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_align": float(np.mean(aligns)),
            "max_align": float(np.max(aligns)),
            "mean_touch": float(np.mean(touches)),
            "solve_rate": solves / args.episodes,
            "drop_rate": drops / args.episodes,
            "mean_len": float(np.mean(lengths)),
        }
    else:
        print("Specify --model <zip> or --baseline grasp")
        return 1

    print("\n=== Evaluation ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:14s}: {v:.4f}")
        else:
            print(f"  {k:14s}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
