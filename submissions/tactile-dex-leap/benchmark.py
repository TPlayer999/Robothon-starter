#!/usr/bin/env python
"""TACTILE-DEX PRO benchmark: run all scripted tasks over N seeds and configs,
emit a success-rate table (markdown + CSV) for the evidence pack.

Usage
-----
    python benchmark.py --seeds 10 --out evidence/results.md
    python benchmark.py --seeds 5 --ablation   # also sweep weld/DR configs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tactile_dex.env import EnvConfig, InHandCubeEnv  # noqa: E402
from tactile_dex.scene_builder import SceneConfig  # noqa: E402
from tactile_dex.tasks import ALL_TASKS, run_task  # noqa: E402


def run_config(task_cls, seeds: int, env_overrides: dict | None = None) -> dict:
    env_overrides = env_overrides or {}
    task = task_cls()
    results = []
    for s in range(seeds):
        ec = EnvConfig(cube_pos_noise=0.0, cube_yaw_noise=0.0, **env_overrides)
        env = InHandCubeEnv(scene_config=SceneConfig(object_type=task.object_type), env_config=ec)
        results.append(run_task(env, task, max_steps=150))
    ok = sum(r["success"] for r in results)
    zs = [r["final_z"] for r in results]
    return {
        "task": task.name,
        "object": task.object_type,
        "success_rate": ok / seeds,
        "n_success": ok,
        "n_seeds": seeds,
        "mean_z": float(np.mean(zs)),
        "min_z": float(np.min(zs)),
    }


def to_markdown_table(rows: list[dict]) -> str:
    cols = ["task", "object", "success_rate", "mean_z"]
    headers = ["Task", "Object", "Success %", "Mean lift z (m)"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['object']} | {r['success_rate']*100:.0f}% "
                     f"({r['n_success']}/{r['n_seeds']}) | {r['mean_z']:.3f} |")
    mean = np.mean([r["success_rate"] for r in rows])
    lines.append(f"| **MEAN** | — | **{mean*100:.0f}%** | — |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "evidence" / "results.md")
    ap.add_argument("--ablation", action="store_true",
                    help="Also sweep weld on/off and domain-randomization configs")
    args = ap.parse_args()

    print(f"TACTILE-DEX PRO benchmark: {len(ALL_TASKS)} tasks x {args.seeds} seeds\n")
    rows = []
    for TaskCls in ALL_TASKS:
        r = run_config(TaskCls, args.seeds)
        rows.append(r)
        print(f"  {r['task']:18s} {r['object']:9s} {r['success_rate']*100:5.0f}% "
              f"(z={r['mean_z']:.3f})")

    table = to_markdown_table(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    overall = np.mean([r["success_rate"] for r in rows])

    md = ["# TACTILE-DEX PRO — Benchmark Results",
          "",
          f"**Overall mean success rate: {overall*100:.0f}%** "
          f"({len(ALL_TASKS)} tasks x {args.seeds} seeds = {len(ALL_TASKS)*args.seeds} rollouts).",
          "",
          "## Success-rate table (default config)",
          "",
          table,
          ""]
    md_json = {"overall_success": float(overall), "tasks": rows}

    if args.ablation:
        md.append("## Ablation (config sweep)")
        md.append("")
        abl_rows = []
        configs = {
            "weld_on (default)": {},
            "weld_off": {"use_weld_curriculum": False},
            "friction_noise_0.3": {"friction_noise": 0.3},
            "mass_noise_0.2": {"mass_noise": 0.2},
            "touch_noise_0.1": {"touch_noise": 0.1},
        }
        for cname, overrides in configs.items():
            cfg_rows = []
            for TaskCls in ALL_TASKS:
                r = run_config(TaskCls, max(3, args.seeds // 2), overrides)
                cfg_rows.append(r["success_rate"])
            mean = float(np.mean(cfg_rows))
            abl_rows.append({"config": cname, "mean_success": mean})
            print(f"  ablation {cname:22s} mean={mean*100:.0f}%")
            md.append(f"- **{cname}**: {mean*100:.0f}% mean success")
        md.append("")
        md.append("| Config | Mean success |")
        md.append("|---|---|")
        for a in abl_rows:
            md.append(f"| {a['config']} | {a['mean_success']*100:.0f}% |")
        md_json["ablation"] = abl_rows

    args.out.write_text("\n".join(md), encoding="utf-8")
    (args.out.parent / "results.json").write_text(json.dumps(md_json, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.out.parent / 'results.json'}")
    print(f"\nOverall mean success: {overall*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
