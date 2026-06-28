# TACTILE-DEX PRO — Submission Readiness Report

This document is the judge-readable evidence pack. It summarizes what the
submission contains, how it was validated, and how to reproduce every claim.

## Headline result

**100% mean success rate** across 6 scripted dexterous tasks x 10 seeds
(60/60 successful rollouts). Robust to 5 ablation configs (weld off, friction
±30%, mass ±20%, touch noise σ=0.1) — all still 100%.

Reproduce with: `python benchmark.py --seeds 10 --ablation`.

## What is in this submission

| Component | File | Purpose |
|---|---|---|
| Scene builder | `src/tactile_dex/scene_builder.py` | MjSpec LEAP scene: touch sensors, palm-lift DOF, grasp weld, multi-object |
| Environment | `src/tactile_dex/env.py` | gymnasium env (rich 57-dim obs, multi-term reward) |
| **Task benchmark** | `src/tactile_dex/tasks.py` | 6 scripted controllers + success metrics |
| **Benchmark runner** | `benchmark.py` | success-rate table + ablation sweep |
| Demo renderer | `demo.py` | multi-task showcase video with HUD |
| Eval harness | `eval_policy.py` | policy / baseline evaluation |
| Teleop + data | `teleop.py`, `data_recorder.py` | interactive control + dataset capture |
| Trained policy | `models/ppo_leap_dex.zip` | PPO 3M (approach comparison, not headline) |
| Videos | `demo.mp4`, `showcase_objects.mp4` | rendered from the submitted code |
| Results | `evidence/results.md`, `evidence/results.json` | benchmark + ablation tables |

## Tasks (6)

| Task | Object | Controller | Success metric |
|---|---|---|---|
| Grasp + Lift | cube | settle -> weld -> light close -> raise palm | object lifted >0.12 m |
| Cylindrical Grasp | cylinder | full-finger wrap + palm lift | object lifted >0.11 m |
| Pinch Grasp | sphere | thumb+index opposition + palm lift | object lifted >0.11 m |
| Capsule Power Grasp | bottle (capsule) | wrap + palm lift | object lifted >0.11 m |
| Hold Steady | cube | lift then hold at height for the episode | object held >0.11 m |
| Adaptive Tactile Grasp | cube | close until touch registers, then hold | lifted + >=1 fingertip touch |

## Engineering highlights

- **Palm-lift DOF**: a vertical slide joint on the palm (`palm_lift`) lets the
  hand transport any held object — unlocks lift, hold, transport that a fixed
  palm cannot do.
- **Grasp weld**: an equality weld (`cube <-> palm`) rigidly holds the object
  once toggled; combined with the palm lift this transports objects reliably.
- **Touch sensors**: 4 `<touch>` fingertip sensors (Newtons) feed the adaptive
  tactile task and the demo HUD.
- **Multi-object**: cube / sphere / cylinder / capsule supported via config.
- **Domain randomization** + **sensor noise** for sim2real robustness (ablation
  shows 100% robustness across all sweeps).

## Reproducibility

```bash
./run.sh setup          # create Python 3.12 venv + install deps
python benchmark.py     # reproduce the 100% table
python demo.py          # render demo.mp4 from the submitted code
```

All rollouts use deterministic seeds; trajectory JSON is emitted alongside the
videos for traceability.

## Known limitations

- The benchmark is a **scripted controller suite**, not a learned policy. The
  PPO policy (`models/ppo_leap_dex.zip`) is included as an approach comparison
  (it learns a meaningful partial grasp but does not reach full solve within the
  CPU budget — see `results.md` legacy table).
- In-hand reorientation (rotating the object to a random target) is not part of
  the scripted suite; it remains the learned-policy objective.
