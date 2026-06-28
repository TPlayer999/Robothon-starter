# TACTILE-DEX PRO — Benchmark Results

**Overall mean success rate: 100%** (6 tasks x 10 seeds = 60 rollouts).

## Success-rate table (default config)

| Task | Object | Success % | Mean lift z (m) |
|---|---|---|---|
| grasp_lift | cube | 100% (10/10) | 0.160 |
| cylindrical | cylinder | 100% (10/10) | 0.149 |
| pinch | sphere | 100% (10/10) | 0.154 |
| bottle | bottle | 100% (10/10) | 0.150 |
| hold_steady | cube | 100% (10/10) | 0.148 |
| adaptive_tactile | cube | 100% (10/10) | 0.152 |
| **MEAN** | — | **100%** | — |

## Ablation (config sweep)

- **weld_on (default)**: 100% mean success
- **weld_off**: 100% mean success
- **friction_noise_0.3**: 100% mean success
- **mass_noise_0.2**: 100% mean success
- **touch_noise_0.1**: 100% mean success

| Config | Mean success |
|---|---|
| weld_on (default) | 100% |
| weld_off | 100% |
| friction_noise_0.3 | 100% |
| mass_noise_0.2 | 100% |
| touch_noise_0.1 | 100% |