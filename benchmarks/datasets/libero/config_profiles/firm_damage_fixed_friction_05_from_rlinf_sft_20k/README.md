# Firm mass-friction damage threshold profile

This profile extends `firm_damage_from_rlinf_sft_20k` for Firm Tasks
`0,1,2,3,5,6,7,8,9`.

- Both Franka finger bodies use static/dynamic friction `0.5/0.5`.
- Task 0 samples object static friction in `[0.4, 0.8]`, dynamic friction in
  `[0.3, 0.6]`, and mass in `[0.8, 1.2] kg` on every reset.
- Other Firm targets use static/dynamic friction `0.5/0.5`.
- Task 5 retains reset-time mass randomization in `[0.08, 0.12] kg`.
- Every Firm target derives its damage threshold at reset as
  `mass * |gravity| / mean(gripper_static, object_static) * 1.1`.
- `consecutive_frames=4` remains unchanged.
- Task 4 remains unchanged and is not part of this Firm evaluation subset.

The directory name is retained so existing Tabero and RLinf configuration paths do
not change. The original report-derived manual thresholds remain documented in
`threshold_provenance.json` as superseded historical inputs.

Direct Tabero evaluation:

```bash
python Tabero/scripts/tools/run_task_evaluations.py \
  --config-path /data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/firm_damage_fixed_friction_05_from_rlinf_sft_20k \
  ...
```

RLinf uses the existing YAML-only handoff:

```yaml
env:
  train:
    init_params:
      libero_config_dir: /data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/firm_damage_fixed_friction_05_from_rlinf_sft_20k
  eval:
    init_params:
      libero_config_dir: /data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/firm_damage_fixed_friction_05_from_rlinf_sft_20k
```

No RLinf source-code change is required.
