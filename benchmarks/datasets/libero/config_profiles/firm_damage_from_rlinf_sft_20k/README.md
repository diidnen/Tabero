# Firm damage thresholds derived from RLinf SFT 20k evaluation

This profile is for `libero_object` Firm Tasks `0,1,2,3,5,6,7,8,9`.

Source report:

```text
Record/experiments/2026-08-06_23-29-10-task0-3_5-9-firm-rlinf_sft_2gpu_mb16-20k.md
```

Derivation rule:

```text
physics.objects.<target>.damage.max_squeeze_force
    = task-level squeeze max meas from the source report
```

All damage terminals use `consecutive_frames: 4`. Task 5 additionally keeps
the existing tomato-sauce mass randomization `Uniform[0.08, 0.12] kg` applied
on every reset. Task 4 is not part of the Firm evaluation subset and has no
damage override in this profile.

Important evidence boundary: `squeeze max meas` is the mean across successful
episodes of each episode's Top-5%-sample mean over non-zero measured squeeze.
It is not a single-frame peak and is not a material-damage calibration.

Use with:

```bash
export LIBERO_CONFIG_DIR=/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/config_profiles/firm_damage_from_rlinf_sft_20k
```
