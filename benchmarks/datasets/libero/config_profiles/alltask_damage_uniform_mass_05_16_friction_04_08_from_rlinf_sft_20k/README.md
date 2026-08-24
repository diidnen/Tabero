# All-task uniform mass/friction damage profile

This profile is copied from
`firm_damage_fixed_friction_05_from_rlinf_sft_20k` and applies the same
reset-time domain randomization to every `libero_object` Task 0--9 target.

- Target mass: uniform `[0.5, 1.6] kg` on every reset.
- Target static friction: uniform `[0.4, 0.8]` on every reset.
- Target dynamic friction: uniform `[0.3, 0.6]` on every reset.
- Friction material buckets: `64`.
- Both Franka fingers use fixed static/dynamic friction `0.5/0.5`.
- Damage threshold is recomputed after reset as
  `mass * |gravity| / mean(gripper_static, object_static) * 1.1`.
- Damage requires strict threshold exceedance for four consecutive environment
  steps.

Task 4 (`ketchup_1`) receives the same physics and damage terminal for the first
time. This is an experimental domain-randomization/stress profile; the common
mass range is not a claim about the real mass of each object.

Direct Tabero evaluation selects the profile with `--config-path`. RLinf can use
the same profile through its existing `libero_config_dir` YAML field; no RLinf
source change is required.
