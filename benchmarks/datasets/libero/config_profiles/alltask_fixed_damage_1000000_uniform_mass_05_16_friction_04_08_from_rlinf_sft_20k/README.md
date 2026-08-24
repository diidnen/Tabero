# All-task uniform mass/friction fixed-high-damage baseline

This profile is copied from
`alltask_damage_uniform_mass_05_16_friction_04_08_from_rlinf_sft_20k` and
keeps the same reset-time domain randomization for every `libero_object`
Task 0--9 target.

- Target mass: uniform `[0.5, 1.6] kg` on every reset.
- Target static friction: uniform `[0.4, 0.8]` on every reset.
- Target dynamic friction: uniform `[0.3, 0.6]` on every reset.
- Friction material buckets: `64`.
- Both Franka fingers use fixed static/dynamic friction `0.5/0.5`.
- Damage uses a manually fixed `max_squeeze_force=1,000,000 N` for every
  target and still requires strict threshold exceedance for four consecutive
  environment steps.

The fixed threshold deliberately remains finite and keeps the same Tabero
terminal code path, but is far above the force range observed in this benchmark.
It is therefore a damage-limit ablation baseline, not a physical calibration.
Any damage trigger invalidates that interpretation and must be reported.

This is an experimental domain-randomization profile; the common mass range is
not a claim about the real mass of each object.

Direct Tabero evaluation selects the profile with `--config-path`. RLinf can use
the same profile through its existing `libero_config_dir` YAML field; no RLinf
source change is required.
