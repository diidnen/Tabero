# Task 0 fixed-mean physics with a target-contact 20 N squeeze override

This single-task `libero_object` profile is derived from
`alltask_fixed_damage_1000000_uniform_mass_05_16_friction_04_08_from_rlinf_sft_20k`
for a bounded Task 0 evaluation.

- `alphabet_soup_1` mass is fixed to the midpoint of `[0.5, 1.6] kg`: `1.05 kg`.
- Object static friction is fixed to the midpoint of `[0.4, 0.8]`: `0.6`.
- Object dynamic friction is fixed to the midpoint of `[0.3, 0.6]`: `0.45`.
- Both Franka fingers retain fixed static/dynamic friction `0.5/0.5`.
- The damage terminal remains installed at `1,000,000 N` for four consecutive steps.
- After the first object-filtered finger contact of at least `0.1 N`, the controller latches a `20 N` normal-force target per finger (`40 N` in Tabero's two-finger squeeze convention).

The control override is opt-in. Profiles without
`control_overrides.target_contact_squeeze`, or with `enabled=false`, retain the
original policy-driven force behavior.

Source JSON SHA-256:
`6e1a446027fde35f591a3c6584be61a3073fc0cc7485225bb1e378c3c473c342`.
