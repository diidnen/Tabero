# Task 0 fixed-mean physics without control overrides

This single-task `libero_object` profile is derived from
`alltask_fixed_damage_1000000_uniform_mass_05_16_friction_04_08_from_rlinf_sft_20k`
for Task 0 training and matched evaluation with physics randomization disabled.

- `alphabet_soup_1` mass is fixed to the midpoint of `[0.5, 1.6] kg`: `1.05 kg`.
- Object static friction is fixed to the midpoint of `[0.4, 0.8]`: `0.6`.
- Object dynamic friction is fixed to the midpoint of `[0.3, 0.6]`: `0.45`.
- Both Franka fingers retain fixed static/dynamic friction `0.5/0.5`.
- The damage terminal remains installed at `1,000,000 N` for four consecutive steps.
- No `control_overrides` block is present. The policy retains full control of the
  gripper-force action, so measured-force reward optimization is not masked by a
  fixed controller target.

Mass, object friction, gripper friction, damage threshold, prompt, and initial
state assignment are otherwise kept aligned between training and evaluation.

Source JSON SHA-256:
`6e1a446027fde35f591a3c6584be61a3073fc0cc7485225bb1e378c3c473c342`.
