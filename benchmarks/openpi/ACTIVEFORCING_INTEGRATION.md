# ActiveForcing executor integration

ActiveForcing is disabled by default. The native Tabero `ForcePositionAction` remains the only low-level controller.

## Runtime contract

When enabled with `--control-mode hybrid` or `--control-mode tactile`, the OpenPI policy response must contain the scalar model decision:

```json
{
  "actions": [[/* native OpenPI actions */]],
  "activeforcing": {"target_force_n": 4.0, "source": "model"}
}
```

A top-level `target_force_n` is also accepted. The action chunk is sliced and executed exactly as before; no arm pose or 13D action slice is rewritten. The executor calls only native `set_target_squeeze_force_n()` when the scalar changes, so repeated model responses do not reset Tabero's persistent Newton correction.

## Offline smoke provider

For interface-only testing, pass `--activeforcing-decision-path` to a JSON file containing `{"target_force_n": 4.0}`. This is an explicit test provider and is not a model rollout.

## Telemetry

Each executed step records `F_des_N`, `F_meas_N`, `F_cmd`, `execution_state`, `force_loop_active`, `contact_status`, and the native force status. Episode results include the telemetry rows under the `activeforcing` field; optional step traces include the same fields.

The runtime boundary does not implement a second force loop, object-specific force-to-gap lookup, pause/rebase, or arm trajectory modification.
