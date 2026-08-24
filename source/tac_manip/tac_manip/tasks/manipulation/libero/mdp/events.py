# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import random
import torch
from typing import TYPE_CHECKING


from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import (
    sample_random_color,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class set_or_randomize_rigid_body_friction(ManagerTermBase):
    """Apply one coherent friction bucket to selected shapes of each environment.

    IsaacLab's generic material randomizer chooses a bucket independently for
    every collision shape.  LIBERO treats one object, or the two gripper
    fingers, as one material scope.  This term therefore chooses one bucket per
    environment and broadcasts it to all selected shapes while preserving the
    existing restitution column.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                "Friction configuration supports only RigidObject or Articulation "
                f"assets, got {type(self.asset)!r} for {self.asset_cfg.name!r}."
            )

        self.selected_shape_ids = self._resolve_selected_shape_ids()
        static_range = cfg.params["static_friction_range"]
        dynamic_range = cfg.params["dynamic_friction_range"]
        num_buckets = int(cfg.params["num_buckets"])
        self.material_buckets = self._sample_buckets(
            static_range=static_range,
            dynamic_range=dynamic_range,
            num_buckets=num_buckets,
        )

    def _resolve_selected_shape_ids(self) -> torch.Tensor:
        total_shapes = int(self.asset.root_physx_view.max_shapes)
        if isinstance(self.asset, RigidObject) or self.asset_cfg.body_ids == slice(None):
            return torch.arange(total_shapes, dtype=torch.long, device="cpu")

        shapes_per_body: list[int] = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
            shapes_per_body.append(int(link_view.max_shapes))
        if sum(shapes_per_body) != total_shapes:
            raise ValueError(
                "Unable to resolve articulation collision-shape ordering for friction configuration: "
                f"expected {total_shapes} shapes, found {sum(shapes_per_body)}."
            )

        shape_ids: list[int] = []
        for body_id in self.asset_cfg.body_ids:
            start = sum(shapes_per_body[:body_id])
            shape_ids.extend(range(start, start + shapes_per_body[body_id]))
        if not shape_ids:
            raise ValueError(
                f"Friction configuration for {self.asset_cfg.name!r} selected no collision shapes."
            )
        return torch.tensor(shape_ids, dtype=torch.long, device="cpu")

    @staticmethod
    def _sample_buckets(
        *,
        static_range: tuple[float, float],
        dynamic_range: tuple[float, float],
        num_buckets: int,
    ) -> torch.Tensor:
        static_minimum, static_maximum = static_range
        dynamic_minimum, dynamic_maximum = dynamic_range
        static_samples = static_minimum + torch.rand(num_buckets) * (
            static_maximum - static_minimum
        )
        dynamic_upper = torch.minimum(
            torch.full_like(static_samples, dynamic_maximum), static_samples
        )
        dynamic_samples = dynamic_minimum + torch.rand(num_buckets) * (
            dynamic_upper - dynamic_minimum
        )
        return torch.stack((static_samples, dynamic_samples), dim=-1).cpu()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        snapshot_key: str,
    ):
        del static_friction_range, dynamic_friction_range, asset_cfg
        if env_ids is None:
            env_ids_cpu = torch.arange(env.scene.num_envs, dtype=torch.long, device="cpu")
        else:
            env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)

        bucket_ids = torch.randint(
            0, num_buckets, (len(env_ids_cpu),), dtype=torch.long, device="cpu"
        )
        samples = self.material_buckets[bucket_ids]
        materials = self.asset.root_physx_view.get_material_properties()
        shape_ids = self.selected_shape_ids
        materials[env_ids_cpu[:, None], shape_ids[None, :], 0] = samples[:, None, 0]
        materials[env_ids_cpu[:, None], shape_ids[None, :], 1] = samples[:, None, 1]
        self.asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

        snapshots = env.extras.setdefault("physics_friction", {})
        previous = snapshots.get(snapshot_key)
        if not isinstance(previous, dict):
            previous = {
                "static_friction": torch.full(
                    (env.scene.num_envs,), float("nan"), dtype=torch.float32
                ),
                "dynamic_friction": torch.full(
                    (env.scene.num_envs,), float("nan"), dtype=torch.float32
                ),
            }
            snapshots[snapshot_key] = previous
        previous["static_friction"][env_ids_cpu] = samples[:, 0]
        previous["dynamic_friction"][env_ids_cpu] = samples[:, 1]


def randomize_domelight_color_intensity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    intensity_range: tuple[float, float] | None = None,
    color_variation: float = 0.15,
    base_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    default_intensity: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
    textures: list[str] | None = None,
    default_texture: str = "",
):
    """Randomize scene light intensity/color; set texture only for DomeLight."""

    asset: AssetBase = env.scene[asset_cfg.name]
    prim = asset.prims[0]
    textures = textures or []

    intensity_attr = prim.GetAttribute("inputs:intensity")
    color_attr = prim.GetAttribute("inputs:color")
    is_dome_light = prim.GetTypeName() == "DomeLight"
    texture_file_attr = prim.GetAttribute("inputs:texture:file") if is_dome_light else None

    if intensity_attr is not None:
        if intensity_range is None:
            if default_intensity is not None:
                intensity_attr.Set(float(default_intensity))
        else:
            sampled_intensity = random.uniform(intensity_range[0], intensity_range[1])
            intensity_attr.Set(float(sampled_intensity))

    if color_attr is not None:
        if color_variation is None or color_variation <= 0.0:
            color_attr.Set(tuple(base_color))
        else:
            color_attr.Set(sample_random_color(base=base_color, variation=color_variation))

    if texture_file_attr and texture_file_attr.IsValid():
        if textures:
            new_texture = random.sample(textures, 1)[0]
            texture_file_attr.Set(new_texture)
        else:
            texture_file_attr.Set(default_texture)
