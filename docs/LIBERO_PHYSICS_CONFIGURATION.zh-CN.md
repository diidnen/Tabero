# LIBERO 物理属性配置

Tabero 支持在每个 LIBERO task 的 JSON 配置中覆盖普通刚体物体的质量、目标物体
表面摩擦，以及 Franka 左右夹爪的表面摩擦。未提供配置时，环境继续使用 USD 或
IsaacLab/PhysX 默认物理属性。

当前版本只支持 `RigidObjectCfg` 物体的质量，暂不支持 articulated fixture，例如 microwave、stove 和 cabinet。配置引用未知物体或 articulated asset 时会在环境创建阶段报错，不会静默忽略。

## 固定质量

在 task 条目中加入：

```json
{
  "objects": {
    "tomato_sauce_1": {
      "type": "tomato_sauce",
      "scale": [0.01, 0.01, 0.01],
      "initial_region": "floor_target_object_region"
    }
  },
  "physics": {
    "objects": {
      "tomato_sauce_1": {
        "mass_kg": 0.25
      }
    }
  }
}
```

固定质量在 IsaacLab asset spawn 时通过 `MassPropertiesCfg` 写入，因此所有并行环境使用相同的 0.25 kg 质量。

## 均匀随机质量

每个并行环境启动时采样一次：

```json
{
  "physics": {
    "objects": {
      "tomato_sauce_1": {
        "mass_kg": {
          "distribution": "uniform",
          "range": [0.08, 0.12],
          "apply_on": "startup"
        }
      }
    }
  }
}
```

每次对应环境 reset 时重新采样：

```json
{
  "physics": {
    "objects": {
      "tomato_sauce_1": {
        "mass_kg": {
          "distribution": "uniform",
          "range": [0.08, 0.12],
          "apply_on": "reset"
        }
      }
    }
  }
}
```

随机化使用 IsaacLab 的 `randomize_rigid_body_mass`，采用绝对质量区间并在修改质量后重新计算惯量。`reset` 模式会在运行期使用 CPU tensor 更新质量，开销高于 `startup`；如果不要求每个 episode 使用新质量，推荐使用 `startup`。

随机数由 IsaacLab/PyTorch 环境种子控制。在相同配置、相同并行环境数量和相同 reset 顺序下，可获得可复现的采样序列。

## RLinf 中选择配置

RLinf 不需要修改代码。复制一个 LIBERO 配置目录，在对应 task 中加入 `physics` 字段，然后在 RLinf 实验 YAML 中指定该目录：

```yaml
env:
  train:
    init_params:
      libero_config_dir: /absolute/path/to/libero_mass_config
  eval:
    init_params:
      libero_config_dir: /absolute/path/to/libero_mass_config
```

目录中仍需包含完整的 `libero_object.json` 等 task-suite 文件。建议为每组物理参数保留独立目录，不要直接覆盖基准配置，以便在训练记录中明确区分环境版本。

训练和评测通常应指向同一个物理配置目录。如果只在 train 中启用随机化、eval 使用固定质量，应在实验报告中明确记录这种 domain randomization 设置。

## 配置约束

- 质量单位为 kg，必须为有限正数。
- `distribution` 当前仅支持 `uniform`。
- `range` 格式为 `[minimum_kg, maximum_kg]`，两端均包含在采样范围内。
- `apply_on` 仅支持 `startup` 或 `reset`，默认是 `startup`。
- `physics.objects` 中的 key 必须与同一 task 的 `objects` key 完全一致，例如 `tomato_sauce_1`，不能只写类型名 `tomato_sauce`。
- 未配置 `mass_kg` 的物体保持原有行为。

## 固定摩擦系数

夹爪摩擦配置位于 `physics.gripper.friction`；物体摩擦配置位于
`physics.objects.<object_name>.friction`。数值简写会同时设置静摩擦和动摩擦：

```json
{
  "physics": {
    "gripper": {
      "friction": 0.8
    },
    "objects": {
      "tomato_sauce_1": {
        "friction": 0.6
      }
    }
  }
}
```

也可以分别设置：

```json
{
  "physics": {
    "gripper": {
      "friction": {
        "static": 1.0,
        "dynamic": 0.8
      }
    },
    "objects": {
      "tomato_sauce_1": {
        "friction": {
          "static": 0.7,
          "dynamic": 0.5
        }
      }
    }
  }
}
```

固定摩擦在环境 startup 时写入。夹爪配置只作用于 `panda_leftfinger` 和
`panda_rightfinger`，不会修改 Franka 其他 link。一个物体的全部 collision shapes
共享同一组系数；左右 finger 也共享同一组系数。

## 均匀随机摩擦

```json
{
  "physics": {
    "gripper": {
      "friction": {
        "distribution": "uniform",
        "static_range": [0.8, 1.2],
        "dynamic_range": [0.6, 1.0],
        "apply_on": "reset",
        "num_buckets": 64
      }
    },
    "objects": {
      "tomato_sauce_1": {
        "friction": {
          "distribution": "uniform",
          "static_range": [0.4, 0.8],
          "dynamic_range": [0.3, 0.6],
          "apply_on": "reset",
          "num_buckets": 64
        }
      }
    }
  }
}
```

Tabero 在环境初始化时预采样 `num_buckets` 组材质参数，之后为每个环境选择一个
bucket。`startup` 只选择一次；`reset` 在每个 episode reset 时重新选择。预采样
bucket 避免长期运行时创建无限数量的 PhysX material。

随机采样先生成静摩擦，再在不超过该静摩擦的范围内生成动摩擦，始终满足
`dynamic <= static`。同一个环境中，选中的 bucket 会广播到物体全部 collision
shapes，或同时广播到左右 finger，不会出现同一物体内部材质不一致。

摩擦事件只修改 PhysX material buffer 的 static/dynamic friction 两列；不修改
restitution、friction combine mode、质量、惯量或碰撞几何。combine mode 继续沿用
当前环境值，默认是 `average`。

### 摩擦配置约束

- 摩擦系数无量纲，必须为有限非负数；`0` 合法。
- 固定配置必须满足 `dynamic <= static`。
- `distribution` 当前仅支持 `uniform`。
- `static_range` 必填；未提供 `dynamic_range` 时使用相同范围。
- 为保证每个 bucket 都能满足 `dynamic <= static`，必须满足
  `dynamic_range[0] <= static_range[0]`。
- `apply_on` 仅支持 `startup` 或 `reset`，随机配置默认 `reset`。
- `num_buckets` 默认为 64，必须为 1--4096 的整数。
- 物体摩擦只支持 `RigidObjectCfg`；未知物体或 articulated fixture 会报错。
- 缺少摩擦配置时严格 no-op，维持 USD/PhysX 原有材质。

OpenPI evaluator 会在每个 episode 的 raw metrics 中记录实际生效值：

```json
{
  "friction": {
    "gripper": {
      "static_friction": 0.8,
      "dynamic_friction": 0.6
    },
    "objects": {
      "tomato_sauce_1": {
        "static_friction": 0.7,
        "dynamic_friction": 0.5
      }
    }
  }
}
```

Task 级 JSON/TXT 同时记录每个 scope 的 count/mean/min/max，便于检查随机化范围和
固定配置是否实际生效。

## 目标物体损坏 terminal

可以为当前 Task 的每个抓取目标物体配置最大实测夹持力。配置只属于当前 Task；
相同物体类型出现在其他 Task 时不会自动继承。

```json
{
  "physics": {
    "objects": {
      "tomato_sauce_1": {
        "damage": {
          "max_squeeze_force": 25.0,
          "consecutive_frames": 4
        }
      }
    }
  }
}
```

- 手动模式使用 `max_squeeze_force`，该值必须为有限正数。
- `consecutive_frames` 可选，默认为 `4`，必须为正整数。
- 当实测夹持力连续 `consecutive_frames` 个 environment steps 严格大于上限时，
  在最后一帧触发 `object_damage_<object_name>` termination。
- 任一帧夹持力小于或等于上限时，连续计数立即清零；每次 environment reset
  也会清零。
- 夹持力使用项目现有 squeeze 定义：
  `2 * min(abs(left_local_z), abs(right_local_z))`。
- 力来自 `contact_grasp_<object_name>` 的 `force_matrix_w`，该 contact sensor
  按目标物体过滤，因此桌面、basket 或其他物体与手指的接触不会累计该目标物体的
  损坏帧数。
- damage 只允许配置在 Task 的 `obj_of_interest` 对象上。配置到 basket、fixture
  或未知对象会在环境创建前报错。
- 未配置 `damage` 的 Task 不创建 damage terminal，保持原行为。

也可以让上限在每次 reset 时根据本轮实际质量和静摩擦系数计算：

```json
{
  "physics": {
    "gripper": {
      "friction": {"static": 0.5, "dynamic": 0.5}
    },
    "objects": {
      "tomato_sauce_1": {
        "friction": {"static": 0.7, "dynamic": 0.5},
        "damage": {
          "threshold": {
            "mode": "mass_friction",
            "tolerance_factor": 1.1
          },
          "consecutive_frames": 4
        }
      }
    }
  }
}
```

计算公式为：

```text
effective_static_friction = (gripper_static_friction + object_static_friction) / 2
max_squeeze_force = mass_kg * gravity_m_s2 / effective_static_friction * tolerance_factor
```

- `tolerance_factor` 默认为 `1.1`，必须为有限正数。
- `max_squeeze_force` 与 `threshold` 必须且只能配置一个。
- 计算模式要求显式配置 `physics.gripper.friction` 和目标物体的 `friction`。
- 质量读取 reset 后 PhysX 中的实际值；未配置 `mass_kg` 时使用 USD/PhysX 默认质量，
  配置 reset 随机化时使用本轮采样质量。
- 静摩擦读取 Tabero 摩擦事件在本轮实际应用的值；双方最小静摩擦之和必须大于零。
- 重力使用当前仿真重力向量的模，默认约为 `9.81 m/s²`。
- IsaacLab 先完成 reset 质量/摩擦事件，再重置 terminal，因此公式只在每次 reset
  计算一次，episode 中保持不变。
- raw JSON/TXT 会记录每个 episode 的质量、双方静摩擦、有效静摩擦、容忍系数和
  最终 threshold，并汇总 count/mean/min/max。

OpenPI evaluator 保留该 terminal，并在 raw episode metrics 中记录：

```json
{
  "success": false,
  "end_reason": "object_damage",
  "terminal_term": "object_damage_tomato_sauce_1"
}
```

如果 success 与 damage 在同一步同时满足，damage termination 优先，该 episode
按物体损坏失败记录。
