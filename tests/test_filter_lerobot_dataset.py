from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.tools.filter_lerobot_dataset import (
    build_filter_plan,
    filter_lerobot_dataset,
    verify_output_dataset,
)


TASKS = (
    "firmly pick up the object",
    "pick up the object tightly",
    "gently pick up the object",
    "pick up the object softly",
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def _stats(value: int, start: int, length: int) -> dict:
    count = [length]
    return {
        "episode_index": value,
        "stats": {
            "value": {
                "min": [value],
                "max": [value],
                "mean": [float(value)],
                "std": [0.0],
                "count": count,
            },
            "frame_index": {
                "min": [0],
                "max": [length - 1],
                "mean": [(length - 1) / 2],
                "std": [0.5],
                "count": count,
            },
            "episode_index": {
                "min": [value],
                "max": [value],
                "mean": [float(value)],
                "std": [0.0],
                "count": count,
            },
            "index": {
                "min": [start],
                "max": [start + length - 1],
                "mean": [start + (length - 1) / 2],
                "std": [0.5],
                "count": count,
            },
            "task_index": {
                "min": [value],
                "max": [value],
                "mean": [float(value)],
                "std": [0.0],
                "count": count,
            },
        },
    }


def _make_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    lengths = (2, 3, 2, 1)
    total_frames = sum(lengths)
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 4,
        "total_frames": total_frames,
        "total_tasks": 4,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": "0:4"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "value": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        root / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": task} for index, task in enumerate(TASKS)],
    )

    episodes = []
    stats = []
    global_index = 0
    schema_metadata = {b"test": b"preserve-me"}
    for episode_index, (task, length) in enumerate(zip(TASKS, lengths)):
        episodes.append(
            {"episode_index": episode_index, "tasks": [task], "length": length}
        )
        stats.append(_stats(episode_index, global_index, length))
        table = pa.table(
            {
                "value": [episode_index * 10 + frame for frame in range(length)],
                "frame_index": list(range(length)),
                "episode_index": [episode_index] * length,
                "index": list(range(global_index, global_index + length)),
                "task_index": [episode_index] * length,
            }
        ).replace_schema_metadata(schema_metadata)
        pq.write_table(
            table,
            data / f"episode_{episode_index:06d}.parquet",
            compression="snappy",
            row_group_size=2,
        )
        global_index += length
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats)


def test_dry_run_matches_prefix_and_suffix_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_dataset(source)

    plan = build_filter_plan(source, output)

    assert [item.old_episode_index for item in plan.episodes] == [0, 1]
    assert plan.removed_episode_indices == [2, 3]
    assert plan.output_frames == 5
    assert plan.removed_frames == 3
    filter_lerobot_dataset(plan, dry_run=True)
    assert not output.exists()


def test_filter_reindexes_data_metadata_and_stats(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_dataset(source)
    plan = build_filter_plan(source, output)

    filter_lerobot_dataset(plan)
    verify_output_dataset(plan, output)

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"]) == (
        2,
        5,
        2,
    )
    tasks = [
        json.loads(line)
        for line in (output / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert tasks == [
        {"task_index": 0, "task": TASKS[0]},
        {"task_index": 1, "task": TASKS[1]},
    ]
    second = pq.read_table(output / "data" / "chunk-000" / "episode_000001.parquet")
    assert second["value"].to_pylist() == [10, 11, 12]
    assert second["episode_index"].to_pylist() == [1, 1, 1]
    assert second["index"].to_pylist() == [2, 3, 4]
    assert second["task_index"].to_pylist() == [1, 1, 1]
    assert second.schema.metadata == {b"test": b"preserve-me"}

    stats = [
        json.loads(line)
        for line in (output / "meta" / "episodes_stats.jsonl").read_text().splitlines()
    ]
    assert stats[1]["stats"]["index"]["min"] == [2]
    assert stats[1]["stats"]["index"]["max"] == [4]
    assert stats[1]["stats"]["task_index"]["mean"] == [1.0]
    assert stats[1]["stats"]["value"] == _stats(1, 2, 3)["stats"]["value"]


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_dataset(source)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    plan = build_filter_plan(source, output)

    try:
        filter_lerobot_dataset(plan)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected existing output to be rejected")
    assert sentinel.read_text(encoding="utf-8") == "keep"
