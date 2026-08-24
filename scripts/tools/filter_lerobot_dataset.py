#!/usr/bin/env python3
"""Create a filtered, self-consistent LeRobot v2.1 dataset.

The default filter is tailored to Tabero Firm data: episodes whose prompt
contains the whole word ``gently`` or ``softly`` are removed, while every
retained prompt must contain ``firmly`` or ``tightly``.  Prompt adverbs may be
at either the beginning or the end of the instruction.

The source dataset is never modified.  The output is assembled in a temporary
directory, fully verified, and atomically renamed into place.  Existing output
directories are never overwritten.

Example:

    ../../T2-VLA/.venv/bin/python scripts/tools/filter_lerobot_dataset.py \
        --source-dir ../datas/tabero \
        --output-dir ../datas/tabero_firm \
        --dry-run
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_EXCLUDED_ADVERBS = ("gently", "softly")
DEFAULT_REQUIRED_ADVERBS = ("firmly", "tightly")
REINDEXED_COLUMNS = ("episode_index", "index", "task_index")


@dataclass(frozen=True)
class EpisodeMapping:
    """Old-to-new identifiers for one retained episode."""

    old_episode_index: int
    new_episode_index: int
    old_task_index: int
    new_task_index: int
    task: str
    length: int
    source_path: Path


@dataclass(frozen=True)
class FilterPlan:
    """Validated source metadata and deterministic output mapping."""

    source_dir: Path
    output_dir: Path
    source_info: dict[str, Any]
    source_episodes: list[dict[str, Any]]
    source_stats: dict[int, dict[str, Any]]
    source_tasks: dict[int, str]
    task_mapping: dict[int, int]
    episodes: list[EpisodeMapping]
    removed_episode_indices: list[int]
    excluded_adverbs: tuple[str, ...]
    required_adverbs: tuple[str, ...]

    @property
    def source_frames(self) -> int:
        return sum(int(record["length"]) for record in self.source_episodes)

    @property
    def output_frames(self) -> int:
        return sum(record.length for record in self.episodes)

    @property
    def removed_frames(self) -> int:
        return self.source_frames - self.output_frames


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required. In this workspace use "
            "T2-VLA/.venv/bin/python or RLinf/.venv/bin/python."
        ) from exc
    return pa, pq


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _word_pattern(words: Sequence[str]) -> re.Pattern[str]:
    normalized = tuple(word.strip().lower() for word in words if word.strip())
    if not normalized:
        return re.compile(r"(?!x)x")
    alternatives = "|".join(re.escape(word) for word in normalized)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def _single_task(record: dict[str, Any], *, source: str) -> str:
    tasks = record.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], str):
        raise ValueError(
            f"{source} must contain exactly one string task, got {tasks!r}"
        )
    return tasks[0]


def _episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    relative = template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return root / relative


def build_filter_plan(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    excluded_adverbs: Sequence[str] = DEFAULT_EXCLUDED_ADVERBS,
    required_adverbs: Sequence[str] = DEFAULT_REQUIRED_ADVERBS,
) -> FilterPlan:
    """Validate source metadata and construct stable old-to-new mappings."""

    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source == output:
        raise ValueError("Source and output directories must differ")

    meta_dir = source / "meta"
    required_files = (
        meta_dir / "info.json",
        meta_dir / "tasks.jsonl",
        meta_dir / "episodes.jsonl",
        meta_dir / "episodes_stats.jsonl",
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required metadata: {missing}")

    info = _read_json(meta_dir / "info.json")
    episodes = _read_jsonl(meta_dir / "episodes.jsonl")
    tasks_records = _read_jsonl(meta_dir / "tasks.jsonl")
    stats_records = _read_jsonl(meta_dir / "episodes_stats.jsonl")

    source_tasks: dict[int, str] = {}
    task_to_old_index: dict[str, int] = {}
    for record in tasks_records:
        task_index = int(record["task_index"])
        task = str(record["task"])
        if task_index in source_tasks:
            raise ValueError(f"Duplicate task_index={task_index}")
        if task in task_to_old_index:
            raise ValueError(f"Duplicate task text: {task!r}")
        source_tasks[task_index] = task
        task_to_old_index[task] = task_index

    source_stats: dict[int, dict[str, Any]] = {}
    for record in stats_records:
        episode_index = int(record["episode_index"])
        if episode_index in source_stats:
            raise ValueError(f"Duplicate stats episode_index={episode_index}")
        source_stats[episode_index] = record

    expected_indices = list(range(len(episodes)))
    actual_indices = [int(record["episode_index"]) for record in episodes]
    if actual_indices != expected_indices:
        raise ValueError(
            "Source episode indices must be ordered and contiguous from zero"
        )
    if set(source_stats) != set(expected_indices):
        raise ValueError(
            "episodes_stats.jsonl does not cover exactly the source episodes"
        )
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError("info.json total_episodes disagrees with episodes.jsonl")
    source_frames = sum(int(record["length"]) for record in episodes)
    if int(info.get("total_frames", -1)) != source_frames:
        raise ValueError("info.json total_frames disagrees with episode lengths")

    exclude_pattern = _word_pattern(excluded_adverbs)
    require_pattern = _word_pattern(required_adverbs)
    kept_records: list[tuple[dict[str, Any], int, Path]] = []
    removed_indices: list[int] = []
    used_old_task_indices: set[int] = set()

    for record in episodes:
        old_episode_index = int(record["episode_index"])
        task = _single_task(record, source=f"episode {old_episode_index}")
        if task not in task_to_old_index:
            raise ValueError(
                f"Episode {old_episode_index} references unknown task {task!r}"
            )
        is_excluded = exclude_pattern.search(task) is not None
        is_required = require_pattern.search(task) is not None
        if is_excluded and is_required:
            raise ValueError(
                f"Episode {old_episode_index} matches both excluded and required adverbs: {task!r}"
            )
        if is_excluded:
            removed_indices.append(old_episode_index)
            continue
        if not is_required:
            raise ValueError(
                f"Episode {old_episode_index} matches neither filter category: {task!r}"
            )
        old_task_index = task_to_old_index[task]
        source_path = _episode_path(source, info, old_episode_index)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source parquet: {source_path}")
        used_old_task_indices.add(old_task_index)
        kept_records.append((record, old_task_index, source_path))

    ordered_old_task_indices = [
        old_index for old_index in source_tasks if old_index in used_old_task_indices
    ]
    task_mapping = {
        old_index: new_index
        for new_index, old_index in enumerate(ordered_old_task_indices)
    }
    mappings = [
        EpisodeMapping(
            old_episode_index=int(record["episode_index"]),
            new_episode_index=new_episode_index,
            old_task_index=old_task_index,
            new_task_index=task_mapping[old_task_index],
            task=_single_task(record, source=f"episode {record['episode_index']}"),
            length=int(record["length"]),
            source_path=source_path,
        )
        for new_episode_index, (record, old_task_index, source_path) in enumerate(
            kept_records
        )
    ]

    return FilterPlan(
        source_dir=source,
        output_dir=output,
        source_info=info,
        source_episodes=episodes,
        source_stats=source_stats,
        source_tasks=source_tasks,
        task_mapping=task_mapping,
        episodes=mappings,
        removed_episode_indices=removed_indices,
        excluded_adverbs=tuple(excluded_adverbs),
        required_adverbs=tuple(required_adverbs),
    )


def _unique_ints(column: Any) -> set[int]:
    return {int(value) for value in column.unique().to_pylist()}


def validate_source_parquets(plan: FilterPlan) -> None:
    """Validate all source index columns and task references."""

    _, pq = _require_pyarrow()
    expected_global_index = 0
    episode_lookup = {item.old_episode_index: item for item in plan.episodes}
    all_paths = {
        int(record["episode_index"]): _episode_path(
            plan.source_dir, plan.source_info, int(record["episode_index"])
        )
        for record in plan.source_episodes
    }
    task_to_index = {task: index for index, task in plan.source_tasks.items()}

    for record in plan.source_episodes:
        episode_index = int(record["episode_index"])
        path = all_paths[episode_index]
        if not path.is_file():
            raise FileNotFoundError(f"Missing source parquet: {path}")
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "index", "task_index"],
        )
        length = int(record["length"])
        task = _single_task(record, source=f"episode {episode_index}")
        expected_task_index = task_to_index[task]
        if len(table) != length:
            raise ValueError(f"{path}: rows={len(table)}, metadata length={length}")
        if _unique_ints(table["episode_index"]) != {episode_index}:
            raise ValueError(f"{path}: wrong episode_index values")
        if _unique_ints(table["task_index"]) != {expected_task_index}:
            raise ValueError(f"{path}: wrong task_index values")
        if table["frame_index"].to_pylist() != list(range(length)):
            raise ValueError(f"{path}: frame_index is not contiguous from zero")
        expected_indices = list(
            range(expected_global_index, expected_global_index + length)
        )
        if table["index"].to_pylist() != expected_indices:
            raise ValueError(f"{path}: global index is not contiguous")
        expected_global_index += length

        selected = episode_lookup.get(episode_index)
        if selected is not None and selected.length != len(table):
            raise ValueError(
                f"{path}: selected episode length changed during validation"
            )

    if expected_global_index != plan.source_frames:
        raise ValueError("Validated source frame total disagrees with metadata")


def _replace_int_column(table: Any, name: str, values: Iterable[int]) -> Any:
    pa, _ = _require_pyarrow()
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Parquet table is missing required column {name!r}")
    field = table.schema.field(column_index)
    replacement = pa.array(values, type=field.type)
    return table.set_column(column_index, field, replacement)


def _constant_stats(original: dict[str, Any], value: int) -> dict[str, Any]:
    return {
        "min": [value],
        "max": [value],
        "mean": [float(value)],
        "std": [0.0],
        "count": copy.deepcopy(original.get("count", [])),
    }


def _reindex_stats(
    source_record: dict[str, Any],
    mapping: EpisodeMapping,
    new_frame_start: int,
) -> dict[str, Any]:
    output = copy.deepcopy(source_record)
    output["episode_index"] = mapping.new_episode_index
    stats = output.get("stats")
    if not isinstance(stats, dict):
        raise ValueError(
            f"Stats for episode {mapping.old_episode_index} do not contain an object"
        )
    for name, value in (
        ("episode_index", mapping.new_episode_index),
        ("task_index", mapping.new_task_index),
    ):
        if name not in stats or not isinstance(stats[name], dict):
            raise ValueError(
                f"Stats for episode {mapping.old_episode_index} are missing {name}"
            )
        stats[name] = _constant_stats(stats[name], value)

    if "index" not in stats or not isinstance(stats["index"], dict):
        raise ValueError(
            f"Stats for episode {mapping.old_episode_index} are missing index"
        )
    original_index_stats = stats["index"]
    stats["index"] = {
        "min": [new_frame_start],
        "max": [new_frame_start + mapping.length - 1],
        "mean": [new_frame_start + (mapping.length - 1) / 2.0],
        "std": copy.deepcopy(original_index_stats.get("std", [])),
        "count": copy.deepcopy(original_index_stats.get("count", [])),
    }
    return output


def _output_info(plan: FilterPlan) -> dict[str, Any]:
    output = copy.deepcopy(plan.source_info)
    chunks_size = int(output.get("chunks_size", 1000))
    output.update(
        {
            "total_episodes": len(plan.episodes),
            "total_frames": plan.output_frames,
            "total_tasks": len(plan.task_mapping),
            "total_videos": 0,
            "total_chunks": max(
                1, (len(plan.episodes) + chunks_size - 1) // chunks_size
            ),
            "splits": {"train": f"0:{len(plan.episodes)}"},
        }
    )
    return output


def _manifest(plan: FilterPlan) -> dict[str, Any]:
    source_meta = plan.source_dir / "meta"
    return {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(plan.source_dir),
        "filter": {
            "exclude_whole_words": list(plan.excluded_adverbs),
            "require_whole_words": list(plan.required_adverbs),
            "match_position": "anywhere",
            "case_sensitive": False,
        },
        "counts": {
            "source_episodes": len(plan.source_episodes),
            "source_frames": plan.source_frames,
            "removed_episodes": len(plan.removed_episode_indices),
            "removed_frames": plan.removed_frames,
            "output_episodes": len(plan.episodes),
            "output_frames": plan.output_frames,
            "output_tasks": len(plan.task_mapping),
        },
        "source_metadata_sha256": {
            name: _sha256(source_meta / name)
            for name in (
                "info.json",
                "tasks.jsonl",
                "episodes.jsonl",
                "episodes_stats.jsonl",
            )
        },
        "removed_episode_indices": plan.removed_episode_indices,
        "task_index_mapping": [
            {
                "old_task_index": old_index,
                "new_task_index": new_index,
                "task": plan.source_tasks[old_index],
            }
            for old_index, new_index in plan.task_mapping.items()
        ],
        "episode_index_mapping": [
            {
                "old_episode_index": item.old_episode_index,
                "new_episode_index": item.new_episode_index,
                "old_task_index": item.old_task_index,
                "new_task_index": item.new_task_index,
                "length": item.length,
            }
            for item in plan.episodes
        ],
    }


def _print_plan(plan: FilterPlan) -> None:
    print(f"source: {plan.source_dir}")
    print(f"output: {plan.output_dir}")
    print(f"exclude whole words: {', '.join(plan.excluded_adverbs)}")
    print(f"require whole words: {', '.join(plan.required_adverbs)}")
    print(f"source episodes: {len(plan.source_episodes)}")
    print(f"source frames: {plan.source_frames}")
    print(f"removed episodes: {len(plan.removed_episode_indices)}")
    print(f"removed frames: {plan.removed_frames}")
    print(f"output episodes: {len(plan.episodes)}")
    print(f"output frames: {plan.output_frames}")
    print(f"output tasks: {len(plan.task_mapping)}")


def _assert_non_reindexed_columns_equal(
    source_table: Any, output_table: Any, path: Path
) -> None:
    for name in source_table.column_names:
        if name in REINDEXED_COLUMNS:
            continue
        if name not in output_table.column_names:
            raise ValueError(f"{path}: output is missing source column {name!r}")
        if not source_table[name].equals(output_table[name]):
            raise ValueError(f"{path}: column {name!r} changed during rewrite")


def _write_dataset(plan: FilterPlan, staging_dir: Path) -> None:
    _, pq = _require_pyarrow()
    (staging_dir / "meta").mkdir(parents=True)
    chunks_size = int(plan.source_info.get("chunks_size", 1000))
    new_frame_start = 0
    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []

    for offset, mapping in enumerate(plan.episodes, start=1):
        source_table = pq.read_table(mapping.source_path)
        if len(source_table) != mapping.length:
            raise ValueError(
                f"{mapping.source_path}: rows={len(source_table)}, expected={mapping.length}"
            )
        output_table = _replace_int_column(
            source_table,
            "episode_index",
            [mapping.new_episode_index] * mapping.length,
        )
        output_table = _replace_int_column(
            output_table,
            "index",
            range(new_frame_start, new_frame_start + mapping.length),
        )
        output_table = _replace_int_column(
            output_table,
            "task_index",
            [mapping.new_task_index] * mapping.length,
        )

        chunk = mapping.new_episode_index // chunks_size
        output_path = (
            staging_dir
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{mapping.new_episode_index:06d}.parquet"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_file = pq.ParquetFile(mapping.source_path)
        row_group_size = parquet_file.metadata.row_group(0).num_rows
        compression_values = {
            parquet_file.metadata.row_group(0).column(index).compression.lower()
            for index in range(parquet_file.metadata.num_columns)
        }
        compression = (
            compression_values.pop() if len(compression_values) == 1 else "snappy"
        )
        pq.write_table(
            output_table,
            output_path,
            compression=compression,
            row_group_size=row_group_size,
        )

        roundtrip = pq.read_table(output_path)
        if not output_table.equals(roundtrip, check_metadata=True):
            raise ValueError(
                f"{output_path}: Parquet round-trip changed table content or metadata"
            )
        _assert_non_reindexed_columns_equal(source_table, roundtrip, output_path)

        source_episode = plan.source_episodes[mapping.old_episode_index]
        episode_record = copy.deepcopy(source_episode)
        episode_record["episode_index"] = mapping.new_episode_index
        episode_record["length"] = mapping.length
        output_episodes.append(episode_record)
        output_stats.append(
            _reindex_stats(
                plan.source_stats[mapping.old_episode_index],
                mapping,
                new_frame_start,
            )
        )
        new_frame_start += mapping.length
        if offset % 25 == 0 or offset == len(plan.episodes):
            print(f"rewritten and verified episodes: {offset}/{len(plan.episodes)}")

    output_tasks = [
        {
            "task_index": new_index,
            "task": plan.source_tasks[old_index],
        }
        for old_index, new_index in plan.task_mapping.items()
    ]
    _write_json(staging_dir / "meta" / "info.json", _output_info(plan))
    _write_jsonl(staging_dir / "meta" / "tasks.jsonl", output_tasks)
    _write_jsonl(staging_dir / "meta" / "episodes.jsonl", output_episodes)
    _write_jsonl(staging_dir / "meta" / "episodes_stats.jsonl", output_stats)
    _write_json(staging_dir / "meta" / "filter_manifest.json", _manifest(plan))
    attributes = plan.source_dir / ".gitattributes"
    if attributes.is_file():
        shutil.copy2(attributes, staging_dir / ".gitattributes")


def verify_output_dataset(plan: FilterPlan, root: str | Path) -> None:
    """Perform an end-to-end structural scan of a generated dataset."""

    _, pq = _require_pyarrow()
    output = Path(root)
    info = _read_json(output / "meta" / "info.json")
    tasks_records = _read_jsonl(output / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(output / "meta" / "episodes.jsonl")
    stats = _read_jsonl(output / "meta" / "episodes_stats.jsonl")

    expected_counts = (
        len(plan.episodes),
        plan.output_frames,
        len(plan.task_mapping),
    )
    actual_counts = (
        int(info.get("total_episodes", -1)),
        int(info.get("total_frames", -1)),
        int(info.get("total_tasks", -1)),
    )
    if actual_counts != expected_counts:
        raise ValueError(
            f"Output info counts {actual_counts} != expected {expected_counts}"
        )
    if len(episodes) != len(plan.episodes) or len(stats) != len(plan.episodes):
        raise ValueError("Output episode metadata line counts are incorrect")
    if len(tasks_records) != len(plan.task_mapping):
        raise ValueError("Output task metadata line count is incorrect")

    task_map = {
        int(record["task_index"]): str(record["task"]) for record in tasks_records
    }
    exclude_pattern = _word_pattern(plan.excluded_adverbs)
    require_pattern = _word_pattern(plan.required_adverbs)
    if sorted(task_map) != list(range(len(task_map))):
        raise ValueError("Output task indices are not contiguous")
    for task in task_map.values():
        if exclude_pattern.search(task) or not require_pattern.search(task):
            raise ValueError(f"Output contains a non-Firm task: {task!r}")

    expected_global_index = 0
    chunks_size = int(info.get("chunks_size", 1000))
    for expected_episode_index, (episode, stats_record) in enumerate(
        zip(episodes, stats)
    ):
        if int(episode["episode_index"]) != expected_episode_index:
            raise ValueError("Output episode indices are not contiguous")
        if int(stats_record["episode_index"]) != expected_episode_index:
            raise ValueError("Output stats episode indices are not contiguous")
        length = int(episode["length"])
        task = _single_task(episode, source=f"output episode {expected_episode_index}")
        path = (
            output
            / "data"
            / f"chunk-{expected_episode_index // chunks_size:03d}"
            / f"episode_{expected_episode_index:06d}.parquet"
        )
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "index", "task_index"],
        )
        task_indices = _unique_ints(table["task_index"])
        if len(table) != length:
            raise ValueError(f"{path}: row count disagrees with metadata")
        if _unique_ints(table["episode_index"]) != {expected_episode_index}:
            raise ValueError(f"{path}: wrong episode_index")
        if len(task_indices) != 1 or task_map[next(iter(task_indices))] != task:
            raise ValueError(f"{path}: task metadata mismatch")
        if table["frame_index"].to_pylist() != list(range(length)):
            raise ValueError(f"{path}: frame_index is not contiguous")
        expected_indices = list(
            range(expected_global_index, expected_global_index + length)
        )
        if table["index"].to_pylist() != expected_indices:
            raise ValueError(f"{path}: global index is not contiguous")
        expected_global_index += length

    if expected_global_index != plan.output_frames:
        raise ValueError("Output global frame count is incorrect")
    parquet_count = sum(1 for _ in (output / "data").rglob("episode_*.parquet"))
    if parquet_count != len(plan.episodes):
        raise ValueError(
            f"Output has {parquet_count} Parquets, expected {len(plan.episodes)}"
        )


def filter_lerobot_dataset(plan: FilterPlan, *, dry_run: bool = False) -> None:
    """Execute a validated filter plan."""

    _print_plan(plan)
    print("validating all source parquet index columns...")
    validate_source_parquets(plan)
    print("source parquet validation: passed")
    if dry_run:
        print("dry-run: no files written")
        return
    if plan.output_dir.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {plan.output_dir}"
        )
    plan.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{plan.output_dir.name}.build-",
            dir=plan.output_dir.parent,
        )
    )
    try:
        print(f"staging directory: {staging_dir}")
        _write_dataset(plan, staging_dir)
        print("running final structural verification...")
        verify_output_dataset(plan, staging_dir)
        if plan.output_dir.exists():
            raise FileExistsError(
                f"Output appeared during generation; refusing to overwrite: {plan.output_dir}"
            )
        os.rename(staging_dir, plan.output_dir)
        print(f"published: {plan.output_dir}")
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-task-adverbs",
        nargs="+",
        default=list(DEFAULT_EXCLUDED_ADVERBS),
        metavar="WORD",
    )
    parser.add_argument(
        "--require-task-adverbs",
        nargs="+",
        default=list(DEFAULT_REQUIRED_ADVERBS),
        metavar="WORD",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print exact counts without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = build_filter_plan(
        args.source_dir,
        args.output_dir,
        excluded_adverbs=args.exclude_task_adverbs,
        required_adverbs=args.require_task_adverbs,
    )
    filter_lerobot_dataset(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
