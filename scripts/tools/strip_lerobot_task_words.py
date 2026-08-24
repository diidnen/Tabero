#!/usr/bin/env python3
"""Create a LeRobot v2.1 dataset with selected words removed from prompts.

The source dataset is never modified. Task strings are rewritten and
deduplicated, every Parquet ``task_index`` is remapped, and task-index episode
statistics are updated. The output is built in a temporary directory, fully
verified, and atomically published. Existing output directories are never
overwritten.

Example:

    ../RLinf/.venv/bin/python scripts/tools/strip_lerobot_task_words.py \
        --source-dir ../datas/tabero_firm \
        --output-dir ../datas/tabero_no_adverb \
        --remove-task-words tightly firmly
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


DEFAULT_REMOVED_WORDS = ("tightly", "firmly")
CORE_METADATA = (
    "info.json",
    "tasks.jsonl",
    "episodes.jsonl",
    "episodes_stats.jsonl",
)


@dataclass(frozen=True)
class EpisodePlan:
    episode_index: int
    length: int
    old_task_index: int
    new_task_index: int
    old_task: str
    new_task: str
    source_path: Path


@dataclass(frozen=True)
class RewritePlan:
    source_dir: Path
    output_dir: Path
    source_info: dict[str, Any]
    source_episodes: list[dict[str, Any]]
    source_stats: list[dict[str, Any]]
    source_tasks: dict[int, str]
    new_tasks: list[str]
    task_mapping: dict[int, int]
    episodes: list[EpisodePlan]
    removed_words: tuple[str, ...]

    @property
    def total_frames(self) -> int:
        return sum(episode.length for episode in self.episodes)


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
    normalized = tuple(word.strip() for word in words if word.strip())
    if not normalized:
        raise ValueError("At least one non-empty word must be supplied")
    alternatives = "|".join(re.escape(word) for word in normalized)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def _strip_words(text: str, pattern: re.Pattern[str]) -> str:
    rewritten = pattern.sub("", text)
    rewritten = re.sub(r"\s+", " ", rewritten)
    rewritten = re.sub(r"\s+([,.;:!?])", r"\1", rewritten).strip()
    if not rewritten:
        raise ValueError(f"Removing words produced an empty task from {text!r}")
    return rewritten


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
    return root / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def build_rewrite_plan(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    removed_words: Sequence[str] = DEFAULT_REMOVED_WORDS,
) -> RewritePlan:
    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source == output:
        raise ValueError("Source and output directories must differ")

    meta = source / "meta"
    missing = [meta / name for name in CORE_METADATA if not (meta / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required metadata: {missing}")

    info = _read_json(meta / "info.json")
    episodes = _read_jsonl(meta / "episodes.jsonl")
    stats = _read_jsonl(meta / "episodes_stats.jsonl")
    task_records = _read_jsonl(meta / "tasks.jsonl")
    pattern = _word_pattern(removed_words)

    source_tasks: dict[int, str] = {}
    task_text_to_index: dict[str, int] = {}
    for record in task_records:
        task_index = int(record["task_index"])
        task = str(record["task"])
        if task_index in source_tasks or task in task_text_to_index:
            raise ValueError(
                f"Duplicate source task: index={task_index}, task={task!r}"
            )
        source_tasks[task_index] = task
        task_text_to_index[task] = task_index
    if sorted(source_tasks) != list(range(len(source_tasks))):
        raise ValueError("Source task indices must be contiguous from zero")

    expected_episode_indices = list(range(len(episodes)))
    if [
        int(record["episode_index"]) for record in episodes
    ] != expected_episode_indices:
        raise ValueError("Source episode indices must be contiguous from zero")
    if [int(record["episode_index"]) for record in stats] != expected_episode_indices:
        raise ValueError("Source stats must cover episodes in contiguous order")
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError("info.json total_episodes disagrees with episodes.jsonl")
    total_frames = sum(int(record["length"]) for record in episodes)
    if int(info.get("total_frames", -1)) != total_frames:
        raise ValueError("info.json total_frames disagrees with episode lengths")

    new_tasks: list[str] = []
    new_task_to_index: dict[str, int] = {}
    task_mapping: dict[int, int] = {}
    for old_task_index, old_task in source_tasks.items():
        if pattern.search(old_task) is None:
            raise ValueError(
                f"Source task does not contain any selected word: {old_task!r}"
            )
        new_task = _strip_words(old_task, pattern)
        if pattern.search(new_task) is not None:
            raise ValueError(f"Selected word remains after rewrite: {new_task!r}")
        if new_task not in new_task_to_index:
            new_task_to_index[new_task] = len(new_tasks)
            new_tasks.append(new_task)
        task_mapping[old_task_index] = new_task_to_index[new_task]

    episode_plans: list[EpisodePlan] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        old_task = _single_task(episode, source=f"episode {episode_index}")
        if old_task not in task_text_to_index:
            raise ValueError(
                f"Episode {episode_index} references unknown task {old_task!r}"
            )
        old_task_index = task_text_to_index[old_task]
        source_path = _episode_path(source, info, episode_index)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source Parquet: {source_path}")
        episode_plans.append(
            EpisodePlan(
                episode_index=episode_index,
                length=int(episode["length"]),
                old_task_index=old_task_index,
                new_task_index=task_mapping[old_task_index],
                old_task=old_task,
                new_task=_strip_words(old_task, pattern),
                source_path=source_path,
            )
        )

    return RewritePlan(
        source_dir=source,
        output_dir=output,
        source_info=info,
        source_episodes=episodes,
        source_stats=stats,
        source_tasks=source_tasks,
        new_tasks=new_tasks,
        task_mapping=task_mapping,
        episodes=episode_plans,
        removed_words=tuple(word.strip() for word in removed_words if word.strip()),
    )


def _unique_ints(column: Any) -> set[int]:
    return {int(value) for value in column.unique().to_pylist()}


def _replace_int_column(table: Any, name: str, value: int) -> Any:
    pa, _ = _require_pyarrow()
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Parquet table is missing required column {name!r}")
    field = table.schema.field(column_index)
    replacement = pa.array([value] * len(table), type=field.type)
    return table.set_column(column_index, field, replacement)


def _constant_stats(original: dict[str, Any], value: int) -> dict[str, Any]:
    return {
        "min": [value],
        "max": [value],
        "mean": [float(value)],
        "std": [0.0],
        "count": copy.deepcopy(original.get("count", [])),
    }


def _output_info(plan: RewritePlan) -> dict[str, Any]:
    output = copy.deepcopy(plan.source_info)
    output["total_tasks"] = len(plan.new_tasks)
    return output


def _manifest(plan: RewritePlan) -> dict[str, Any]:
    source_meta = plan.source_dir / "meta"
    return {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(plan.source_dir),
        "transformation": {
            "remove_whole_words": list(plan.removed_words),
            "match_position": "anywhere",
            "case_sensitive": False,
            "normalize_whitespace": True,
            "deduplicate_tasks": True,
        },
        "counts": {
            "source_episodes": len(plan.episodes),
            "source_frames": plan.total_frames,
            "source_tasks": len(plan.source_tasks),
            "output_episodes": len(plan.episodes),
            "output_frames": plan.total_frames,
            "output_tasks": len(plan.new_tasks),
        },
        "source_metadata_sha256": {
            name: _sha256(source_meta / name) for name in CORE_METADATA
        },
        "task_index_mapping": [
            {
                "old_task_index": old_index,
                "new_task_index": new_index,
                "old_task": plan.source_tasks[old_index],
                "new_task": plan.new_tasks[new_index],
            }
            for old_index, new_index in plan.task_mapping.items()
        ],
    }


def _validate_source_episode(
    plan: RewritePlan, episode: EpisodePlan, table: Any
) -> None:
    if len(table) != episode.length:
        raise ValueError(
            f"{episode.source_path}: rows={len(table)}, expected={episode.length}"
        )
    expected_index_start = sum(
        item.length for item in plan.episodes[: episode.episode_index]
    )
    expected_indices = list(
        range(expected_index_start, expected_index_start + episode.length)
    )
    if _unique_ints(table["episode_index"]) != {episode.episode_index}:
        raise ValueError(f"{episode.source_path}: wrong episode_index")
    if _unique_ints(table["task_index"]) != {episode.old_task_index}:
        raise ValueError(f"{episode.source_path}: wrong task_index")
    if table["frame_index"].to_pylist() != list(range(episode.length)):
        raise ValueError(f"{episode.source_path}: frame_index is not contiguous")
    if table["index"].to_pylist() != expected_indices:
        raise ValueError(f"{episode.source_path}: global index is not contiguous")


def _write_dataset(plan: RewritePlan, staging_dir: Path) -> None:
    _, pq = _require_pyarrow()
    (staging_dir / "meta").mkdir(parents=True)
    chunks_size = int(plan.source_info.get("chunks_size", 1000))
    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []

    for offset, episode in enumerate(plan.episodes, start=1):
        source_table = pq.read_table(episode.source_path)
        _validate_source_episode(plan, episode, source_table)
        output_table = _replace_int_column(
            source_table, "task_index", episode.new_task_index
        )
        output_path = (
            staging_dir
            / "data"
            / f"chunk-{episode.episode_index // chunks_size:03d}"
            / f"episode_{episode.episode_index:06d}.parquet"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_file = pq.ParquetFile(episode.source_path)
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
            raise ValueError(f"{output_path}: Parquet round-trip changed table content")
        for name in source_table.column_names:
            if name != "task_index" and not source_table[name].equals(roundtrip[name]):
                raise ValueError(f"{output_path}: column {name!r} changed")

        episode_record = copy.deepcopy(plan.source_episodes[episode.episode_index])
        episode_record["tasks"] = [episode.new_task]
        output_episodes.append(episode_record)

        stats_record = copy.deepcopy(plan.source_stats[episode.episode_index])
        stats_body = stats_record.get("stats")
        if not isinstance(stats_body, dict) or not isinstance(
            stats_body.get("task_index"), dict
        ):
            raise ValueError(
                f"Stats for episode {episode.episode_index} lack task_index"
            )
        stats_body["task_index"] = _constant_stats(
            stats_body["task_index"], episode.new_task_index
        )
        output_stats.append(stats_record)

        if offset % 25 == 0 or offset == len(plan.episodes):
            print(
                f"rewritten and verified episodes: {offset}/{len(plan.episodes)}",
                flush=True,
            )

    output_tasks = [
        {"task_index": index, "task": task} for index, task in enumerate(plan.new_tasks)
    ]
    _write_json(staging_dir / "meta" / "info.json", _output_info(plan))
    _write_jsonl(staging_dir / "meta" / "tasks.jsonl", output_tasks)
    _write_jsonl(staging_dir / "meta" / "episodes.jsonl", output_episodes)
    _write_jsonl(staging_dir / "meta" / "episodes_stats.jsonl", output_stats)
    _write_json(staging_dir / "meta" / "transform_manifest.json", _manifest(plan))
    attributes = plan.source_dir / ".gitattributes"
    if attributes.is_file():
        shutil.copy2(attributes, staging_dir / ".gitattributes")


def verify_output_dataset(plan: RewritePlan, root: str | Path) -> None:
    _, pq = _require_pyarrow()
    output = Path(root)
    info = _read_json(output / "meta" / "info.json")
    tasks = _read_jsonl(output / "meta" / "tasks.jsonl")
    episodes = _read_jsonl(output / "meta" / "episodes.jsonl")
    stats = _read_jsonl(output / "meta" / "episodes_stats.jsonl")
    pattern = _word_pattern(plan.removed_words)

    actual_counts = (
        int(info.get("total_episodes", -1)),
        int(info.get("total_frames", -1)),
        int(info.get("total_tasks", -1)),
    )
    expected_counts = (len(plan.episodes), plan.total_frames, len(plan.new_tasks))
    if actual_counts != expected_counts:
        raise ValueError(f"Output counts {actual_counts} != {expected_counts}")
    if len(episodes) != len(plan.episodes) or len(stats) != len(plan.episodes):
        raise ValueError("Output metadata line counts are incorrect")
    task_map = {int(record["task_index"]): str(record["task"]) for record in tasks}
    if sorted(task_map) != list(range(len(plan.new_tasks))):
        raise ValueError("Output task indices are not contiguous")
    if len(set(task_map.values())) != len(task_map):
        raise ValueError("Output tasks are not unique")
    if any(pattern.search(task) for task in task_map.values()):
        raise ValueError("Output task metadata still contains a selected word")

    expected_global_index = 0
    chunks_size = int(info.get("chunks_size", 1000))
    for episode_index, (episode, stats_record) in enumerate(zip(episodes, stats)):
        if int(episode["episode_index"]) != episode_index:
            raise ValueError("Output episode indices are not contiguous")
        if int(stats_record["episode_index"]) != episode_index:
            raise ValueError("Output stats indices are not contiguous")
        task = _single_task(episode, source=f"output episode {episode_index}")
        if pattern.search(task):
            raise ValueError(f"Output episode still contains selected word: {task!r}")
        length = int(episode["length"])
        path = (
            output
            / "data"
            / f"chunk-{episode_index // chunks_size:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "index", "task_index"]
        )
        task_indices = _unique_ints(table["task_index"])
        if len(table) != length:
            raise ValueError(f"{path}: row count disagrees with metadata")
        if _unique_ints(table["episode_index"]) != {episode_index}:
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
        task_stats = stats_record.get("stats", {}).get("task_index", {})
        expected_task_index = next(iter(task_indices))
        if task_stats.get("min") != [expected_task_index] or task_stats.get("max") != [
            expected_task_index
        ]:
            raise ValueError(f"{path}: task_index stats mismatch")
        expected_global_index += length

    if expected_global_index != plan.total_frames:
        raise ValueError("Output global frame count is incorrect")
    parquet_count = sum(1 for _ in (output / "data").rglob("episode_*.parquet"))
    if parquet_count != len(plan.episodes):
        raise ValueError(
            f"Output has {parquet_count} Parquets, expected {len(plan.episodes)}"
        )


def rewrite_dataset(plan: RewritePlan, *, dry_run: bool = False) -> None:
    print(f"source: {plan.source_dir}")
    print(f"output: {plan.output_dir}")
    print(f"remove whole words: {', '.join(plan.removed_words)}")
    print(f"episodes: {len(plan.episodes)}")
    print(f"frames: {plan.total_frames}")
    print(f"tasks: {len(plan.source_tasks)} -> {len(plan.new_tasks)}")
    if dry_run:
        print("dry-run: metadata plan validated; no files written")
        return
    if plan.output_dir.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {plan.output_dir}"
        )
    plan.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{plan.output_dir.name}.build-", dir=plan.output_dir.parent
        )
    )
    try:
        print(f"staging directory: {staging_dir}")
        _write_dataset(plan, staging_dir)
        print("running final structural verification...", flush=True)
        verify_output_dataset(plan, staging_dir)
        if plan.output_dir.exists():
            raise FileExistsError(
                f"Output appeared during generation: {plan.output_dir}"
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
        "--remove-task-words",
        nargs="+",
        default=list(DEFAULT_REMOVED_WORDS),
        metavar="WORD",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = build_rewrite_plan(
        args.source_dir,
        args.output_dir,
        removed_words=args.remove_task_words,
    )
    rewrite_dataset(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
