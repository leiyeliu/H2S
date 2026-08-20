"""Dataset layout and metadata helpers for audio preprocessing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


NUM_SOURCES = 8
SAMPLE_RATE = 16_000
SOURCE_WAV_TEMPLATE = "source_{index:02d}.wav"
SOURCE_FEATURE_TEMPLATE = "source_{index:02d}.npy"


class DatasetFormatError(ValueError):
    """Raised when a split annotation does not satisfy the AVIS contract."""


@dataclass(frozen=True)
class DirectoryNames:
    """Dataset-relative directory names used by the preprocessing pipeline."""

    raw_audio: Path = Path("WAVAudios")
    separated_audio: Path = Path("WAVAudios_sep")
    features: Path = Path("FEATAudios_sep")
    images: Path = Path("JPEGImages")


@dataclass(frozen=True)
class SplitLayout:
    """Resolved paths for one dataset split."""

    name: str
    dataset_root: Path
    split_root: Path
    annotation_file: Path
    raw_audio_root: Path
    separated_audio_root: Path
    feature_root: Path
    image_root: Path


@dataclass(frozen=True)
class VideoSpec:
    """The part of an AVIS video record needed by preprocessing."""

    split: str
    key: str
    length: int
    file_names: tuple[str, ...]


def safe_relative_directory(value: str | os.PathLike[str], option: str) -> Path:
    """Return a normalized dataset-relative directory or raise a clear error."""

    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{option} must be a non-empty path relative to each split: {value!s}")
    return path


def validate_split_name(split: str) -> str:
    """Reject absolute/traversing split values before paths are constructed."""

    path = PurePosixPath(split)
    if not split or path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError(f"Invalid split name: {split!r}")
    return split


def resolve_annotation_file(dataset_root: Path, split: str) -> Path:
    """Resolve the common AVIS annotation locations in deterministic order."""

    candidates = (
        dataset_root / f"{split}.json",
        dataset_root / split / f"{split}.json",
        dataset_root / "annotations" / f"{split}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find the {split!r} annotation. Checked:\n  - {rendered}"
    )


def make_split_layout(
    dataset_root: Path,
    split: str,
    directories: DirectoryNames,
) -> SplitLayout:
    """Resolve all paths for a split from one dataset root."""

    split = validate_split_name(split)
    dataset_root = dataset_root.expanduser().resolve()
    split_root = dataset_root / split
    return SplitLayout(
        name=split,
        dataset_root=dataset_root,
        split_root=split_root,
        annotation_file=resolve_annotation_file(dataset_root, split),
        raw_audio_root=split_root / directories.raw_audio,
        separated_audio_root=split_root / directories.separated_audio,
        feature_root=split_root / directories.features,
        image_root=split_root / directories.images,
    )


def _safe_annotation_path(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetFormatError(f"{context} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetFormatError(f"{context} must be a safe relative path: {value!r}")
    return value


def _video_key(video: dict[str, object], file_names: Sequence[str], context: str) -> str:
    first_path = PurePosixPath(file_names[0])
    if len(first_path.parts) >= 2:
        key = first_path.parts[0]
        if any(PurePosixPath(name).parts[0] != key for name in file_names):
            raise DatasetFormatError(f"{context} contains frames from more than one directory")
        return key

    for field in ("name", "video_name"):
        value = video.get(field)
        if isinstance(value, str) and value:
            return _safe_annotation_path(value, context=f"{context}.{field}")
    raise DatasetFormatError(
        f"{context}.file_names must include a video directory (for example, video_id/frame.jpg)"
    )


def load_video_specs(layout: SplitLayout) -> list[VideoSpec]:
    """Load and strictly validate video metadata from one split JSON file."""

    try:
        with layout.annotation_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"Invalid JSON in {layout.annotation_file}: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("videos"), list):
        raise DatasetFormatError(f"{layout.annotation_file} must contain a top-level 'videos' list")

    specs: list[VideoSpec] = []
    seen_keys: set[str] = set()
    for index, raw_video in enumerate(payload["videos"]):
        context = f"{layout.annotation_file}:videos[{index}]"
        if not isinstance(raw_video, dict):
            raise DatasetFormatError(f"{context} must be an object")

        length = raw_video.get("length")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise DatasetFormatError(f"{context}.length must be a positive integer")

        raw_file_names = raw_video.get("file_names")
        if not isinstance(raw_file_names, list) or not raw_file_names:
            raise DatasetFormatError(f"{context}.file_names must be a non-empty list")
        file_names = tuple(
            _safe_annotation_path(value, context=f"{context}.file_names[{frame_index}]")
            for frame_index, value in enumerate(raw_file_names)
        )
        if len(file_names) != length:
            raise DatasetFormatError(
                f"{context} declares length={length}, but has {len(file_names)} frame names"
            )

        key = _video_key(raw_video, file_names, context)
        if key in seen_keys:
            raise DatasetFormatError(f"Duplicate video directory {key!r} in {layout.annotation_file}")
        seen_keys.add(key)
        specs.append(VideoSpec(layout.name, key, length, file_names))

    if not specs:
        raise DatasetFormatError(f"{layout.annotation_file} contains no videos")
    return specs


def load_dataset(
    dataset_root: Path,
    splits: Iterable[str],
    directories: DirectoryNames,
) -> list[tuple[SplitLayout, list[VideoSpec]]]:
    """Resolve requested splits and load their video metadata."""

    result: list[tuple[SplitLayout, list[VideoSpec]]] = []
    seen: set[str] = set()
    for split in splits:
        split = validate_split_name(split)
        if split in seen:
            raise ValueError(f"Split {split!r} was provided more than once")
        seen.add(split)
        layout = make_split_layout(dataset_root, split, directories)
        result.append((layout, load_video_specs(layout)))
    if not result:
        raise ValueError("At least one split is required")
    return result


def source_wav_paths(layout: SplitLayout, video: VideoSpec) -> tuple[Path, ...]:
    directory = layout.separated_audio_root / video.key
    return tuple(directory / SOURCE_WAV_TEMPLATE.format(index=index) for index in range(1, NUM_SOURCES + 1))


def source_feature_paths(layout: SplitLayout, video: VideoSpec) -> tuple[Path, ...]:
    directory = layout.feature_root / video.key
    return tuple(
        directory / SOURCE_FEATURE_TEMPLATE.format(index=index)
        for index in range(1, NUM_SOURCES + 1)
    )

