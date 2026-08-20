"""Strict, weight-free validation of H2S audio preprocessing inputs/outputs."""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import (
    NUM_SOURCES,
    SAMPLE_RATE,
    SplitLayout,
    VideoSpec,
    source_feature_paths,
    source_wav_paths,
)


@dataclass(frozen=True)
class ValidationIssue:
    split: str
    video: str
    kind: str
    path: Path
    message: str


@dataclass
class ValidationReport:
    checked_videos: int = 0
    checked_frames: int = 0
    checked_raw_audio: int = 0
    checked_sources: int = 0
    checked_features: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    omitted_issues: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues and self.omitted_issues == 0

    @property
    def issue_count(self) -> int:
        return len(self.issues) + self.omitted_issues


def inspect_wave(
    path: Path,
    *,
    expected_rate: int | None = None,
    expected_channels: int | None = None,
    expected_sample_width: int | None = None,
) -> str | None:
    """Return an error message for an invalid WAV, otherwise ``None``."""

    if not path.is_file():
        return "file is missing"
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            compression = handle.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        return f"cannot read WAV header: {exc}"
    if compression != "NONE":
        return f"must be uncompressed PCM, got {compression}"
    if channels <= 0 or sample_width <= 0 or rate <= 0 or frames <= 0:
        return (
            "invalid WAV metadata "
            f"(channels={channels}, sample_width={sample_width}, rate={rate}, frames={frames})"
        )
    if expected_rate is not None and rate != expected_rate:
        return f"expected {expected_rate} Hz, got {rate} Hz"
    if expected_channels is not None and channels != expected_channels:
        return f"expected {expected_channels} channel, got {channels}"
    if expected_sample_width is not None and sample_width != expected_sample_width:
        return f"expected {expected_sample_width * 8}-bit PCM, got {sample_width * 8}-bit"
    return None


def inspect_raw_wave(path: Path) -> str | None:
    """Check that an original dataset WAV is readable and non-empty.

    AVIS includes ``WAVE_FORMAT_EXTENSIBLE`` files, which Python's standard
    :mod:`wave` module rejects even though SciPy and the MixIT input loader can
    read them. Original audio therefore gets a permissive container check;
    strict mono/16-kHz/PCM16 checks remain reserved for generated sources.
    """

    if not path.is_file():
        return "file is missing"
    try:
        import soundfile

        metadata = soundfile.info(str(path))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return f"cannot read WAV: {exc}"
    if metadata.samplerate <= 0:
        return f"invalid sample rate: {metadata.samplerate}"
    if metadata.channels <= 0 or metadata.frames <= 0:
        return (
            "invalid or empty WAV metadata: "
            f"channels={metadata.channels}, frames={metadata.frames}"
        )
    return None


def inspect_feature(path: Path, expected_length: int) -> str | None:
    """Return an error for a malformed/non-finite VGGish .npy file."""

    if not path.is_file():
        return "file is missing"
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        return f"cannot load NumPy array: {exc}"
    if array.shape != (expected_length, 128):
        return f"expected shape ({expected_length}, 128), got {array.shape}"
    if not np.issubdtype(array.dtype, np.number):
        return f"expected a numeric dtype, got {array.dtype}"
    if not np.isfinite(array).all():
        return "contains NaN or Inf"
    return None


def separated_set_is_valid(layout: SplitLayout, video: VideoSpec) -> bool:
    """Fast resume check for all eight separated PCM16 WAV files."""

    paths = source_wav_paths(layout, video)
    return len(paths) == NUM_SOURCES and all(
        inspect_wave(
            path,
            expected_rate=SAMPLE_RATE,
            expected_channels=1,
            expected_sample_width=2,
        )
        is None
        for path in paths
    )


def _add_issue(
    report: ValidationReport,
    issue: ValidationIssue,
    max_issues: int,
) -> None:
    if len(report.issues) < max_issues:
        report.issues.append(issue)
    else:
        report.omitted_issues += 1


def validate_split(
    layout: SplitLayout,
    videos: list[VideoSpec],
    report: ValidationReport,
    *,
    max_issues: int,
) -> None:
    """Validate annotations, JPEGs, original WAVs, sources, and features."""

    for video in videos:
        report.checked_videos += 1
        raw_audio = layout.raw_audio_root / f"{video.key}.wav"
        report.checked_raw_audio += 1
        raw_error = inspect_raw_wave(raw_audio)
        if raw_error:
            _add_issue(
                report,
                ValidationIssue(layout.name, video.key, "raw-audio", raw_audio, raw_error),
                max_issues,
            )

        for file_name in video.file_names:
            frame = layout.image_root / file_name
            report.checked_frames += 1
            if not frame.is_file():
                _add_issue(
                    report,
                    ValidationIssue(layout.name, video.key, "jpeg", frame, "file is missing"),
                    max_issues,
                )

        for source_path in source_wav_paths(layout, video):
            report.checked_sources += 1
            source_error = inspect_wave(
                source_path,
                expected_rate=SAMPLE_RATE,
                expected_channels=1,
                expected_sample_width=2,
            )
            if source_error:
                _add_issue(
                    report,
                    ValidationIssue(
                        layout.name,
                        video.key,
                        "separated-audio",
                        source_path,
                        source_error,
                    ),
                    max_issues,
                )

        for feature_path in source_feature_paths(layout, video):
            report.checked_features += 1
            feature_error = inspect_feature(feature_path, video.length)
            if feature_error:
                _add_issue(
                    report,
                    ValidationIssue(
                        layout.name,
                        video.key,
                        "feature",
                        feature_path,
                        feature_error,
                    ),
                    max_issues,
                )


def validate_dataset(
    dataset: list[tuple[SplitLayout, list[VideoSpec]]],
    *,
    max_issues: int = 100,
) -> ValidationReport:
    """Validate all requested splits without loading either neural network."""

    if max_issues <= 0:
        raise ValueError("max_issues must be positive")
    report = ValidationReport()
    for layout, videos in dataset:
        validate_split(layout, videos, report, max_issues=max_issues)
    return report
