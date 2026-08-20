#!/usr/bin/env python3
"""Unified MixIT -> VGGish preprocessing CLI for H2S."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.preprocess.common import (  # type: ignore[import-not-found]
        DirectoryNames,
        SAMPLE_RATE,
        load_dataset,
        safe_relative_directory,
        source_feature_paths,
        source_wav_paths,
    )
else:
    from .common import (
        DirectoryNames,
        SAMPLE_RATE,
        load_dataset,
        safe_relative_directory,
        source_feature_paths,
        source_wav_paths,
    )


LOGGER = logging.getLogger("h2s.preprocess")


@dataclass
class WorkSummary:
    label: str
    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def log(self) -> None:
        LOGGER.info(
            "%s summary: total=%d, succeeded=%d, resumed=%d, failed=%d",
            self.label,
            self.total,
            self.succeeded,
            self.skipped,
            self.failed,
        )


def _add_layout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="AVIS dataset root containing split directories and split JSON files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        metavar="SPLIT",
        help="One or more split names, for example: train val test.",
    )
    parser.add_argument(
        "--raw-audio-dir",
        default="WAVAudios",
        help="Original-WAV directory relative to each split (default: WAVAudios).",
    )
    parser.add_argument(
        "--separated-audio-dir",
        default="WAVAudios_sep",
        help="Separated-WAV directory relative to each split (default: WAVAudios_sep).",
    )
    parser.add_argument(
        "--feature-dir",
        default="FEATAudios_sep",
        help="VGGish feature directory relative to each split (default: FEATAudios_sep).",
    )


def _add_resume_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute valid existing outputs. Invalid/incomplete outputs are always repaired.",
    )


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="TensorFlow device visibility: cpu or cuda:<index> (default: cuda:0).",
    )


def _add_mixit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mixit-checkpoint",
        type=Path,
        required=True,
        help="MixIT TensorFlow checkpoint prefix (omit .index/.data suffixes).",
    )
    parser.add_argument(
        "--mixit-metagraph",
        type=Path,
        required=True,
        help="MixIT inference.meta path.",
    )
    parser.add_argument(
        "--mixit-input-tensor",
        default="input_audio/receiver_audio:0",
        help="Inference graph input tensor name.",
    )
    parser.add_argument(
        "--mixit-output-tensor",
        default="denoised_waveforms:0",
        help="Inference graph output tensor name.",
    )
    parser.add_argument(
        "--input-peak",
        type=float,
        default=0.99,
        help="Peak normalization applied before separation (default: 0.99).",
    )


def _add_vggish_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vggish-checkpoint",
        type=Path,
        required=True,
        help="VGGish TensorFlow checkpoint prefix (omit .index/.data suffixes).",
    )
    parser.add_argument(
        "--vggish-batch-size",
        type=int,
        default=64,
        help="Maximum number of one-second examples per VGGish inference call (default: 64).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare H2S audio inputs with a reproducible 16-kHz MixIT -> VGGish pipeline. "
            "All dataset paths are derived from --dataset-root."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    separate = subparsers.add_parser("separate", help="Create eight 16-kHz PCM16 sources per video.")
    _add_layout_arguments(separate)
    _add_resume_argument(separate)
    _add_device_argument(separate)
    _add_mixit_arguments(separate)

    extract = subparsers.add_parser("extract", help="Extract [T, 128] VGGish features for each source.")
    _add_layout_arguments(extract)
    _add_resume_argument(extract)
    _add_device_argument(extract)
    _add_vggish_arguments(extract)

    validate = subparsers.add_parser(
        "validate",
        help="Validate JSON, JPEGs, WAVs, eight separated sources, and features.",
    )
    _add_layout_arguments(validate)
    validate.add_argument(
        "--max-issues",
        type=int,
        default=100,
        help="Maximum individual validation issues to print (default: 100).",
    )

    all_steps = subparsers.add_parser("all", help="Run separate, extract, then strict validation.")
    _add_layout_arguments(all_steps)
    _add_resume_argument(all_steps)
    _add_device_argument(all_steps)
    _add_mixit_arguments(all_steps)
    _add_vggish_arguments(all_steps)
    all_steps.add_argument(
        "--max-issues",
        type=int,
        default=100,
        help="Maximum individual validation issues to print (default: 100).",
    )
    return parser


def _directories(args: argparse.Namespace) -> DirectoryNames:
    return DirectoryNames(
        raw_audio=safe_relative_directory(args.raw_audio_dir, "--raw-audio-dir"),
        separated_audio=safe_relative_directory(
            args.separated_audio_dir, "--separated-audio-dir"
        ),
        features=safe_relative_directory(args.feature_dir, "--feature-dir"),
    )


def _load_requested_dataset(args: argparse.Namespace):
    dataset = load_dataset(args.dataset_root, args.splits, _directories(args))
    for layout, videos in dataset:
        LOGGER.info(
            "Loaded split %s: %d videos from %s",
            layout.name,
            len(videos),
            layout.annotation_file,
        )
    return dataset


def run_separate(args: argparse.Namespace, dataset) -> WorkSummary:
    """Run resumable MixIT separation for every requested video."""

    if __package__ in {None, ""}:
        from tools.preprocess.mixit import MixITSeparator, read_mixture, write_sources_atomic
        from tools.preprocess.validation import separated_set_is_valid
    else:
        from .mixit import MixITSeparator, read_mixture, write_sources_atomic
        from .validation import separated_set_is_valid

    summary = WorkSummary("separation")
    pending = []
    for layout, videos in dataset:
        for video in videos:
            summary.total += 1
            if not args.overwrite and separated_set_is_valid(layout, video):
                summary.skipped += 1
                continue
            raw_audio = layout.raw_audio_root / f"{video.key}.wav"
            if not raw_audio.is_file():
                LOGGER.error("[%s/%s] Missing original WAV: %s", layout.name, video.key, raw_audio)
                summary.failed += 1
                continue
            pending.append((layout, video, raw_audio))

    if not pending:
        summary.log()
        return summary

    with MixITSeparator(
        args.mixit_checkpoint,
        args.mixit_metagraph,
        device=args.device,
        input_tensor_name=args.mixit_input_tensor,
        output_tensor_name=args.mixit_output_tensor,
    ) as separator:
        for item_index, (layout, video, raw_audio) in enumerate(pending, 1):
            LOGGER.info(
                "Separating %d/%d: [%s/%s]",
                item_index,
                len(pending),
                layout.name,
                video.key,
            )
            try:
                mixture = read_mixture(raw_audio, sample_rate=SAMPLE_RATE, peak=args.input_peak)
                separated = separator.separate(mixture)
                write_sources_atomic(layout.separated_audio_root / video.key, separated, SAMPLE_RATE)
                summary.succeeded += 1
            except Exception:
                LOGGER.exception("[%s/%s] MixIT separation failed", layout.name, video.key)
                summary.failed += 1
    summary.log()
    return summary


def run_extract(args: argparse.Namespace, dataset) -> WorkSummary:
    """Run resumable VGGish extraction with one persistent model session."""

    if __package__ in {None, ""}:
        from tools.preprocess.extraction import (
            VGGishExtractor,
            feature_file_is_valid,
            save_feature_atomic,
        )
        from tools.preprocess.validation import inspect_wave
    else:
        from .extraction import VGGishExtractor, feature_file_is_valid, save_feature_atomic
        from .validation import inspect_wave

    summary = WorkSummary("feature extraction")
    pending = []
    for layout, videos in dataset:
        for video in videos:
            for source_path, feature_path in zip(
                source_wav_paths(layout, video),
                source_feature_paths(layout, video),
            ):
                summary.total += 1
                if not args.overwrite and feature_file_is_valid(feature_path, video.length):
                    summary.skipped += 1
                    continue
                source_error = inspect_wave(
                    source_path,
                    expected_rate=SAMPLE_RATE,
                    expected_channels=1,
                    expected_sample_width=2,
                )
                if source_error:
                    LOGGER.error(
                        "[%s/%s] Invalid separated source %s: %s",
                        layout.name,
                        video.key,
                        source_path,
                        source_error,
                    )
                    summary.failed += 1
                    continue
                pending.append((layout, video, source_path, feature_path))

    if not pending:
        summary.log()
        return summary

    with VGGishExtractor(
        args.vggish_checkpoint,
        device=args.device,
        batch_size=args.vggish_batch_size,
    ) as extractor:
        for item_index, (layout, video, source_path, feature_path) in enumerate(pending, 1):
            LOGGER.info(
                "Extracting %d/%d: [%s/%s] %s",
                item_index,
                len(pending),
                layout.name,
                video.key,
                source_path.name,
            )
            try:
                features = extractor.extract(source_path, video.length)
                save_feature_atomic(feature_path, features)
                summary.succeeded += 1
            except Exception:
                LOGGER.exception(
                    "[%s/%s] VGGish extraction failed for %s",
                    layout.name,
                    video.key,
                    source_path,
                )
                summary.failed += 1
    summary.log()
    return summary


def run_validate(args: argparse.Namespace, dataset) -> bool:
    """Run strict validation and print a bounded, actionable report."""

    if __package__ in {None, ""}:
        from tools.preprocess.validation import validate_dataset
    else:
        from .validation import validate_dataset

    report = validate_dataset(dataset, max_issues=args.max_issues)
    for issue in report.issues:
        LOGGER.error(
            "[%s/%s] %s: %s (%s)",
            issue.split,
            issue.video,
            issue.kind,
            issue.path,
            issue.message,
        )
    if report.omitted_issues:
        LOGGER.error("... %d additional validation issues omitted", report.omitted_issues)
    LOGGER.info(
        "Validation summary: videos=%d, JPEGs=%d, original WAVs=%d, "
        "separated WAVs=%d, features=%d, issues=%d",
        report.checked_videos,
        report.checked_frames,
        report.checked_raw_audio,
        report.checked_sources,
        report.checked_features,
        report.issue_count,
    )
    return report.ok


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        dataset = _load_requested_dataset(args)
        if args.command == "separate":
            return 0 if run_separate(args, dataset).ok else 1
        if args.command == "extract":
            return 0 if run_extract(args, dataset).ok else 1
        if args.command == "validate":
            return 0 if run_validate(args, dataset) else 1
        if args.command == "all":
            separation = run_separate(args, dataset)
            extraction = run_extract(args, dataset)
            validation_ok = run_validate(args, dataset)
            return 0 if separation.ok and extraction.ok and validation_ok else 1
        parser.error(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130
    except Exception:
        LOGGER.exception("Audio preprocessing failed")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

