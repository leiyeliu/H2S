# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MixIT inference helpers adapted for the H2S offline preprocessing pipeline.

Only inference-graph loading and waveform separation live here. The upstream
MixIT training repository and checkpoints are intentionally not vendored.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .common import NUM_SOURCES, SAMPLE_RATE


DEFAULT_INPUT_TENSOR = "input_audio/receiver_audio:0"
DEFAULT_OUTPUT_TENSOR = "denoised_waveforms:0"


def configure_tensorflow_device(device: str) -> None:
    """Configure TensorFlow visibility before TensorFlow is imported."""

    normalized = device.strip().lower()
    if normalized == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return
    if normalized.startswith("cuda:") and normalized[5:].isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = normalized[5:]
        return
    raise ValueError("--device must be 'cpu' or 'cuda:<index>'")


def checkpoint_exists(checkpoint: Path) -> bool:
    """Return whether a TensorFlow checkpoint prefix has an index file."""

    checkpoint = checkpoint.expanduser()
    return checkpoint.is_file() or Path(f"{checkpoint}.index").is_file()


def read_mixture(
    audio_file: Path,
    sample_rate: int = SAMPLE_RATE,
    peak: float = 0.99,
) -> np.ndarray:
    """Load the first channel, resample it, and apply global peak normalization.

    This matches the preprocessing used to generate the H2S paper features:
    ``librosa.load(..., mono=False, sr=16000)``, first-channel selection, then
    scaling the peak absolute amplitude to 0.99.
    """

    if not 0.0 < peak <= 1.0:
        raise ValueError(f"peak must be in (0, 1], got {peak}")
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "librosa is required for MixIT input loading; install requirements/preprocess.txt"
        ) from exc

    waveform, loaded_rate = librosa.load(
        str(audio_file),
        sr=sample_rate,
        mono=False,
        dtype=np.float32,
    )
    if loaded_rate != sample_rate:
        raise RuntimeError(f"librosa returned {loaded_rate} Hz for {audio_file}, expected {sample_rate} Hz")
    if waveform.ndim == 2:
        waveform = waveform[0]
    elif waveform.ndim != 1:
        raise ValueError(f"Unsupported waveform shape {waveform.shape} in {audio_file}")
    if waveform.size == 0:
        raise ValueError(f"Empty audio file: {audio_file}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"Non-finite samples in {audio_file}")

    maximum = float(np.max(np.abs(waveform)))
    if maximum > 0.0:
        waveform = waveform * np.float32(peak / maximum)
    return np.ascontiguousarray(waveform, dtype=np.float32)


class MixITSeparator:
    """One persistent TensorFlow v1 session for a MixIT inference graph."""

    def __init__(
        self,
        checkpoint: Path,
        metagraph: Path,
        *,
        device: str,
        input_tensor_name: str = DEFAULT_INPUT_TENSOR,
        output_tensor_name: str = DEFAULT_OUTPUT_TENSOR,
    ) -> None:
        checkpoint = checkpoint.expanduser().resolve()
        metagraph = metagraph.expanduser().resolve()
        if not checkpoint_exists(checkpoint):
            raise FileNotFoundError(f"MixIT checkpoint prefix not found: {checkpoint}")
        if not metagraph.is_file():
            raise FileNotFoundError(f"MixIT metagraph not found: {metagraph}")

        configure_tensorflow_device(device)
        try:
            import tensorflow.compat.v1 as tf
        except ImportError as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError(
                "TensorFlow is required for MixIT inference; install requirements/preprocess.txt"
            ) from exc

        tf.disable_v2_behavior()
        self._tf = tf
        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        try:
            with self.graph.as_default():
                saver = tf.train.import_meta_graph(str(metagraph), clear_devices=True)
                saver.restore(self.session, str(checkpoint))
                self.input_tensor = self.graph.get_tensor_by_name(input_tensor_name)
                self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)
        except Exception:
            self.close()
            raise

    def separate(self, mono_waveform: np.ndarray) -> np.ndarray:
        """Separate one mono waveform and return exactly eight sources."""

        if mono_waveform.ndim != 1 or mono_waveform.size == 0:
            raise ValueError(f"Expected a non-empty mono waveform, got {mono_waveform.shape}")
        model_input = mono_waveform[np.newaxis, np.newaxis, :]
        separated = np.asarray(
            self.session.run(self.output_tensor, feed_dict={self.input_tensor: model_input})
        )
        if separated.ndim == 3:
            if separated.shape[0] != 1:
                raise ValueError(f"Unexpected MixIT batch dimension: {separated.shape}")
            separated = separated[0]
        if separated.ndim != 2:
            raise ValueError(f"Unexpected MixIT output shape: {separated.shape}")
        if separated.shape[0] < NUM_SOURCES:
            raise ValueError(
                f"MixIT returned only {separated.shape[0]} sources; H2S requires {NUM_SOURCES}"
            )
        separated = separated[:NUM_SOURCES]
        if separated.shape[1] == 0 or not np.isfinite(separated).all():
            raise ValueError("MixIT returned an empty or non-finite waveform")
        return np.asarray(separated, dtype=np.float32)

    def close(self) -> None:
        session = getattr(self, "session", None)
        if session is not None:
            session.close()
            self.session = None

    def __enter__(self) -> "MixITSeparator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def write_sources_atomic(
    output_directory: Path,
    separated: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[Path, ...]:
    """Write eight mono PCM16 WAV files using same-directory atomic replaces."""

    if separated.ndim != 2 or separated.shape[0] != NUM_SOURCES:
        raise ValueError(f"Expected [{NUM_SOURCES}, samples], got {separated.shape}")
    try:
        from scipy.io import wavfile
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "SciPy is required to write separated WAV files; install requirements/preprocess.txt"
        ) from exc

    output_directory.mkdir(parents=True, exist_ok=True)
    final_paths = tuple(
        output_directory / f"source_{source_index:02d}.wav"
        for source_index in range(1, NUM_SOURCES + 1)
    )
    temporary_paths: list[Path] = []
    try:
        for source, final_path in zip(separated, final_paths):
            temporary_path = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")
            temporary_paths.append(temporary_path)
            pcm16 = np.int16(np.clip(source, -1.0, 1.0) * 32767)
            wavfile.write(str(temporary_path), sample_rate, pcm16)
            if temporary_path.stat().st_size <= 44:
                raise OSError(f"Separated WAV is empty: {temporary_path}")
        for temporary_path, final_path in zip(temporary_paths, final_paths):
            os.replace(temporary_path, final_path)
        return final_paths
    finally:
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

