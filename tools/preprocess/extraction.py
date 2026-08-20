"""Persistent VGGish feature extraction and atomic NumPy output."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .common import SAMPLE_RATE
from .mixit import checkpoint_exists, configure_tensorflow_device
from .vggish import vggish_input
from .vggish import vggish_params as params


def feature_file_is_valid(path: Path, length: int) -> bool:
    """Return whether a feature file satisfies the complete H2S contract."""

    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            array.shape == (length, params.EMBEDDING_SIZE)
            and np.issubdtype(array.dtype, np.number)
            and bool(np.isfinite(array).all())
        )
    except (OSError, TypeError, ValueError, EOFError):
        return False


def save_feature_atomic(path: Path, features: np.ndarray) -> None:
    """Write one .npy file completely before atomically replacing its target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, features, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class VGGishExtractor:
    """One graph, session, and checkpoint restore shared by every source file."""

    def __init__(self, checkpoint: Path, *, device: str, batch_size: int = 64) -> None:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint_exists(checkpoint):
            raise FileNotFoundError(f"VGGish checkpoint prefix not found: {checkpoint}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        configure_tensorflow_device(device)
        try:
            import tensorflow.compat.v1 as tf
        except ImportError as exc:  # pragma: no cover - user environment
            raise RuntimeError(
                "TensorFlow is required for VGGish inference; install requirements/preprocess.txt"
            ) from exc
        try:
            from .vggish import vggish_slim
        except ImportError as exc:  # pragma: no cover - user environment
            raise RuntimeError(
                "tf-slim is required for VGGish inference; install requirements/preprocess.txt"
            ) from exc

        tf.disable_v2_behavior()
        self._tf = tf
        self.batch_size = batch_size
        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        try:
            with self.graph.as_default():
                vggish_slim.define_vggish_slim(training=False)
                self.input_tensor = self.graph.get_tensor_by_name(params.INPUT_TENSOR_NAME)
                self.output_tensor = self.graph.get_tensor_by_name(params.OUTPUT_TENSOR_NAME)
                vggish_slim.load_vggish_slim_checkpoint(self.session, str(checkpoint))
        except Exception:
            self.close()
            raise

    def extract(self, wav_file: Path, length: int) -> np.ndarray:
        """Extract a finite ``[length, 128]`` embedding matrix."""

        input_examples = vggish_input.wavfile_to_examples(wav_file, length)
        if input_examples.shape != (length, params.NUM_FRAMES, params.NUM_BANDS):
            raise ValueError(f"Unexpected VGGish input shape {input_examples.shape} for {wav_file}")

        batches: list[np.ndarray] = []
        for start in range(0, length, self.batch_size):
            batch = input_examples[start : start + self.batch_size]
            embedding = self.session.run(
                self.output_tensor,
                feed_dict={self.input_tensor: batch},
            )
            batches.append(np.asarray(embedding, dtype=np.float32))
        features = np.concatenate(batches, axis=0)
        if features.shape != (length, params.EMBEDDING_SIZE):
            raise ValueError(f"Unexpected VGGish output shape {features.shape} for {wav_file}")
        if not np.isfinite(features).all():
            raise ValueError(f"VGGish produced non-finite features for {wav_file}")
        return features

    def close(self) -> None:
        session = getattr(self, "session", None)
        if session is not None:
            session.close()
            self.session = None

    def __enter__(self) -> "VGGishExtractor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
