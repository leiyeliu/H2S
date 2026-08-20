# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert waveforms into VGGish log-mel examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from . import mel_features
from . import vggish_params as params


def waveform_to_examples(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert one waveform into the standard VGGish example tensor."""

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sample_rate != params.SAMPLE_RATE:
        try:
            import resampy
        except ImportError as exc:  # pragma: no cover - user environment
            raise RuntimeError("resampy is required for non-16-kHz VGGish input") from exc
        data = resampy.resample(data, sample_rate, params.SAMPLE_RATE)

    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=params.SAMPLE_RATE,
        log_offset=params.LOG_OFFSET,
        window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=params.NUM_MEL_BINS,
        lower_edge_hertz=params.MEL_MIN_HZ,
        upper_edge_hertz=params.MEL_MAX_HZ,
    )
    feature_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS
    example_window = int(round(params.EXAMPLE_WINDOW_SECONDS * feature_rate))
    example_hop = int(round(params.EXAMPLE_HOP_SECONDS * feature_rate))
    return mel_features.frame(log_mel, example_window, example_hop)


def wavfile_to_examples(wav_file: Path, num_seconds: int) -> np.ndarray:
    """Create exactly ``num_seconds`` VGGish examples from a mono PCM16 WAV.

    The output time dimension comes from the split JSON, not from audio duration.
    Audio longer than that value is truncated. For a final partial second that
    is too short to form a 0.96-second VGGish example, the historical pipeline
    leaves the corresponding log-mel input at zero.
    """

    sample_rate, sound = wavfile.read(str(wav_file))
    if sample_rate != params.SAMPLE_RATE:
        raise ValueError(
            f"Expected {params.SAMPLE_RATE} Hz separated audio, got {sample_rate} Hz: {wav_file}"
        )
    if sound.ndim != 1:
        raise ValueError(f"Expected mono separated audio, got shape {sound.shape}: {wav_file}")
    if sound.dtype != np.int16:
        raise ValueError(f"Expected PCM16 separated audio, got {sound.dtype}: {wav_file}")

    desired_samples = sample_rate * num_seconds
    waveform = sound[:desired_samples].astype(np.float64) / 32768.0

    examples = np.zeros((num_seconds, params.NUM_FRAMES, params.NUM_BANDS), dtype=np.float64)
    for second in range(num_seconds):
        start = second * sample_rate
        end = start + sample_rate
        second_examples = waveform_to_examples(waveform[start:end], sample_rate)
        if second_examples.shape[0] > 0:
            examples[second] = second_examples[0]
    return examples
