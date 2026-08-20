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
"""NumPy mel-spectrogram frontend from the TensorFlow VGGish release."""

from __future__ import annotations

import numpy as np


def frame(data: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
    """Convert an array into complete, possibly overlapping frames."""

    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    if num_frames <= 0:
        return np.empty((0, window_length) + data.shape[1:], dtype=data.dtype)
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length: int) -> np.ndarray:
    """Return a periodic Hann window of the requested length."""

    return 0.5 - 0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length))


def stft_magnitude(
    signal: np.ndarray,
    fft_length: int,
    hop_length: int,
    window_length: int,
) -> np.ndarray:
    """Calculate the short-time Fourier-transform magnitude."""

    frames = frame(signal, window_length, hop_length)
    windowed_frames = frames * periodic_hann(window_length)
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz: np.ndarray | float) -> np.ndarray | float:
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ
    )


def spectrogram_to_mel_matrix(
    num_mel_bins: int = 20,
    num_spectrogram_bins: int = 129,
    audio_sample_rate: int = 8000,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 3800.0,
) -> np.ndarray:
    """Return the matrix mapping magnitude spectrogram bins to mel bins."""

    nyquist_hertz = audio_sample_rate / 2.0
    if lower_edge_hertz < 0.0:
        raise ValueError(f"lower_edge_hertz {lower_edge_hertz:.1f} must be >= 0")
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(
            f"lower_edge_hertz {lower_edge_hertz:.1f} >= upper_edge_hertz {upper_edge_hertz:.1f}"
        )
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError(
            f"upper_edge_hertz {upper_edge_hertz:.1f} is greater than Nyquist {nyquist_hertz:.1f}"
        )

    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz),
        hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2,
    )
    weights = np.empty((num_spectrogram_bins, num_mel_bins))
    for index in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[index : index + 3]
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (
            center_mel - lower_edge_mel
        )
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (
            upper_edge_mel - center_mel
        )
        weights[:, index] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    weights[0, :] = 0.0
    return weights


def log_mel_spectrogram(
    data: np.ndarray,
    audio_sample_rate: int = 8000,
    log_offset: float = 0.0,
    window_length_secs: float = 0.025,
    hop_length_secs: float = 0.010,
    **kwargs: float | int,
) -> np.ndarray:
    """Convert a waveform to a stabilized log-mel spectrogram."""

    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    spectrogram = stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples,
    )
    mel_spectrogram = np.dot(
        spectrogram,
        spectrogram_to_mel_matrix(
            num_spectrogram_bins=spectrogram.shape[1],
            audio_sample_rate=audio_sample_rate,
            **kwargs,
        ),
    )
    return np.log(mel_spectrogram + log_offset)

