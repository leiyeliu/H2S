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
"""VGGish architecture and frontend parameters."""

NUM_FRAMES = 96
NUM_BANDS = 64
EMBEDDING_SIZE = 128

SAMPLE_RATE = 16_000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01
EXAMPLE_WINDOW_SECONDS = 0.96
EXAMPLE_HOP_SECONDS = 0.96

INIT_STDDEV = 0.01

INPUT_TENSOR_NAME = "vggish/input_features:0"
OUTPUT_TENSOR_NAME = "vggish/embedding:0"

