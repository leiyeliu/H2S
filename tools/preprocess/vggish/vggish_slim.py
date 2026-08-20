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
"""TensorFlow-Slim definition of the VGGish embedding network."""

from __future__ import annotations

import tensorflow.compat.v1 as tf
import tf_slim as slim

from . import vggish_params as params


def define_vggish_slim(training: bool = False):
    """Define VGGish in the current graph and return its 128-D embedding."""

    with (
        slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_initializer=tf.truncated_normal_initializer(stddev=params.INIT_STDDEV),
            biases_initializer=tf.zeros_initializer(),
            activation_fn=tf.nn.relu,
            trainable=training,
        ),
        slim.arg_scope([slim.conv2d], kernel_size=[3, 3], stride=1, padding="SAME"),
        slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=2, padding="SAME"),
        tf.variable_scope("vggish"),
    ):
        features = tf.placeholder(
            tf.float32,
            shape=(None, params.NUM_FRAMES, params.NUM_BANDS),
            name="input_features",
        )
        network = tf.reshape(features, [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])
        network = slim.conv2d(network, 64, scope="conv1")
        network = slim.max_pool2d(network, scope="pool1")
        network = slim.conv2d(network, 128, scope="conv2")
        network = slim.max_pool2d(network, scope="pool2")
        network = slim.repeat(network, 2, slim.conv2d, 256, scope="conv3")
        network = slim.max_pool2d(network, scope="pool3")
        network = slim.repeat(network, 2, slim.conv2d, 512, scope="conv4")
        network = slim.max_pool2d(network, scope="pool4")
        network = slim.flatten(network)
        network = slim.repeat(network, 2, slim.fully_connected, 4096, scope="fc1")
        network = slim.fully_connected(network, params.EMBEDDING_SIZE, scope="fc2")
        return tf.identity(network, name="embedding")


def load_vggish_slim_checkpoint(session, checkpoint_path: str) -> None:
    """Restore the VGGish variables in the active graph from a checkpoint."""

    variables = tf.global_variables(scope="vggish")
    saver = tf.train.Saver(variables, name="vggish_load_pretrained")
    saver.restore(session, checkpoint_path)

