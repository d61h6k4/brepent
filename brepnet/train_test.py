# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>. All Rights Reserved.
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for brepnet.train"""

import jax

import jax.numpy as jnp
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized

from brepnet import train


class TrainTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # Make sure tf does not allocate gpu memory.
        tf.config.experimental.set_visible_devices([], "GPU")

        # Print the current platform (the default device).
        platform = jax.local_devices()[0].platform
        print("Running on platform:", platform.upper())

    @parameterized.parameters(
        dict(loss=[[0.5, 1.], [1.5, 1.3], [2., 1.2]],
             logits=[[-1., 1.], [1., 1.], [2., 0.]],
             labels=[[0, jnp.nan], [1, 0], [0, 1]],
             mask=[[True, False], [True, True], [False, False]],
             expected_results={
                 "loss": 1.1,
                 "accuracy": 2 / 3
             }), )
    def test_train_metrics(self, loss, logits, labels, mask, expected_results):
        loss = jnp.asarray(loss)
        logits = jnp.asarray(logits)
        labels = jnp.asarray(labels)
        mask = jnp.asarray(mask)

        train_metrics = train.TrainMetrics.single_from_model_output(
            loss=loss, logits=logits, labels=labels, mask=mask).compute()
        for metric in expected_results:
            self.assertAlmostEqual(expected_results[metric],
                                   train_metrics[metric])


if __name__ == "__main__":
    absltest.main()
