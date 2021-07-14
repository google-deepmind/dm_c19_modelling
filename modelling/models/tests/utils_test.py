# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for `utils.py`."""

from absl.testing import absltest
from dm_c19_modelling.modelling import definitions
from dm_c19_modelling.modelling.models import utils
import numpy as np


class RolloutFeaturesTest(absltest.TestCase):

  def test_expected_rollout(self):

    feature_names = ["feature1", definitions.SITE_ID_INTEGER, "feature2",
                     "feature3", definitions.WEEK_DAY_INTEGER]

    target_names = ["feature3", "feature1"]
    constant_features = ["feature2"]
    cadence = 2

    features = np.array([
        # First date. Day #2.
        [
            # First site.
            [10.1, 25., 30., 40.1, 2],
            # Second site.
            [10.2, 27., 30., 40.2, 2],
        ],
        # Second date. Day #4.
        [
            # First site.
            [11.1, 25., 30., 41.1, 4],
            # Second site.
            [11.2, 27., 30., 41.2, 4],
        ],
    ])

    next_steps_targets = np.array([
        # Third date.  Day #6.
        [
            # First site.
            [42.1, 12.1],
            # Second site.
            [42.2, 12.2],
        ],
        # Fourth date. Day #8.
        [
            # First site.
            [43.1, 13.1],
            # Second site.
            [43.2, 13.2],
        ],
    ])

    output = utils.rollout_features_with_predictions(
        features=features,
        next_steps_targets=next_steps_targets,
        feature_names=feature_names,
        target_names=target_names,
        cadence=cadence,
        constant_features=constant_features)

    expected_additional_features = np.array([
        # Third date. Day #6.
        [
            # First site.
            [12.1, 25., 30., 42.1, 6],
            # Second site.
            [12.2, 27., 30., 42.2, 6],
        ],
        # Fourth date. Day #8.
        [
            # First site.
            [13.1, 25., 30., 43.1, 1],
            # Second site.
            [13.2, 27., 30., 43.2, 1],
        ],
    ])
    expected_output = np.concatenate(
        [features, expected_additional_features], axis=0)

    np.testing.assert_allclose(output, expected_output)


if __name__ == "__main__":
  absltest.main()
