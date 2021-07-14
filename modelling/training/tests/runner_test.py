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
"""Tests for `runner.py`."""

import datetime

import tempfile
from unittest import mock

from absl import flags

from absl.testing import absltest
from absl.testing import parameterized
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import dataset_factory
from dm_c19_modelling.modelling.training import base_config
from dm_c19_modelling.modelling.training import runner
import numpy as np

_SEED = 42

_MODEL_NAMES = base_config.get_config("").model.model_specific_kwargs.keys()


FLAGS = flags.FLAGS


def get_mock_dataset(model_name):
  num_training_dates = 100
  num_forecast_dates = 7
  num_sites = 20

  dates = np.array([
      (datetime.datetime(year=2020, month=1, day=1) + datetime.timedelta(days=i)
      ).strftime(constants.DATE_FORMAT)
      for i in range(num_training_dates + num_forecast_dates)])
  training_dates = dates[:num_training_dates]
  evaluation_dates = dates[num_training_dates:]
  sites = np.array([f"site_{i}" for i in range(num_sites)])

  if model_name == "lstm":
    # LSTM requires the features to either be constant, or be targets.
    feature_names = np.array(["population", "target_2", "target_1"])
    num_features = 3
  else:
    num_features = 6
    feature_names = np.array([f"feature_{i}" for i in range(num_features - 1)] +
                             ["population"])

  if model_name == "seir_lstm":
    # This model is only compatible with this target specifically.
    target_names = np.array([constants.DECEASED_NEW])
    num_targets = 1
  else:
    num_targets = 20
    target_names = np.array([f"target_{i}" for i in range(num_targets)])

  rand = np.random.RandomState(_SEED)

  training_features = rand.normal(
      size=[num_training_dates, num_sites, num_features])
  training_targets = rand.normal(
      size=[num_training_dates, num_sites, num_targets])
  evaluation_targets = rand.normal(
      size=[num_forecast_dates, num_sites, num_targets])

  sum_past_targets = rand.normal(size=[num_sites, num_targets])

  return dataset_factory.Dataset(
      training_targets=training_targets,
      evaluation_targets=evaluation_targets,
      training_features=training_features,
      sum_past_targets=sum_past_targets,
      feature_names=feature_names,
      target_names=target_names,
      training_dates=training_dates,
      evaluation_dates=evaluation_dates,
      sites=sites,
      dataset_index_key="dummy_key",
      cadence=1,
      )


class RunnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Need to force parsing FLAGS,
    FLAGS(["runner_test.py"])

  @parameterized.parameters(
      *({"model_name": model_name} for model_name in _MODEL_NAMES))
  def test_smoke_test(self, model_name):
    config = base_config.get_config("")

    config.model.model_name = model_name

    with tempfile.TemporaryDirectory() as tmp_dir:
      config.checkpointer.directory = tmp_dir
      config.training.log_interval = 10
      config.training.checkpoint_interval = 40
      config.training.training_steps = 200
      config.fine_tune.fine_tune_steps = 100

      FLAGS.config = config

      with mock.patch.object(
          dataset_factory, "get_dataset",
          return_value=get_mock_dataset(model_name)):
        FLAGS.mode = "train"
        runner.main(None)

        FLAGS.mode = "eval"
        runner.main(None)


if __name__ == "__main__":
  absltest.main()
