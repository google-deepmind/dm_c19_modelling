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
"""Tests for dm_c19_modelling.evaluation.baseline_models."""

import datetime
import functools

from absl.testing import absltest
from absl.testing import parameterized
from dm_c19_modelling.evaluation import baseline_models
from dm_c19_modelling.evaluation import dataset_factory

import numpy as np


def _get_dataset(num_training_dates, num_forecast_dates, num_sites):
  sites = ["site_1", "site_2", "site_3"]
  training_datetimes = [
      datetime.datetime.strptime("2020-05-07", "%Y-%m-%d") +
      datetime.timedelta(days=i) for i in range(num_training_dates)
  ]
  eval_dates = range(num_training_dates,
                     num_training_dates + num_forecast_dates)
  eval_datetimes = [
      datetime.datetime.strptime("2020-05-07", "%Y-%m-%d") +
      datetime.timedelta(days=i) for i in eval_dates
  ]
  training_targets = np.random.randint(
      0, 100, (num_training_dates, num_sites, 1))
  eval_targets = np.random.randint(
      0, 100, (num_forecast_dates, num_sites, 1))
  sum_past_targets = np.random.randint(0, 100, (len(sites), 1))
  return dataset_factory.Dataset(
      training_targets=training_targets,
      evaluation_targets=eval_targets,
      sum_past_targets=sum_past_targets,
      training_features=[],
      target_names=["new_confirmed"],
      feature_names=[],
      training_dates=[
          datetime.datetime.strftime(date, "%Y-%m-%d")
          for date in training_datetimes
      ],
      evaluation_dates=[
          datetime.datetime.strftime(date, "%Y-%m-%d")
          for date in eval_datetimes
      ],
      sites=sites,
      dataset_index_key="12345",
      cadence=1
  )


class BaselineModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._num_training_dates = 20
    self._num_forecast_dates = 14
    self._num_sites = 3
    self._dataset = _get_dataset(
        self._num_training_dates, self._num_forecast_dates, self._num_sites)

  @parameterized.named_parameters([
      ("logistic", baseline_models.Logistic),
      ("gompertz", baseline_models.Gompertz),
      ("quadratic", functools.partial(
          baseline_models.PolynomialFit,
          polynomial_degree=2, num_context_dates=2, fit_cumulatives=False)),])
  def test_curve_fitting_model_predict(self, model_class):
    """Checks that predictions are the correct shape, defined, and positive."""
    model = model_class()
    predictions = model.predict(self._dataset)
    self.assertEqual(
        predictions.shape, (self._num_forecast_dates, self._num_sites, 1))
    if isinstance(model_class,
                  (baseline_models.Logistic, baseline_models.Gompertz)):
      self.assertFalse(np.any(predictions < 0))
    self.assertFalse(np.any(np.isnan(predictions)))

  def test_repeat_weekly_model_insufficient_data_raises_value_error(self):
    """Checks that the repeat weekly model fails with only 6 days of data."""
    model = baseline_models.RepeatLastWeek()
    dataset = _get_dataset(6, self._num_forecast_dates, self._num_sites)
    with self.assertRaisesRegex(ValueError,
                                "At least 1 week of training data required *"):
      model.predict(dataset)

  def test_repeat_weekly_deaths_model_6_day_horizon_outputs_correctly(self):
    """Checks predictions from the repeating model with horizon < 1 week."""
    model = baseline_models.RepeatLastWeek()
    dataset = _get_dataset(self._num_training_dates, 6, self._num_sites)
    predictions = model.predict(dataset)
    np.testing.assert_array_equal(predictions, dataset.training_targets[-7:-1])

  def test_repeat_weekly_deaths_model_12_day_horizon_outputs_correctly(self):
    """Checks predictions from the repeating model with horizon > 1 week."""
    model = baseline_models.RepeatLastWeek()
    dataset = _get_dataset(self._num_training_dates, 12, self._num_sites)
    predictions = model.predict(dataset)
    np.testing.assert_array_equal(predictions[:7],
                                  dataset.training_targets[-7:])
    np.testing.assert_array_equal(predictions[7:],
                                  dataset.training_targets[-7:-2])


if __name__ == "__main__":
  absltest.main()
