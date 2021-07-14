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
"""Tests for dm_c19_modelling.evaluation.evaluation."""

import datetime
import itertools

from absl.testing import absltest
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import dataset_factory
from dm_c19_modelling.evaluation import evaluation
from dm_c19_modelling.evaluation import forecast_indexing
import numpy as np
import pandas as pd

_TEST_DATASET = "test_dataset"


def _get_forecasts(dates, sites):
  return pd.DataFrame({
      constants.DATE: np.repeat(dates, len(sites)),
      constants.SITE_ID: np.tile(sites, len(dates)),
      constants.PREDICTION: np.random.rand(len(dates) * len(sites)) * 10,
  })


def _get_entry(last_observation_date="2020-05-07",
               dataset_index_key="dset_index_1", cadence=1):
  return forecast_indexing.build_entry(
      forecast_id="",
      file_location="",
      dataset_name=_TEST_DATASET,
      creation_timestamp="",
      dataset_index_key=dataset_index_key,
      dataset_location="",
      last_observation_date=last_observation_date,
      cadence=cadence,
      extra_info={})


def _get_dataset(eval_dates, sites, target="new_confirmed", cadence=1):
  training_datetime = (
      datetime.datetime.strptime(eval_dates[0], "%Y-%m-%d") -
      datetime.timedelta(days=1))
  return dataset_factory.Dataset(
      training_targets=np.random.randint(0, 10, (1, len(sites), 1)),
      training_features=[],
      evaluation_targets=np.random.randint(0, 10,
                                           (len(eval_dates), len(sites), 1)),
      sum_past_targets=np.random.randint(0, 10, (len(sites), 1)),
      target_names=[target],
      feature_names=[],
      training_dates=[
          datetime.datetime.strftime(training_datetime, "%Y-%m-%d")
      ],
      evaluation_dates=eval_dates,
      sites=np.array(sites),
      dataset_index_key="test_dset_index_1",
      cadence=cadence)


class EvaluationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._default_dates = ["2020-05-07", "2020-05-08", "2020-05-09"]
    self._default_sites = ["site_1", "site_2"]

  def test_comparable_forecasts_subset_sites(self):
    """Check that validation fails for forecasts with different sites."""
    forecasts_1 = _get_forecasts(self._default_dates, self._default_sites)
    forecasts_2 = _get_forecasts(self._default_dates, ["site_1", "site_3"])
    all_forecast_arrays = evaluation._convert_forecasts_to_arrays(
        [forecasts_1, forecasts_2])
    _, sites_to_eval, sites_to_drop, updated_data_arrays = (
        evaluation._get_forecast_spec_and_comparable_predictions(
            all_forecast_arrays))
    np.testing.assert_array_equal(sites_to_eval, ["site_1"])
    np.testing.assert_array_equal(sites_to_drop, ["site_2", "site_3"])
    np.testing.assert_array_equal(
        updated_data_arrays, np.array(
            [arr.data_array for arr in all_forecast_arrays])[:, :, 0:1, :])
    np.testing.assert_array_equal(updated_data_arrays[0].shape,
                                  updated_data_arrays[1].shape)

  def test_incomparable_forecasts_subset_dates(self):
    """Check that validation fails for forecasts with different start dates."""
    forecasts_1 = _get_forecasts(self._default_dates, self._default_sites)
    forecasts_2 = _get_forecasts(["2020-05-06", "2020-05-07"],
                                 self._default_sites)
    all_forecast_arrays = evaluation._convert_forecasts_to_arrays(
        [forecasts_1, forecasts_2])
    dates_to_eval, _, _, updated_data_arrays = (
        evaluation._get_forecast_spec_and_comparable_predictions(
            all_forecast_arrays))
    np.testing.assert_array_equal(dates_to_eval, ["2020-05-07"])
    np.testing.assert_array_equal(
        updated_data_arrays[0], all_forecast_arrays[0].data_array[0:1])
    np.testing.assert_array_equal(
        updated_data_arrays[1], all_forecast_arrays[1].data_array[1:2])
    np.testing.assert_array_equal(updated_data_arrays[0].shape,
                                  updated_data_arrays[1].shape)

  def test_valid_different_forecast_horizons(self):
    """Check that validation passes for forecasts with different horizons."""
    forecasts_1 = _get_forecasts(self._default_dates, self._default_sites)
    forecasts_2 = _get_forecasts(["2020-05-07", "2020-05-08"],
                                 self._default_sites)
    all_forecast_arrays = evaluation._convert_forecasts_to_arrays(
        [forecasts_1, forecasts_2])
    evaluation._get_forecast_spec_and_comparable_predictions(
        all_forecast_arrays)

  def test_badly_formatted_forecasts(self):
    """Checks that forecasts with unexpected format fail evaluation."""
    forecasts = _get_forecasts(self._default_dates, self._default_sites)
    forecasts["extra_column"] = ""
    all_forecast_arrays = evaluation._convert_forecasts_to_arrays([forecasts])
    with self.assertRaisesRegex(AssertionError,
                                "Unexpected columns in forecasts:*"):
      evaluation._get_forecast_spec_and_comparable_predictions(
          all_forecast_arrays)

  def test_incomparable_last_observation_dates(self):
    """Checks that validation fails for different last_observation_dates."""
    entry_1 = _get_entry(last_observation_date="2020-05-06")
    entry_2 = _get_entry(last_observation_date="2020-05-07")
    with self.assertRaisesRegex(
        ValueError,
        "Models can only be compared if they have the same "
        "last_observation_date. *"):
      evaluation._get_last_observation_date_and_validate_comparable(
          [entry_1, entry_2])

  def test_incomparable_forecast_cadences(self):
    """Checks that validation fails for forecasts with different cadences."""
    entry_1 = _get_entry(cadence=1)
    entry_2 = _get_entry(cadence=7)
    with self.assertRaisesRegex(
        ValueError,
        "Models can only be compared if they have the same forecast cadence *"):
      evaluation._get_last_observation_date_and_validate_comparable(
          [entry_1, entry_2])

  def test_incomparable_forecast_sources(self):
    """Checks validation fails for forecasts trained on different datasets."""
    entry_1 = _get_entry(dataset_index_key="dset_index_1")
    entry_2 = _get_entry(dataset_index_key="dset_index_2")
    with self.assertRaisesRegex(
        ValueError,
        "Models can only be compared if they were trained using the same "
        "dataset.*"):
      evaluation._get_last_observation_date_and_validate_comparable(
          [entry_1, entry_2])

  def test_incomparable_eval_dset_missing_sites(self):
    """Checks that validation fails when the dataset is missing sites."""
    dataset = _get_dataset(self._default_dates, ["site_1", "site_3"])
    with self.assertRaisesRegex(
        ValueError, "Not all of the sites in the forecasts are present in the "
        "evaluation dataset*"):
      evaluation._validate_eval_dataset_comparable(dataset, self._default_dates,
                                                   self._default_sites)

  def test_incomparable_eval_dset_missing_dates(self):
    """Checks that validation fails when the dataset is missing dates."""
    dataset = _get_dataset(["2020-05-08", "2002-05-09"], self._default_sites)
    with self.assertRaisesWithLiteralMatch(
        AssertionError, "Dates in forecasts differ from dates in evaluation "
        "dataset"):
      evaluation._validate_eval_dataset_comparable(dataset, self._default_dates,
                                                   self._default_sites)

  def test_calculate_metrics(self):
    """Checks that metric calculations yield the correct values."""
    predictions = np.array([[[1.], [2.], [3.]], [[1.], [2.], [2.]]])
    ground_truth = np.array([[[1.], [1.], [1.]], [[1.], [1.], [2.]]])
    metrics_df = evaluation._calculate_metrics("", predictions, ground_truth,
                                               "new_confirmed")
    self.assertAlmostEqual(
        1.,
        metrics_df.query("metric_name=='rmse'").metric_value.values[0])

  def test_convert_forecasts_to_arrays(self):
    """Checks all data is preserved when converting DataFrames to arrays."""
    forecasts_1 = _get_forecasts(self._default_dates, self._default_sites)
    forecasts_2 = _get_forecasts(self._default_dates, self._default_sites[::-1])
    forecast_dfs = [forecasts_1, forecasts_2]
    forecast_arrays = evaluation._convert_forecasts_to_arrays(forecast_dfs)
    for array, df in zip(forecast_arrays, forecast_dfs):
      np.testing.assert_array_equal(array.dates_array, self._default_dates)
      np.testing.assert_array_equal(array.sites_array, self._default_sites)
      np.testing.assert_array_equal(array.features_array,
                                    [constants.PREDICTION])
      for date, site in itertools.product(self._default_dates,
                                          self._default_sites):
        date_idx = list(array.dates_array).index(date)
        site_idx = list(array.sites_array).index(site)
        array_entry = array.data_array[date_idx][site_idx][0]
        df_entry = df.query(
            f"site_id=='{site}' and date=='{date}'").prediction.values[0]
        self.assertAlmostEqual(df_entry, array_entry)


if __name__ == "__main__":
  absltest.main()
