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
"""Tests for dm_c19_modelling.evaluation.forecast_indexing."""

import os

from absl.testing import absltest
from dm_c19_modelling.evaluation import forecast_indexing
import numpy as np
import pandas as pd

_TEST_DATASET = "test_dataset"
_TEST_FORECASTS_FILE = "test_forecasts.csv"


def _get_test_predictions_and_arrays():
  dates = np.array(["2020-05-07", "2020-05-08", "2020-05-09"])
  sites = np.array(["site_1", "site_2"])
  targets = np.array(["new_confirmed", "new_deceased"])
  predictions = np.random.rand(len(dates), len(sites), len(targets)) * 10
  return predictions, dates, sites, targets


def _get_test_entry(directory):
  return {
      "forecast_id": "12345",
      "file_location": os.path.join(directory, _TEST_FORECASTS_FILE),
      "source_data_info": ["test_dataset_1"],
      "creation_timestamp": "2020-06-07_12:43:02",
      "dataset_name": _TEST_DATASET,
      "last_observation_date": "2020-05-04",
      "cadence": 1,
      "features_used": ["new_deceased"],
      "extra_info": {"model_description": "test_model"}
  }


def _create_test_forecasts(file_location):
  target_dfs = []
  for target_name in ["new_deceased", "new_confirmed"]:
    target_dfs.append(pd.DataFrame({
        "site_id": ["site_1", "site_2", "site_1", "site_2"],
        "date": ["2020-05-05", "2020-05-05", "2020-05-06", "2020-05-06"],
        "target_name": [target_name] * 4,
        "prediction": np.random.rand(4) * 10
    }))
  df = pd.concat(target_dfs)
  df.to_csv(file_location, index=False)


class ForecastIndexingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._preds_arr, self._dates_arr, self._sites_arr, self._targets_arr = (
        _get_test_predictions_and_arrays())
    self._test_dir = absltest.get_default_test_tmpdir()
    os.makedirs(self._test_dir, exist_ok=True)
    self._key = "12345"
    self._entry = _get_test_entry(self._test_dir)
    _create_test_forecasts(self._entry["file_location"])
    self._remove_index_if_exists()

  def _remove_index_if_exists(self):
    index_path = os.path.join(
        self._test_dir, f"forecast_index-{_TEST_DATASET}.json")
    if os.path.exists(index_path):
      os.remove(index_path)

  def test_prediction_df_columns(self):
    """Checks the columns in the predictions dataframe are as expected."""
    predictions_df = forecast_indexing.build_predictions_df(
        self._preds_arr, self._dates_arr, self._sites_arr, self._targets_arr)
    np.testing.assert_array_equal(
        sorted(predictions_df.columns),
        ["date", "prediction", "site_id", "target_name"])

  def test_prediction_df_entries(self):
    """Checks that values in the predictions dataframe are as expected."""
    predictions_df = forecast_indexing.build_predictions_df(
        self._preds_arr, self._dates_arr, self._sites_arr, self._targets_arr)
    np.testing.assert_array_equal(sorted(predictions_df.site_id.unique()),
                                  self._sites_arr)
    np.testing.assert_array_equal(sorted(predictions_df.date.unique()),
                                  self._dates_arr)
    np.testing.assert_array_equal(sorted(predictions_df.target_name.unique()),
                                  self._targets_arr)
    sample_entry = predictions_df.query(
        "site_id=='site_1' & date=='2020-05-09' & target_name=='new_confirmed'"
    ).prediction
    np.testing.assert_array_almost_equal(
        sample_entry, [self._preds_arr[2][0][0]])

    sample_entry = predictions_df.query(
        "site_id=='site_2' & date=='2020-05-07' & target_name=='new_deceased'"
    ).prediction
    np.testing.assert_array_almost_equal(
        sample_entry, [self._preds_arr[0][1][1]])

  def test_predictions_df_bad_shape(self):
    """Checks that building the dataframe fails with inconsistent shapes."""
    sites_arr = np.append(self._sites_arr, ["site_3"])
    with self.assertRaisesRegex(
        ValueError, "Predictions have unexpected shape *"):
      forecast_indexing.build_predictions_df(
          self._preds_arr, self._dates_arr, sites_arr, self._targets_arr)

  def test_add_to_index_and_query(self):
    """Tests that a well-formatted forecasts entry is added to the index."""
    with forecast_indexing.ForecastIndex(self._test_dir, _TEST_DATASET,
                                         read_only=False) as index:
      index.add_entry(self._key, self._entry)

    read_index = forecast_indexing.ForecastIndex(self._test_dir, _TEST_DATASET)
    self.assertIsNotNone(read_index.query_by_forecast_id("12345"))

  def test_fails_validation_nan_predictions(self):
    """Checks that validation fails if there are NaNs in predictions."""
    df = pd.read_csv(self._entry["file_location"])
    df.loc[0, "prediction"] = np.nan
    df.to_csv(self._entry["file_location"], index=False)
    with forecast_indexing.ForecastIndex(self._test_dir, _TEST_DATASET,
                                         read_only=False) as index:
      with self.assertRaisesWithLiteralMatch(
          ValueError, "NaNs founds in forecasts"):
        index.add_entry(self._key, self._entry)

  def test_fails_validation_missing_predictions(self):
    """Checks that validation fails if a date is undefined for a site."""
    df = pd.read_csv(self._entry["file_location"])
    df.drop(0, inplace=True)
    df.to_csv(self._entry["file_location"], index=False)
    with forecast_indexing.ForecastIndex(self._test_dir, _TEST_DATASET,
                                         read_only=False) as index:
      with self.assertRaisesRegex(
          ValueError, "Missing data found in the forecasts*"):
        index.add_entry(self._key, self._entry)

  def test_fails_validation_missing_column(self):
    """Checks that validation fails when a column is missing."""
    df = pd.read_csv(self._entry["file_location"])
    df.drop(columns=["target_name"], inplace=True)
    df.to_csv(self._entry["file_location"], index=False)
    with forecast_indexing.ForecastIndex(self._test_dir, _TEST_DATASET,
                                         read_only=False) as index:
      with self.assertRaisesRegex(
          ValueError, "Forecasts must have columns*"):
        index.add_entry(self._key, self._entry)

  def test_fails_validation_inconsistent_cadence(self):
    """Checks that validation fails when forecasts have inconsistent cadence."""
    df = pd.read_csv(self._entry["file_location"])
    df_extra = df[df.date == "2020-05-06"]
    df_extra.date = "2020-05-08"
    df = pd.concat([df, df_extra])
    df.to_csv(self._entry["file_location"], index=False)
    with forecast_indexing.ForecastIndex(self._test_dir, _TEST_DATASET,
                                         read_only=False) as index:
      with self.assertRaisesWithLiteralMatch(
          ValueError, "Inconsistent cadence found in forecasts"):
        index.add_entry(self._key, self._entry)

  def test_fails_invalid_target_name(self):
    """Checks that validation fails when forecasts contain an invalid target."""
    df = pd.read_csv(self._entry["file_location"])
    df.loc[0:3, "target_name"] = "bad_target"
    df.to_csv(self._entry["file_location"], index=False)

    with forecast_indexing.ForecastIndex(self._test_dir, _TEST_DATASET,
                                         read_only=False) as index:
      with self.assertRaisesWithLiteralMatch(
          ValueError, "Invalid target in forecasts: bad_target"):
        index.add_entry(self._key, self._entry)

if __name__ == "__main__":
  absltest.main()
