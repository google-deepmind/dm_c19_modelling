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
"""Tests for dm_c19_modelling.evaluation.dataset_factory."""

from unittest import mock

from absl.testing import absltest
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import dataset_factory
import numpy as np
import pandas as pd

_DEFAULT_ARGS = {"directory": "", "dataset_name": "", "creation_date": "",
                 "cadence": 1}


def _get_raw_dataset():
  dates = ["2020-05-01",
           "2020-05-02",
           "2020-05-03",
           "2020-05-04",
           "2020-05-05",
           "2020-05-06",
           "2020-05-07",
           "2020-05-08",
           "2020-05-09"]
  sites = ["site_1", "site_2"]
  df = pd.DataFrame({
      constants.DATE: np.repeat(dates, len(sites)),
      constants.SITE_ID: np.tile(sites, len(dates)),
      "new_deceased": np.random.randint(0, 5,
                                        len(sites) * len(dates)),
      "new_confirmed": np.random.randint(0, 9,
                                         len(sites) * len(dates)),
      "feature_1": np.random.rand(len(sites) * len(dates)),
      "feature_2": np.random.rand(len(sites) * len(dates)),
      "feature_3": np.random.rand(len(sites) * len(dates)),
      "feature_4": np.random.rand(len(sites) * len(dates))
  })
  df[constants.DATE] = pd.to_datetime(df[constants.DATE])
  return df


class DataFactoryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._raw_dataset = _get_raw_dataset()
    self._key = "12345"
    self._mock_raw_dataset = self.enter_context(
        mock.patch.object(
            dataset_factory, "_load_dataset_by_creation_date", autospec=True))
    self._mock_raw_dataset.return_value = self._raw_dataset, self._key

  def test_invalid_last_observation_date(self):
    """Checks for failure when the last date is beyond the defined range."""
    last_observation_date = "2020-05-10"
    with self.assertRaisesRegex(
        ValueError,
        f"Forecast date {last_observation_date} not found in dataset. *"):
      dataset_factory.get_dataset(
          last_observation_date=last_observation_date,
          targets=[],
          features=[],
          num_forecast_dates=14,
          **_DEFAULT_ARGS)

  def test_insufficient_num_forecast_dates(self):
    """Checks padding applied when the dataset is missing evaluation dates."""
    last_observation_date = "2020-05-08"
    num_forecast_dates = 2
    dataset = dataset_factory.get_dataset(
        last_observation_date=last_observation_date,
        targets=[constants.Targets.CONFIRMED_NEW,
                 constants.Targets.DECEASED_NEW],
        features=[],
        num_forecast_dates=num_forecast_dates,
        **_DEFAULT_ARGS)
    np.testing.assert_equal(dataset.evaluation_dates,
                            ["2020-05-09", "2020-05-10"])
    self.assertTrue(np.isnan(dataset.evaluation_targets[1:]).all())

  def test_invalid_min_num_training_dates(self):
    """Checks for failure when there's insufficient training data."""
    last_observation_date = "2020-05-08"
    min_num_training_dates = 9
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f"Could not retrieve {min_num_training_dates} days of data before "
        f"{last_observation_date} from dataset."):
      dataset_factory.get_dataset(
          last_observation_date=last_observation_date,
          targets=[],
          features=[],
          num_forecast_dates=1,
          min_num_training_dates=min_num_training_dates,
          **_DEFAULT_ARGS)

  def test_valid_min_num_training_dates(self):
    """Checks that a valid constraint on training data succeeds."""
    last_observation_date = "2020-05-08"
    min_num_training_dates = 2
    dataset = dataset_factory.get_dataset(
        last_observation_date=last_observation_date,
        targets=[],
        features=[],
        num_forecast_dates=1,
        min_num_training_dates=min_num_training_dates,
        **_DEFAULT_ARGS)
    self.assertLen(dataset.training_dates, 8)

  def test_invalid_feature(self):
    """Checks for failure when a requested feature isn't present."""
    features = ["feature_4", "feature_5"]
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Could not find requested features ['feature_5'] in dataset"):
      dataset_factory.get_dataset(
          last_observation_date="2020-05-08",
          targets=[],
          features=features,
          num_forecast_dates=1,
          **_DEFAULT_ARGS)

  def test_training_data_truncation(self):
    """Checks that training data is truncated when requested."""
    last_observation_date = "2020-05-08"
    dataset = dataset_factory.get_dataset(
        last_observation_date=last_observation_date,
        targets=[constants.Targets.DECEASED_NEW,
                 constants.Targets.CONFIRMED_NEW],
        features=[],
        max_num_training_dates=1,
        num_forecast_dates=1,
        **_DEFAULT_ARGS)
    self.assertEqual(dataset.training_dates[0], last_observation_date)

    actual_sum_past_targets = self._raw_dataset[
        dataset.target_names].to_numpy().reshape(9, 2, 2)[:-2].sum(0)
    np.testing.assert_array_equal(
        actual_sum_past_targets, dataset.sum_past_targets)

  def test_dataset_shapes_and_values(self):
    """Checks the shapes and values of a valid dataset specification."""
    dataset = dataset_factory.get_dataset(
        last_observation_date="2020-05-08",
        targets=[constants.Targets.CONFIRMED_NEW],
        features=["feature_1", "feature_4", "feature_2"],
        num_forecast_dates=1,
        **_DEFAULT_ARGS)

    self.assertEqual(dataset.training_targets.shape, (8, 2, 1))
    self.assertEqual(dataset.evaluation_targets.shape, (1, 2, 1))
    self.assertEqual(dataset.training_features.shape, (8, 2, 3))

    actual_target = self._raw_dataset[dataset.target_names].to_numpy().reshape(
        9, 2, 1)
    np.testing.assert_array_equal(actual_target[:8], dataset.training_targets)
    np.testing.assert_array_equal(actual_target[8:], dataset.evaluation_targets)
    np.testing.assert_array_equal(
        actual_target[0] * 0, dataset.sum_past_targets)

    actual_features = (
        self._raw_dataset[dataset.feature_names].to_numpy().reshape(9, 2, 3))
    np.testing.assert_array_equal(actual_features[:8],
                                  dataset.training_features)

    np.testing.assert_array_equal(["2020-05-01",
                                   "2020-05-02",
                                   "2020-05-03",
                                   "2020-05-04",
                                   "2020-05-05",
                                   "2020-05-06",
                                   "2020-05-07",
                                   "2020-05-08"],
                                  dataset.training_dates)
    np.testing.assert_array_equal(["2020-05-09"], dataset.evaluation_dates)
    np.testing.assert_array_equal(["site_1", "site_2"], dataset.sites)

  def test_nan_targets(self):
    """Checks for failure when there are rows with undefined targets."""
    self._raw_dataset.loc[1, "new_deceased"] = np.nan
    self._mock_raw_dataset.return_value = self._raw_dataset, self._key
    with self.assertRaisesWithLiteralMatch(ValueError,
                                           "NaNs found in the target columns."):
      dataset_factory.get_dataset(
          last_observation_date="2020-05-08",
          targets=[
              constants.Targets.CONFIRMED_NEW, constants.Targets.DECEASED_NEW
          ],
          features=["feature_1", "feature_2"],
          num_forecast_dates=1,
          **_DEFAULT_ARGS)

  def test_missing_row(self):
    """Checks for failure when there are missing rows."""
    dataset = self._raw_dataset.loc[1:]
    self._mock_raw_dataset.return_value = dataset, self._key
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Found missing rows in the dataset for a date and site."):
      dataset_factory.get_dataset(
          last_observation_date="2020-05-08",
          targets=[
              constants.Targets.CONFIRMED_NEW, constants.Targets.DECEASED_NEW
          ],
          features=["feature_1", "feature_2"],
          num_forecast_dates=1,
          **_DEFAULT_ARGS)

  def test_duplicate_row(self):
    """Checks for failure when there are duplicate rows."""
    dataset = pd.concat(
        [self._raw_dataset, self._raw_dataset[self._raw_dataset.index == 0]])
    self._mock_raw_dataset.return_value = dataset, self._key
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Found duplicate rows in the dataset for a date and site."):
      dataset_factory.get_dataset(
          last_observation_date="2020-05-08",
          targets=[
              constants.Targets.CONFIRMED_NEW, constants.Targets.DECEASED_NEW
          ],
          features=["feature_1", "feature_2"],
          num_forecast_dates=1,
          **_DEFAULT_ARGS)

  def test_shape_and_value_non_daily_cadence_eval_data(self):
    """Checks that evaluation data is downsampled to fit a required cadence."""
    args = _DEFAULT_ARGS.copy()
    args["cadence"] = 2
    dataset = dataset_factory.get_dataset(
        last_observation_date="2020-05-08",
        targets=[
            constants.Targets.CONFIRMED_NEW, constants.Targets.DECEASED_NEW
        ],
        features=["feature_1", "feature_2"],
        num_forecast_dates=1,
        **args)
    np.testing.assert_array_equal(dataset.evaluation_dates, ["2020-05-10"])
    np.testing.assert_array_equal(
        dataset.evaluation_targets.shape,
        (1, 2, 2))  # 1 evaluation date, 2 sites, 2 targets
    self.assertTrue(np.all(np.isnan(dataset.evaluation_targets)))

  def test_shape_and_value_non_daily_cadence_train_data(self):
    """Checks that training data is downsampled to fit a required cadence."""
    args = _DEFAULT_ARGS.copy()
    args["cadence"] = 2
    feature_names = ["feature_1", "feature_2"]
    dataset = dataset_factory.get_dataset(
        last_observation_date="2020-05-09",
        targets=[
            constants.Targets.CONFIRMED_NEW, constants.Targets.DECEASED_NEW
        ],
        features=feature_names,
        num_forecast_dates=1,
        **args)
    np.testing.assert_array_equal(dataset.training_dates, [
        "2020-05-03",
        "2020-05-05",
        "2020-05-07",
        "2020-05-09"])
    np.testing.assert_array_equal(
        dataset.training_targets.shape, (4, 2, 2))
    actual_targets = self._raw_dataset[dataset.target_names].to_numpy().reshape(
        9, 2, 2)
    np.testing.assert_array_equal(
        dataset.training_targets[-1], np.sum(actual_targets[-2:], axis=0))
    np.testing.assert_array_equal(
        dataset.training_features.shape, (4, 2, 2))
    actual_features = self._raw_dataset[feature_names].to_numpy().reshape(
        9, 2, 2)
    np.testing.assert_array_equal(
        dataset.training_features[-1], np.mean(actual_features[-2:], axis=0))

    # Cadence of 2, should have discarded the first date, but this should
    # still show in the summed_past_targets.
    actual_sum_past_targets = self._raw_dataset[
        dataset.target_names].to_numpy().reshape(9, 2, 2)[:1].sum(0)
    np.testing.assert_array_equal(
        actual_sum_past_targets, dataset.sum_past_targets)

  def test_non_daily_cadence_train_data_no_features(self):
    """Checks that downsampling works when there are no features."""
    args = _DEFAULT_ARGS.copy()
    args["cadence"] = 2
    feature_names = []
    dataset = dataset_factory.get_dataset(
        last_observation_date="2020-05-09",
        targets=[
            constants.Targets.CONFIRMED_NEW, constants.Targets.DECEASED_NEW
        ],
        features=feature_names,
        num_forecast_dates=1,
        **args)
    self.assertEqual(dataset.training_features.size, 0)

  def test_error_when_dropping_sites(self):
    """Checks for failure when sites have completely missing data."""
    self._raw_dataset["feature_1"].loc[self._raw_dataset.site_id ==
                                       "site_1"] = np.nan
    self._mock_raw_dataset.return_value = self._raw_dataset, self._key
    with self.assertRaisesRegex(
        ValueError,
        "Found 1 sites where at least 1 feature was entirely missing *"):
      dataset_factory.get_dataset(
          last_observation_date="2020-05-08",
          targets=[
              constants.Targets.CONFIRMED_NEW, constants.Targets.DECEASED_NEW
          ],
          features=["feature_1", "feature_2"],
          num_forecast_dates=1,
          **_DEFAULT_ARGS)

  def test_sites_dropped_when_permitted(self):
    """Checks that sites are dropped when they completely missing data."""
    self._raw_dataset["feature_1"].loc[self._raw_dataset.site_id ==
                                       "site_1"] = np.nan
    self._mock_raw_dataset.return_value = self._raw_dataset, self._key
    dataset = dataset_factory.get_dataset(
        last_observation_date="2020-05-08",
        targets=[
            constants.Targets.CONFIRMED_NEW, constants.Targets.DECEASED_NEW
        ],
        features=["feature_1", "feature_2"],
        num_forecast_dates=1,
        allow_dropped_sites=True,
        **_DEFAULT_ARGS)
    self.assertEqual(dataset.sites, ["site_2"])

if __name__ == "__main__":
  absltest.main()
