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

"""Tests for dm_c19_modelling.evaluation.download_data."""

from absl.testing import absltest
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import download_data

import numpy as np
import pandas as pd


class DownloadDataTest(absltest.TestCase):

  def test_join_dataframes_by_key_and_date(self):
    df1 = {
        "key": ["a", "b"],
        "date": ["2020-06-07", "2020-07-07"],
        "values_1": [1, 2]
    }
    df2 = {
        "key": ["a", "c"],
        "date": ["2020-06-07", "2020-07-07"],
        "values_2": [2, 3]
    }
    df_joined = download_data._join_two(pd.DataFrame(df1), pd.DataFrame(df2))
    np.testing.assert_array_equal(df_joined.columns,
                                  ["key", "date", "values_1", "values_2"])
    np.testing.assert_array_equal(df_joined["key"], ["a", "b", "c"])
    np.testing.assert_array_equal(df_joined["date"],
                                  ["2020-06-07", "2020-07-07", "2020-07-07"])
    np.testing.assert_array_equal(df_joined["values_1"], [1, 2, np.nan])
    np.testing.assert_array_equal(df_joined["values_2"], [2, np.nan, 3])

  def test_join_dataframes_by_key(self):
    df1 = {"key": ["a", "c"], "values_1": [1, 2]}
    df2 = {
        "key": ["a", "a", "c"],
        "date": ["2020-06-07", "2020-07-07", "2020-07-07"],
        "values_2": [2, 3, 4]
    }
    df_joined = download_data._join_two(pd.DataFrame(df1), pd.DataFrame(df2))
    np.testing.assert_array_equal(df_joined.columns,
                                  ["key", "values_1", "date", "values_2"])
    np.testing.assert_array_equal(df_joined["key"], ["a", "a", "c"])
    np.testing.assert_array_equal(df_joined["date"],
                                  ["2020-06-07", "2020-07-07", "2020-07-07"])
    np.testing.assert_array_equal(df_joined["values_1"], [1, 1, 2])
    np.testing.assert_array_equal(df_joined["values_2"], [2, 3, 4])

  def test_non_daily_cadence(self):
    df = {
        "site_id": ["a", "a", "b"],
        "date": ["2020-01-05", "2020-01-15", "2020-01-05"],
        "new_confirmed": [10, 20, 30]
    }
    df = download_data._drop_sites_with_insufficient_data(
        pd.DataFrame(df), "covid_open_data_world", "new_confirmed")
    np.testing.assert_array_equal(df["site_id"], ["b"])

  def test_all_sites_dropped(self):
    df = {
        "site_id": ["a"],
        # The first required date for for the world data is 2020-01-05, so this
        # should be dropped.
        "date": ["2020-01-15"],
        "new_confirmed": [10]
    }
    with self.assertRaisesWithLiteralMatch(
        ValueError, "All sites have been dropped."):
      download_data._drop_sites_with_insufficient_data(
          pd.DataFrame(df), "covid_open_data_world", "new_confirmed")

  def test_missing_early_data_dropped(self):
    df = {
        "site_id": ["a", "b", "b"],
        "date": ["2020-01-05", "2020-01-06", "2020-01-07"],
        "new_confirmed": [10, 30, 20]
    }
    df = download_data._drop_sites_with_insufficient_data(
        pd.DataFrame(df), "covid_open_data_world", "new_confirmed")
    np.testing.assert_array_equal(df["site_id"], ["a"])

  def test_grace_period(self):
    df = {
        "key": ["a", "a", "a",
                "b", "b", "b",
               ],
        "date": ["2020-03-16", "2020-03-17", "2020-03-18",
                 "2020-03-15", "2020-03-16", "2020-03-17",],
        "new_confirmed": [1, 2, 3,
                          4, 5, 6],
        "new_deceased": [10, 20, 30,
                         40, 50, 60]
    }

    df = download_data._maybe_set_zero_epidemiology_targets_in_grace_period(
        pd.DataFrame(df), "covid_open_data_us_states")

    self.assertSetEqual({"a", "b"}, set(df["key"].unique()))

    # Check values and dates for "a":
    a_df = df.query("key=='a'").sort_values("date")
    expected_dates = list(pd.date_range("2020-01-22", "2020-03-18").strftime(
        constants.DATE_FORMAT))
    np.testing.assert_array_equal(a_df["date"], expected_dates)

    expected_confirmed = [0] * 54 + [1, 2, 3]
    np.testing.assert_array_equal(a_df["new_confirmed"], expected_confirmed)

    expected_deceased = [0] * 54 + [10, 20, 30]
    np.testing.assert_array_equal(a_df["new_deceased"], expected_deceased)

    # Check values and dates for "b":
    b_df = df.query("key=='b'").sort_values("date")
    expected_dates = list(pd.date_range("2020-01-22", "2020-03-17").strftime(
        constants.DATE_FORMAT))
    np.testing.assert_array_equal(b_df["date"], expected_dates)

    expected_confirmed = [0] * 53 + [4, 5, 6]
    np.testing.assert_array_equal(b_df["new_confirmed"], expected_confirmed)

    expected_deceased = [0] * 53 + [40, 50, 60]
    np.testing.assert_array_equal(b_df["new_deceased"], expected_deceased)

  def test_missing_data_dropped(self):
    df = {
        "site_id": ["a", "a", "a", "b"],
        "date": ["2020-01-05", "2020-01-06", "2020-01-07", "2020-01-05"],
        "new_confirmed": [10, np.NaN, 10, 20]
    }
    df = download_data._drop_sites_with_insufficient_data(
        pd.DataFrame(df), "covid_open_data_world", "new_confirmed")
    np.testing.assert_array_equal(df["site_id"], ["b"])

  def test_data_truncation(self):
    df = {
        "site_id": ["a", "a", "a", "b", "b"],
        "date": [
            "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-05", "2020-01-06"
        ],
        "new_confirmed": [10, 10, 10, 20, 30]
    }
    df = download_data._drop_sites_with_insufficient_data(
        pd.DataFrame(df), "covid_open_data_world", "new_confirmed")
    self.assertEqual(df.date.max(), "2020-01-06")


if __name__ == "__main__":
  absltest.main()
