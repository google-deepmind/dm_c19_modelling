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
"""Tests for dm_c19_modelling.evaluation.forecast_utils."""

import datetime

from absl.testing import absltest
from absl.testing import parameterized
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import forecast_utils
import numpy as np


class ForecastUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ("saturday_start_friday_end", "2020-10-17", 13, ["2020-10-24"],
       [slice(1, 8)]),
      ("saturday_start_saturday_end", "2020-10-17", 14,
       ["2020-10-24", "2020-10-31"], [slice(1, 8), slice(8, 15)]),
      ("sunday_start_saturday_end", "2020-10-18", 13,
       ["2020-10-24", "2020-10-31"], [slice(0, 7), slice(7, 14)]),
      ("sunday_start_sunday_end", "2020-10-18", 14,
       ["2020-10-24", "2020-10-31"], [slice(0, 7), slice(7, 14)])
  ])
  def test_epiweekly_pooling_dates(self, first_eval_date, num_eval_dates,
                                   expected_eval_dates, expected_summed_slices):
    """Checks that evaluation dates and predictions are pooled correctly."""
    eval_datetimes = [
        datetime.datetime.strptime(first_eval_date, constants.DATE_FORMAT) +
        datetime.timedelta(days=i) for i in range(num_eval_dates + 1)
    ]
    eval_dates = [
        date.strftime(constants.DATE_FORMAT) for date in eval_datetimes
    ]
    predictions = np.random.randint(0, 10, (len(eval_dates), 3, 2))
    epiweekly_preds, epiweekly_eval_dates = (
        forecast_utils.pool_daily_forecasts_to_weekly(
            predictions, eval_dates, "Saturday"))
    # Ensure that the pooled evaluation dates are the Saturdays in range for
    # which there is a full week of predictions available.
    np.testing.assert_array_equal(np.array(epiweekly_eval_dates),
                                  expected_eval_dates)
    expected_preds = [
        np.sum(predictions[expected_slice], axis=0)
        for expected_slice in expected_summed_slices
    ]
    # Ensure that the pooled predictions are summed over epiweeks.
    np.testing.assert_array_equal(epiweekly_preds, np.array(expected_preds))

  def test_epiweekly_pooling_insufficient_data(self):
    """Checks that pooling fails when there's insufficient data."""
    eval_dates = ["2020-10-17", "2020-10-18"]
    predictions = np.random.randint(0, 10, (2, 2, 2))
    with self.assertRaises(ValueError):
      forecast_utils.pool_daily_forecasts_to_weekly(predictions, eval_dates,
                                                    "Saturday")

if __name__ == "__main__":
  absltest.main()
