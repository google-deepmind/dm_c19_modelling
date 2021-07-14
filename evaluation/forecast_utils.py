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
"""Utilities for manipulating forecasts."""

import datetime
import time
from typing import List, Tuple

from dm_c19_modelling.evaluation import constants
import numpy as np


def _get_weekly_slices(datetimes: np.ndarray, week_end_day: str) -> List[slice]:
  """Gets slices from consecutive datetimes corresponding to specified weeks.

  Args:
    datetimes: 1D array of datetimes from which we want to extract weeks.
    week_end_day: the required day of the week that the extracted weeks should
      end on. Specified as the full weekday name e.g. 'Saturday'.
  Returns:
    the slices in the datetimes array that correspond to the specified weeks.
  """
  week_start_day = time.strptime(week_end_day, "%A").tm_wday + 1
  week_starts = np.where(
      [date.weekday() == week_start_day for date in datetimes])[0]
  return [
      slice(start, start + 7)
      for start in week_starts
      if (start + 7) <= len(datetimes)
  ]


def pool_daily_forecasts_to_weekly(
    predictions: np.ndarray, evaluation_dates: np.ndarray,
    week_end_day: str) -> Tuple[np.ndarray, np.ndarray]:
  """Sums daily forecasts up to complete epiweekly forecasts.

  Args:
    predictions: shape(num_dates, num_sites, num_targets).
    evaluation_dates: strings, shape (num_dates).
    week_end_day: the desired day of the week to pool up to. E.g. a value of
      Sunday would pool predictions over normal weeks; a value of Saturday would
      pool predictions over epidemiological weeks.

  Returns:
    predictions summed across weeks ending on week_end_day, and the end dates
    corresponding to the end of those weeks. Where there are insufficient
    predictions to construct a full week, these are discarded.
  """
  evaluation_datetimes = np.array([
      datetime.datetime.strptime(evaluation_date, constants.DATE_FORMAT)
      for evaluation_date in evaluation_dates
  ])
  if not np.all(np.diff(evaluation_datetimes) == datetime.timedelta(days=1)):
    raise ValueError("Pooling forecasts to epiweekly is only available for "
                     "daily predictions.")
  week_slices = _get_weekly_slices(evaluation_datetimes, week_end_day)
  if not week_slices:
    raise ValueError("Insufficient predictions to pool to weekly cadence.")
  incremental_weekly_preds = [
      np.sum(predictions[week_slice], axis=0) for week_slice in week_slices
  ]
  # Get the last date in the pooled week for each week in the evaluation dates.
  weekly_evaluation_datetimes = [
      evaluation_datetimes[week_slice][-1] for week_slice in week_slices
  ]
  condition = np.all([
      date.strftime("%A") == week_end_day
      for date in weekly_evaluation_datetimes
  ])

  assert condition, "Incorrect day found in evaluation datetimes"
  weekly_evaluation_dates = [
      date.strftime(constants.DATE_FORMAT)
      for date in weekly_evaluation_datetimes
  ]
  return np.array(incremental_weekly_preds), np.array(weekly_evaluation_dates)
