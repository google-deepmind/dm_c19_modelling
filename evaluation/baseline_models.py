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
"""Baseline models to predict COVID-19 deaths / cases by region."""

import abc
import math
from typing import Optional, Tuple

from dm_c19_modelling.evaluation import dataset_factory
import numpy as np
from scipy import optimize


class CurveFittingModel(metaclass=abc.ABCMeta):
  """Models that fit a single target as a function of time for each site.

  A curve function is used to fit time to the target, separately for each site.
  """

  def __init__(self, num_context_dates: Optional[int], fit_cumulatives: bool):
    """Gets target predictions for the evaluation dates.

    Args:
      num_context_dates: Number of most recent data points to fit the curve to.
          If None, it will fit all available steps in the inputs.
      fit_cumulatives: Whether to fit the function to raw targets, or to
          cumulatives.
    """
    super().__init__()
    self._fit_cumulatives = fit_cumulatives
    self._num_context_dates = num_context_dates

  @abc.abstractmethod
  def _fit_and_predict(
      self, x_inputs: np.ndarray, y_inputs: np.ndarray, x_outputs: np.ndarray
      ) -> np.ndarray:
    """Returns the predictions for x_outputs, by fitting the inputs."""

  def predict(self, dataset: dataset_factory.Dataset) -> np.ndarray:
    """Uses per-site fitted params to predict targets for evaluation dates.

    Args:
      dataset: The training dataset.
    Returns:
      Predictions for the evaluation dates, of shape:
        (num_evaluation_dates, num_sites, 1)
    """

    num_context_dates = self._num_context_dates
    if num_context_dates is None:
      num_context_dates = len(dataset.training_dates)

    if num_context_dates > len(dataset.training_dates):
      raise ValueError(
          f"`Not enough training dates ({len(dataset.training_dates)}) for"
          f"`the required number of context dates ({num_context_dates}).")

    training_date_range = np.arange(0, num_context_dates)
    eval_date_range = np.arange(
        num_context_dates, num_context_dates + len(dataset.evaluation_dates))
    training_targets = dataset.training_targets

    if self._fit_cumulatives:
      # We want incremental predictions for the evaluation dates, so need to
      # produce a value for the final training date as well to enable taking
      # the diff of predictions.
      # Alternatively we could consider using the last ground truth value to
      # produce the diff for the first day.
      eval_date_range = np.concatenate(
          [[eval_date_range[0] - 1], eval_date_range])
      training_targets = np.cumsum(
          training_targets, axis=0) + dataset.sum_past_targets

    # Clip after calculating the `cumsum` so the totals are still correct.
    training_targets = training_targets[-num_context_dates:]  # pylint: disable=invalid-unary-operand-type

    predictions_all_sites = np.full(dataset.evaluation_targets.shape, np.nan)
    for target_idx in range(len(dataset.target_names)):
      for site_idx in range(len(dataset.sites)):
        train_targets_for_site = training_targets[:, site_idx, target_idx]

        prediction_targets_for_site = self._fit_and_predict(
            x_inputs=training_date_range,
            y_inputs=train_targets_for_site,
            x_outputs=eval_date_range)

        if self._fit_cumulatives:
          prediction_targets_for_site = np.diff(
              prediction_targets_for_site, axis=0)

        predictions_all_sites[:, site_idx, target_idx] = (
            prediction_targets_for_site)

    return predictions_all_sites


class PolynomialFit(CurveFittingModel):
  """Extrapolates fitting a polynomial to data from the last weeks."""

  def __init__(
      self, polynomial_degree: int, num_context_dates: Optional[int],
      fit_cumulatives: bool = False):
    """Gets target predictions for the evaluation dates.

    Args:
      polynomial_degree: Degree of the polynomial to fit.
      num_context_dates: See base class.
      fit_cumulatives: If True, fit cumulatives, instead of globals.

    """
    super().__init__(
        num_context_dates=num_context_dates, fit_cumulatives=fit_cumulatives)
    self._polynomial_degree = polynomial_degree

  def _fit_and_predict(
      self, x_inputs: np.ndarray, y_inputs: np.ndarray, x_outputs: np.ndarray
      ) -> np.ndarray:
    """Returns the predictions for x_outputs, by fitting the inputs."""
    fit_coefficients = np.polyfit(x_inputs, y_inputs, self._polynomial_degree)
    return np.polyval(fit_coefficients, x_outputs)


class ScipyCurveFittingModel(CurveFittingModel):
  """Model that fits an arbitrary function using scipy.optimize.curve_fit."""

  @abc.abstractmethod
  def _curve_function(self, *params):
    """The function to use to fit the target to time for each site."""

  @abc.abstractmethod
  def _get_initial_params(self, x_inputs: np.ndarray,
                          y_inputs: np.ndarray) -> Tuple[float, ...]:
    """Gets initialisation values for the parameters in the curve function."""

  def _fit_and_predict(
      self, x_inputs: np.ndarray, y_inputs: np.ndarray, x_outputs: np.ndarray
      ) -> np.ndarray:
    """Returns the predictions for x_outputs, by fitting the inputs."""
    params, _ = optimize.curve_fit(
        self._curve_function,
        x_inputs, y_inputs,
        maxfev=int(1e5),
        p0=self._get_initial_params(x_inputs, y_inputs))
    return self._curve_function(x_outputs, *params)


class Logistic(ScipyCurveFittingModel):
  """Fits a logistic function to the cumulative sum of the target."""

  def __init__(self, num_context_dates: int = None):
    super().__init__(fit_cumulatives=True, num_context_dates=num_context_dates)

  def _curve_function(self, t: np.ndarray, a: float, b: float,
                      c: float) -> np.ndarray:
    return a / (1.0 + np.exp(-b * (t - c)))

  def _get_initial_params(
      self, x_inputs: np.ndarray,
      y_inputs: np.ndarray) -> Tuple[float, float, float]:
    return (max(y_inputs), 1, np.median(x_inputs))


class Gompertz(ScipyCurveFittingModel):
  """Fits a Gompertz function to the cumulative sum of the target."""

  def __init__(self, num_context_dates: Optional[int] = None):
    super().__init__(fit_cumulatives=True, num_context_dates=num_context_dates)

  def _curve_function(self, t: np.ndarray, a: float, b: float,
                      c: float) -> np.ndarray:
    return a * np.exp(-b * np.exp(-c * t))

  def _get_initial_params(
      self, x_inputs: np.ndarray,
      y_inputs: np.ndarray) -> Tuple[float, float, float]:
    return (max(y_inputs), np.median(x_inputs), 1)


class RepeatLastWeek:
  """Repeats the last week's data to predict targets for evaluation dates."""

  def predict(self, dataset: dataset_factory.Dataset) -> np.ndarray:
    """Gets target predictions for the evaluation dates.

    Args:
      dataset: The training dataset.
    Returns:
      Predictions for the evaluation dates, of shape:
        (num_forecast_dates, num_sites, num_targets)
    """
    if dataset.cadence == 1:
      repeat_period = 7
    elif dataset.cadence == 7:
      repeat_period = 1
    else:
      raise ValueError(
          "Repeating the last week of data is only valid with a daily or "
          f"weekly cadence. Found cadence of {dataset.cadence}.")

    if len(dataset.training_dates) < repeat_period:
      raise ValueError(
          "At least 1 week of training data required to repeat weekly targets. "
          f"Found {len(dataset.training_dates)} days.")
    last_week_of_observed_targets = dataset.training_targets[-repeat_period:]
    num_forecast_dates = len(dataset.evaluation_dates)
    num_forecast_weeks = math.ceil(num_forecast_dates / repeat_period)
    predictions = np.concatenate([
        last_week_of_observed_targets for _ in range(num_forecast_weeks)
    ])[:num_forecast_dates]
    return predictions
