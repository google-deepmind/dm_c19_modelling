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
"""Tools for evaluating model forecasts."""

import datetime
import itertools
import os
from typing import List, Optional, Sequence, Tuple

from absl import logging
from dm_c19_modelling.evaluation import base_indexing
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import dataset_factory
from dm_c19_modelling.evaluation import dataset_indexing
from dm_c19_modelling.evaluation import forecast_indexing
from dm_c19_modelling.evaluation import metrics
from dm_c19_modelling.evaluation import plot_utils

import numpy as np
import pandas as pd

# Internal imports.

_METRICS_TO_CALCULATE = {"rmse": metrics.rmse, "mae": metrics.mae}


def _all_arrays_equal(arrs: Sequence[np.ndarray]) -> bool:
  """Checks whether all elements of a list of numpy arrays are equal."""
  first = arrs[0]
  for arr in arrs[1:]:
    if arr.shape != first.shape or not np.all(arr == first):
      return False
  return True


def _get_sorted_intersection_of_arrays(
    arrs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
  """Gets the intersecting and non-intersecting elements of a list of arrays."""
  sets = [set(arr) for arr in arrs]
  intersecting_elts = set.intersection(*sets)
  non_intersecting_elts = set.union(*sets) - set(intersecting_elts)
  return np.sort(list(intersecting_elts)), np.sort(list(non_intersecting_elts))


def _load_all_entries_and_forecasts(
    directory: str, dataset_name: str, forecast_ids: Sequence[str],
    target_name: str
) -> Tuple[List[base_indexing.IndexEntryType], List[pd.DataFrame]]:
  """Loads all forecast index entries and dataframes for the forecast IDs."""
  forecast_index = forecast_indexing.ForecastIndex(directory, dataset_name)
  all_forecast_entries = []
  all_forecasts = []
  for forecast_id in forecast_ids:
    key = forecast_index.query_by_forecast_id(forecast_id)
    if key is None:
      raise ValueError(f"Could not find forecast ID {forecast_id} in forecast "
                       f"index for dataset {dataset_name} in directory "
                       f"{directory}")
    all_forecast_entries.append(forecast_index.get_entry(key))
    forecast_df = forecast_index.load_file_by_key(key)
    # Filter the forecasts for the target of interest
    forecast_df_for_target = forecast_df[forecast_df[constants.TARGET_NAME] ==
                                         target_name]
    if forecast_df_for_target.empty:
      raise ValueError(f"Unable to find forecasts for target {target_name} in "
                       f"forecast {forecast_id}")
    # We only require the date, site and prediction columns; drop any others
    forecast_df_for_target = forecast_df_for_target[[
        constants.DATE, constants.SITE_ID, constants.PREDICTION
    ]]
    all_forecasts.append(forecast_df_for_target)
  return all_forecast_entries, all_forecasts


def _convert_forecasts_to_arrays(
    forecasts: Sequence[pd.DataFrame]) -> List[dataset_factory.DataArrays]:
  return [dataset_factory.df_to_arrays(forecast) for forecast in forecasts]


def _get_last_observation_date_and_validate_comparable(
    forecast_entries: Sequence[base_indexing.IndexEntryType]
) -> Tuple[str, int]:
  """Checks that the forecast index entries are compatible for evaluation."""
  last_observation_dates = np.array(
      [entry["last_observation_date"] for entry in forecast_entries])
  if not _all_arrays_equal(last_observation_dates):
    raise ValueError("Models can only be compared if they have the same "
                     "last_observation_date. Found last_observation_dates: "
                     f"{last_observation_dates}")
  forecast_cadences = np.array([entry["cadence"] for entry in forecast_entries])
  if not _all_arrays_equal(forecast_cadences):
    raise ValueError(
        "Models can only be compared if they have the same forecast cadence. "
        f"Found cadences: {forecast_cadences}")

  forecast_sources = [
      np.array(entry["source_data_info"]["dataset_key"])
      for entry in forecast_entries
  ]
  if not _all_arrays_equal(forecast_sources):
    raise ValueError(
        "Models can only be compared if they were trained using the same "
        f"dataset. Found dataset keys: {forecast_sources}")
  return str(last_observation_dates[0]), int(forecast_cadences[0])


def _get_forecast_spec_and_comparable_predictions(
    forecasts_arrays: List[dataset_factory.DataArrays],
    num_forecast_dates: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Validates that the forecast dataframes are compatible for evaluation."""
  date_arrays = [arrays.dates_array for arrays in forecasts_arrays]
  site_arrays = [arrays.sites_array for arrays in forecasts_arrays]
  feature_arrays = [arrays.features_array for arrays in forecasts_arrays]
  data_arrays = [arrays.data_array for arrays in forecasts_arrays]

  feature_set = set(itertools.chain.from_iterable(feature_arrays))
  assert feature_set == {
      "prediction"
  }, (f"Unexpected columns in forecasts: {feature_set - {'prediction'}}")

  if not _all_arrays_equal(site_arrays):
    overlapping_sites, sites_to_drop = _get_sorted_intersection_of_arrays(
        site_arrays)
    if not overlapping_sites.size:
      raise ValueError("Models can only be compared if they produce "
                       "predictions for overlapping sites.")
    updated_data_arrays = []
    for site_array, data_array in zip(site_arrays, data_arrays):
      site_locs = np.in1d(site_array, overlapping_sites)
      updated_data_arrays.append(data_array[:, site_locs, :])
    data_arrays = updated_data_arrays
  else:
    overlapping_sites = site_arrays[0]
    sites_to_drop = np.array([])

  if not _all_arrays_equal(date_arrays):
    overlapping_dates, _ = _get_sorted_intersection_of_arrays(date_arrays)
    if not overlapping_dates.size:
      raise ValueError("Models can only be compared if they produce "
                       "predictions for overlapping dates.")
    logging.warn(
        "Using the set of dates that are valid for all "
        "forecasts, from %s to %s", overlapping_dates[0], overlapping_dates[-1])

  else:
    overlapping_dates = date_arrays[0]

  updated_data_arrays = []
  if num_forecast_dates:
    overlapping_dates = overlapping_dates[:num_forecast_dates]
  for date_array, data_array in zip(date_arrays, data_arrays):
    date_locs = np.in1d(date_array, overlapping_dates)
    if not np.all(np.diff(np.where(date_locs)[0]) == 1):
      raise ValueError("Overlapping dates aren't consecutive through time.")
    updated_data_arrays.append(data_array[date_locs])

  return overlapping_dates, overlapping_sites, sites_to_drop, np.array(
      updated_data_arrays)


def _validate_eval_dataset_comparable(dataset: dataset_factory.Dataset,
                                      dates: np.ndarray, sites: np.ndarray):
  """Checks the eval dataset contains all the sites & dates in the forecasts."""
  assert np.all(np.sort(sites) == sites), "Forecast sites should be sorted"
  assert np.all(np.sort(dataset.sites) == dataset.sites), (
      "Dataset sites should be sorted")

  forecast_sites = set(sites)
  sites_available_in_dataset = set(dataset.sites)
  if not forecast_sites.issubset(sites_available_in_dataset):
    raise ValueError(
        "Not all of the sites in the forecasts are present in the evaluation "
        "dataset. Missing data for sites: "
        f"{forecast_sites - sites_available_in_dataset}")
  assert np.array_equal(dates, dataset.evaluation_dates), (
      "Dates in forecasts differ from dates in evaluation dataset")


def _calculate_metrics(forecast_id: str, predictions: np.ndarray,
                       ground_truth: np.ndarray,
                       target_name: str) -> pd.DataFrame:
  """Calculates metrics for a given dataframe of forecasts."""
  assert predictions.shape == ground_truth.shape, (
      f"Predictions array has shape {predictions.shape}, ground truth has "
      f"shape {ground_truth.shape}")
  metrics_data = []
  for metric_name, metric_fn in _METRICS_TO_CALCULATE.items():
    metric_value = metric_fn(predictions=predictions, ground_truth=ground_truth)
    metrics_data.append([forecast_id, metric_name, metric_value, target_name])
  return pd.DataFrame(
      data=metrics_data,
      columns=[
          "forecast_id", "metric_name", "metric_value", constants.TARGET_NAME
      ])


def get_recorded_creation_date(directory: str, dataset_name: str,
                               key: str) -> str:
  """Gets the actual creation date in case the creation date is 'latest'."""
  index = dataset_indexing.DatasetIndex(directory, dataset_name)
  entry = index.get_entry(key)
  return str(entry["creation_date"])


def evaluate(directory: str, dataset_name: str, eval_dataset_creation_date: str,
             target_name: constants.Targets, forecast_ids: Sequence[str],
             save: bool,
             sites_permitted_to_drop: Optional[Sequence[str]],
             num_forecast_dates: Optional[int]) -> pd.DataFrame:
  """Calculates and saves metrics for model forecasts if they are comparable."""
  all_forecast_entries, all_forecasts = _load_all_entries_and_forecasts(
      directory, dataset_name, forecast_ids, target_name.value)
  last_observation_date, forecast_cadence = (
      _get_last_observation_date_and_validate_comparable(all_forecast_entries))

  all_forecast_arrays = _convert_forecasts_to_arrays(all_forecasts)

  dates_to_eval, sites_to_eval, sites_to_drop, all_predictions = (
      _get_forecast_spec_and_comparable_predictions(all_forecast_arrays,
                                                    num_forecast_dates))

  if sites_to_drop.size and set(sites_to_drop) != set(sites_permitted_to_drop):
    raise ValueError(
        f"Only {sites_permitted_to_drop} are allowed to be dropped but "
        f"{len(sites_to_drop)} non-intersecting sites were found: "
        f"{sites_to_drop}.")

  elif sites_to_drop.size:
    logging.warn(
        "Using the set of sites that are defined for all forecasts: the "
        "following sites are being dropped: %s", sites_to_drop)

  forecast_horizon = (
      datetime.datetime.strptime(max(dates_to_eval), constants.DATE_FORMAT) -
      datetime.datetime.strptime(last_observation_date,
                                 constants.DATE_FORMAT)).days

  eval_dataset = dataset_factory.get_dataset(
      directory=directory,
      dataset_name=dataset_name,
      creation_date=eval_dataset_creation_date,
      last_observation_date=last_observation_date,
      targets=[target_name],
      features=[],
      num_forecast_dates=len(dates_to_eval),
      cadence=forecast_cadence)

  _validate_eval_dataset_comparable(eval_dataset, dates_to_eval, sites_to_eval)

  if np.any(np.isnan(eval_dataset.evaluation_targets)):
    raise ValueError(
        "NaNs found in the ground truth. A likely cause is that "
        f"the dataset does not contain {forecast_horizon} days "
        "of data after the last_observation_date. A later creation date "
        "may be required.")

  # Get the ground truth data for the required sites on the evaluation dates
  available_sites = eval_dataset.sites
  sites_locs = np.where(np.in1d(available_sites, sites_to_eval))[0]

  available_dates = eval_dataset.evaluation_dates
  dates_locs = np.where(np.in1d(available_dates, dates_to_eval))[0]
  ground_truth = eval_dataset.evaluation_targets[:, sites_locs, :]
  ground_truth = ground_truth[dates_locs, :, :]
  metrics_dfs = []
  for forecast_id, predictions in zip(forecast_ids, all_predictions):
    metrics_dfs.append(
        _calculate_metrics(forecast_id, predictions, ground_truth,
                           target_name.value))
  metrics_df = pd.concat(metrics_dfs)

  # Get the actual evaluation creation date in case using 'latest'
  eval_dataset_creation_date = get_recorded_creation_date(
      directory, dataset_name, eval_dataset.dataset_index_key)

  metrics_dir = os.path.join(directory, "metrics")

  if save:
    filename_base = (
        f"metrics_{'_'.join(forecast_ids)}_{eval_dataset_creation_date}_"
        f"{forecast_horizon}d"
    )
    _save_metrics(metrics_dir, filename_base, eval_dataset_creation_date,
                  metrics_df)
    _plot_metrics_and_save(
        directory=metrics_dir,
        filename_base=filename_base,
        target_name=target_name.value,
        metrics_df=metrics_df,
        forecast_index_entries=all_forecast_entries,
        last_observation_date=last_observation_date,
        forecast_horizon=forecast_horizon,
        eval_dataset_creation_date=eval_dataset_creation_date,
        num_dates=len(dates_to_eval),
        num_sites=len(sites_to_eval),
        cadence=forecast_cadence,
        dropped_sites=sites_to_drop
    )
  return metrics_df


def _save_metrics(directory: str, filename_base: str,
                  eval_dataset_creation_date: str, metrics_df: pd.DataFrame):
  """Saves metrics dataframe as a csv file in the metrics directory."""
  if not os.path.exists(directory):
    os.makedirs(directory)
  data_filepath = os.path.join(directory, f"{filename_base}.csv")
  if os.path.exists(data_filepath):
    raise IOError(f"Metrics already exist at {data_filepath}")
  logging.info("Saving metrics data to %s", data_filepath)
  with open(data_filepath, "w") as fid:
    metrics_df.to_csv(fid, index=False)


def _plot_metrics_and_save(directory: str, filename_base: str, target_name: str,
                           metrics_df: pd.DataFrame,
                           forecast_index_entries: Sequence[
                               base_indexing.IndexEntryType],
                           last_observation_date: str, forecast_horizon: int,
                           eval_dataset_creation_date: str, num_dates: int,
                           num_sites: int,
                           cadence: int,
                           dropped_sites: np.ndarray) -> None:
  """Plots metrics as a series of bar plots and saves them to file."""
  plot_filepath = os.path.join(directory, f"{filename_base}.png")
  fig = plot_utils.plot_metrics(
      metrics_df=metrics_df,
      forecast_index_entries=forecast_index_entries,
      target_name=target_name,
      last_observation_date=last_observation_date,
      forecast_horizon=forecast_horizon,
      eval_dataset_creation_date=eval_dataset_creation_date,
      num_sites=num_sites,
      num_dates=num_dates,
      cadence=cadence,
      dropped_sites=dropped_sites)
  logging.info("Saving metrics plots to %s", plot_filepath)
  with open(plot_filepath, 'wb') as fid:
    fig.savefig(fid, format="png", bbox_inches="tight")
