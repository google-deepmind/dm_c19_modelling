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
"""Tools for constructing datasets for modelling."""
import datetime
from typing import NamedTuple, Optional, Sequence, Tuple

from absl import logging
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import dataset_indexing
import numpy as np
import pandas as pd


class Dataset(NamedTuple):
  """A dataset with specific training / eval dates, targets & features."""
  training_targets: np.ndarray  # (num_train_dates, num_sites, num_targets)
  evaluation_targets: np.ndarray  # (num_forecast_dates, num_sites, num_targets)
  training_features: np.ndarray  # (num_train_dates, num_sites, num_features)
  # Number of summed targets up to the beginning of the training dates.
  # Since targets are incremental numbers some models may use this.
  sum_past_targets: np.ndarray  # (num_sites, num_targets)
  feature_names: np.ndarray
  target_names: np.ndarray
  training_dates: np.ndarray
  evaluation_dates: np.ndarray
  sites: np.ndarray
  dataset_index_key: str
  cadence: int


class DataArrays(NamedTuple):
  """Internal helper structure for arrays of data."""
  data_array: np.ndarray  # Shape (num_dates, num_sites, num_features)
  dates_array: np.ndarray  # Array of dates for the data
  sites_array: np.ndarray  # Array of site IDs for the data
  features_array: np.ndarray  # Array of available feature names for the data


def _load_dataset_by_creation_date(
    directory: str, dataset_name: str,
    creation_date: str) -> Tuple[pd.DataFrame, str]:
  """Loads a dataset according to its creation date."""
  index = dataset_indexing.DatasetIndex(directory, dataset_name)
  key = index.query_by_creation_date(creation_date)
  dataset = index.load_file_by_key(key)
  dataset[constants.DATE] = pd.to_datetime(dataset[constants.DATE])
  return dataset, key


def _load_dataset_by_key(directory: str, dataset_name: str,
                         dataset_index_key: str) -> pd.DataFrame:
  index = dataset_indexing.DatasetIndex(directory, dataset_name)
  return index.get_entry(dataset_index_key)


def df_to_arrays(df: pd.DataFrame) -> DataArrays:
  """Converts dataframe into a 3D array with axes: (time, site, feature).

  Args:
    df: Dataframe containing site, date and feature columns.

  Returns:
    DataArrays containing:
    * data_array: the dataset transformed into a 3D numpy array with axes
      (time, site, feature)
    * date_array: the dates that are present in the dataset
    * site_array: the sites that are present in the dataset
    * feature_array: the available features / targets in the dataset
  """
  num_sites = len(df[constants.SITE_ID].unique())
  num_dates = len(df[constants.DATE].unique())
  # The total number of feature columns is the total columns, minus the site
  # and date columns.
  total_num_columns = len(df.columns) - 2

  # For each new grouping of the table, we will take:
  # * The count, so we can verify later that there are zero, or 1 elements at
  #   most per data and per site.
  # * The first value corresponding to the only element.
  pivoted_df = pd.pivot_table(
      df,
      index=constants.DATE,
      columns=[constants.SITE_ID],
      dropna=False,
      aggfunc=["first", len]).swaplevel(axis=1)

  # Assert that we had no extra rows. The pivot table replaces missing values
  # with NaNs, so we replace the NaN lengths by zeros, and then check that at
  # most one row contributed to each value.
  if pivoted_df["len"].isna().any().any():
    raise ValueError("Found missing rows in the dataset for a date and site.")
  if not np.all(pivoted_df["len"].values == 1.):
    raise ValueError("Found duplicate rows in the dataset for a date and site.")

  # Get the first (and only) value for each feature, date and site.
  pivoted_df = pivoted_df["first"].swaplevel(axis=1)
  # Convert pivot to a numpy array of shape (num_dates, num_sites, num_columns)
  data_array = pivoted_df.to_numpy().reshape(num_dates, total_num_columns,
                                             num_sites).swapaxes(2, 1)

  dates_array = pivoted_df.index
  features_array, sites_array = pivoted_df.columns.levels
  return DataArrays(
      data_array=data_array,
      dates_array=np.array(dates_array),
      sites_array=np.array(sites_array),
      features_array=np.array(features_array))


def _get_train_and_eval_date_slices(
    dates_arr: np.ndarray, last_observation_date: str,
    min_num_days_in_training: Optional[int],
    max_num_days_in_training: Optional[int],
    num_days_in_evaluation_period: int) -> Tuple[slice, slice]:
  """Gets the slices along the date axis for the training and eval dates."""
  last_observation_datetime = np.datetime64(last_observation_date)

  forecast_index = np.where(dates_arr == last_observation_datetime)[0]
  if not forecast_index.size:
    raise ValueError(
        f"Forecast date {last_observation_date} not found in dataset. "
        f"The following dates are available: {dates_arr}")
  assert len(forecast_index) == 1, "Found duplicate dates."
  forecast_index = forecast_index[0]

  first_training_index = 0
  if max_num_days_in_training:
    first_training_index = max(
        forecast_index - max_num_days_in_training + 1, first_training_index)

  if min_num_days_in_training is None:
    min_num_days_in_training = 0
  if forecast_index - first_training_index + 1 < min_num_days_in_training:
    raise ValueError(
        f"Could not retrieve {min_num_days_in_training} days of data before "
        f"{last_observation_date} from dataset.")

  last_eval_index = forecast_index + num_days_in_evaluation_period
  if last_eval_index >= len(dates_arr):
    logging.info(
        "Could not retrieve %s days of data after %s from dataset. "
        "Evaluation data will be padded with NaNs",
        num_days_in_evaluation_period, last_observation_date)

  return (slice(first_training_index, forecast_index + 1),
          slice(forecast_index + 1, last_eval_index + 1))


def get_dataset(directory: Optional[str],
                dataset_name: str,
                creation_date: Optional[str],
                last_observation_date: Optional[str],
                targets: Sequence[constants.Targets],
                features: Sequence[str],
                cadence: int,
                num_forecast_dates: int,
                allow_dropped_sites: bool = False,
                min_num_training_dates: Optional[int] = None,
                max_num_training_dates: Optional[int] = None) -> Dataset:
  """Gets a dataset.

  Args:
    directory: The directory where the dataset index is stored.
    dataset_name: The name of the dataset (typically the region).
    creation_date: The creation date of the dataset to be used. To use the
      most recently available dataset, pass 'latest'.
    last_observation_date: The last date to include in the training data
      (inclusive).
    targets: The names of the targets to be modelled.
    features: The names of the features to use to predict the targets.
    cadence: The cadence of the data to retrieve. All datasets have daily
      cadence by default. If the cadence is greater than 1, then incremental
      target values are summed over the cadence period, and other features are
      averaged.
    num_forecast_dates: The number of dates after the last_observation_date to
      use for evaluation.
    allow_dropped_sites: Whether to allow sites to be dropped if any of the
      requested features aren't defined for that site for at least one training
      date.
    min_num_training_dates: Optional requirement for a minimum number of dates
      that must be included in the training data. An error is raised if there is
      insufficient data available to satisfy this for the given forecast date.
    max_num_training_dates: Optional setting for the maximum number of dates
      that can be included in the training data up to and including the
      last_observation_date. The training data is truncated to at most this
      number of dates.

  Returns:
    Dataset, containing the following fields:
      training_targets: the targets for the training dates, shape
        (time, site, target)
      training_features: the features for the training dates, shape
        (time, site, feature)
      evaluation_targets: the targets for the evaluation dates, shape
        (time, site, target)
      target_names: the list of target names, corresponding to the final axis
        in the target arrays.
      feature_names: the list of feature names, corresponding to the final axis
        in the feature arrays.
      training_dates: the list of training dates, corresponding to the first
        axis in the training target & feature arrays.
      evaluation_dates: the list of evaluation dates, corresponding to the first
        axis in the evaluation target array.
      sites: the list of site IDs, corresponding to the second axis in the
        target & feature arrays.
  """

  if num_forecast_dates <= 0:
    raise ValueError("At least one future evaluation date must be specified.")

  if cadence < 1:
    raise ValueError("Cadence must be at least daily.")

  targets = sorted([target.value for target in targets])
  features = sorted(features)
  raw_dataset, dataset_index_key = _load_dataset_by_creation_date(
      directory, dataset_name, creation_date)

  data_arrays = df_to_arrays(raw_dataset)

  missing_features = [
      feat for feat in features + targets
      if feat not in data_arrays.features_array
  ]
  if missing_features:
    raise ValueError(
        f"Could not find requested features {missing_features} in dataset")

  num_days_in_evaluation_period = num_forecast_dates * cadence
  max_num_days_in_training = (
      max_num_training_dates
      if not max_num_training_dates else max_num_training_dates * cadence)
  min_num_days_in_training = (
      min_num_training_dates
      if not min_num_training_dates else min_num_training_dates * cadence)

  train_date_slice, eval_date_slice = _get_train_and_eval_date_slices(
      data_arrays.dates_array, last_observation_date,
      min_num_days_in_training, max_num_days_in_training,
      num_days_in_evaluation_period)

  target_indices = np.where(np.in1d(data_arrays.features_array, targets))[0]
  assert len(target_indices) == len(targets)

  feature_indices = np.where(np.in1d(data_arrays.features_array, features))[0]
  assert len(feature_indices) == len(features)

  if pd.isnull(data_arrays.data_array[:, :, target_indices]).any():
    raise ValueError("NaNs found in the target columns.")

  training_features = data_arrays.data_array[
      train_date_slice, :, feature_indices].astype(np.float64)

  dates_str_arr = np.array([
      datetime.datetime.strftime(
          pd.to_datetime(str(date)), constants.DATE_FORMAT)
      for date in data_arrays.dates_array
  ])
  training_targets = data_arrays.data_array[train_date_slice, :,
                                            target_indices].astype(np.float64)

  # We assume our source of data had data from the beginning of pandemic, so
  # sum of past targets is zero.
  sum_past_targets = np.zeros_like(training_targets[0])
  # Plus any discarded initial dates.
  if train_date_slice.start:  # pylint: disable=using-constant-test
    sum_past_targets += data_arrays.data_array[
        :train_date_slice.start, :, target_indices].astype(np.float64).sum(0)

  evaluation_targets = _maybe_pad_data(
      data_arrays.data_array[eval_date_slice, :, target_indices],
      num_days_in_evaluation_period).astype(np.float64)
  evaluation_dates = _get_evaluation_dates(last_observation_date,
                                           num_days_in_evaluation_period)

  # A site is 'valid' to be included in the dataset if all of the requested
  # features are defined for at one least date in the training dates.
  if features:
    features_and_sites_with_at_least_one_date = ~np.all(
        np.isnan(training_features), axis=0)
    # There must be at least one date defined for all features.
    valid_site_indices = np.where(
        np.all(features_and_sites_with_at_least_one_date, axis=1))[0]
  else:
    valid_site_indices = np.arange(len(data_arrays.sites_array))

  if len(valid_site_indices) != len(data_arrays.sites_array):
    sites_to_drop = set(data_arrays.sites_array) - set(
        np.take(data_arrays.sites_array, valid_site_indices))
    if not allow_dropped_sites:
      raise ValueError(f"Found {len(sites_to_drop)} sites where at least 1 "
                       "feature was entirely missing for the requested "
                       f"features {features}. Set allow_dropped_sites to True "
                       "if you want to allow these sites to be dropped.")
    logging.warn(
        "Found %s sites where at least 1 feature was entirely missing "
        "for the requested features %s. These sites are being dropped: %s",
        len(sites_to_drop), ",".join(features), "\n".join(sites_to_drop))

  dataset = Dataset(
      training_targets=np.take(training_targets, valid_site_indices, axis=1),
      training_features=np.take(training_features, valid_site_indices, axis=1),
      evaluation_targets=np.take(
          evaluation_targets, valid_site_indices, axis=1),
      sum_past_targets=np.take(
          sum_past_targets, valid_site_indices, axis=0),
      target_names=np.array(targets),
      feature_names=np.array(features),
      training_dates=dates_str_arr[train_date_slice],
      evaluation_dates=evaluation_dates,
      sites=np.take(data_arrays.sites_array, valid_site_indices),
      dataset_index_key=dataset_index_key,
      cadence=1)

  if cadence > 1:
    dataset = _downsample_dataset_in_time(dataset, cadence)

  return dataset


def _downsample_dataset_in_time(dataset: Dataset, cadence: int) -> Dataset:
  """Downsamples a dataset in time according to the required cadence."""
  # Crop remainder, removing data from the beginning.
  remainder = len(dataset.training_dates) % cadence
  training_dates = dataset.training_dates[remainder:]
  # Select the last date from each group as the data to be used as label for
  # the downsample data. For example, if data is downsampled by putting
  # data from Tuesday to Monday together, Monday will be used as the date for
  # the data point.
  training_dates = training_dates[cadence - 1::cadence]
  evaluation_dates = dataset.evaluation_dates[cadence - 1::cadence]

  training_features = _downsample_features(
      dataset.training_features[remainder:], training_dates,
      dataset.feature_names, cadence)
  training_targets = _downsample_features(
      dataset.training_targets[remainder:], training_dates,
      dataset.target_names, cadence)
  evaluation_targets = _downsample_features(
      dataset.evaluation_targets, evaluation_dates,
      dataset.target_names, cadence)

  sum_past_targets = dataset.sum_past_targets
  if remainder:
    sum_past_targets += dataset.training_targets[:remainder].sum(0)

  return Dataset(training_targets=training_targets,
                 evaluation_targets=evaluation_targets,
                 training_features=training_features,
                 sum_past_targets=sum_past_targets,
                 feature_names=dataset.feature_names,
                 target_names=dataset.target_names,
                 training_dates=training_dates,
                 evaluation_dates=evaluation_dates,
                 sites=dataset.sites,
                 dataset_index_key=dataset.dataset_index_key,
                 cadence=cadence)


def _downsample_features(features: np.ndarray, downsampled_dates: np.ndarray,
                         feature_names: np.ndarray, cadence: int):
  """Downsamples an array of features according to the downsampled dates."""
  if not feature_names.size:
    return features[::cadence]

  # Reshape the features into [downsampled_dates, cadence, ....]
  reshaped_features = features.reshape(
      [len(downsampled_dates), cadence, -1, len(feature_names)])

  output_values = []
  for feature_index, feature_name in enumerate(feature_names):
    feature_values = reshaped_features[..., feature_index]
    if feature_name in [target.value for target in constants.Targets]:
      # Accumulate incremental target features.
      summarized_values = feature_values.sum(axis=1)
    else:
      # Take the mean otherwise.
      summarized_values = feature_values.mean(axis=1)
    output_values.append(summarized_values)

  # Rebuild the data array.
  output_values = np.stack(output_values, axis=-1)
  return output_values


def _maybe_pad_data(data: np.ndarray, required_num_days: int) -> np.ndarray:
  """Maybe pads the date axis of a data array up the required number of days."""
  num_dates, num_sites, num_targets = data.shape
  padding = np.full((required_num_days - num_dates, num_sites, num_targets),
                    np.nan)
  return np.concatenate([data, padding], axis=0)


def _get_evaluation_dates(last_observation_date: str,
                          num_days_in_evaluation_period: int) -> np.ndarray:
  """Gets num_days_in_evaluation_period post last_observation_date."""
  last_observation_datetime = datetime.datetime.strptime(
      last_observation_date, constants.DATE_FORMAT)
  eval_datetimes = [
      last_observation_datetime + datetime.timedelta(days=1 + eval_days_ahead)
      for eval_days_ahead in range(num_days_in_evaluation_period)
  ]
  return np.array([
      datetime.datetime.strftime(eval_datetime, constants.DATE_FORMAT)
      for eval_datetime in eval_datetimes
  ])
