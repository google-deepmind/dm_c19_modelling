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
"""Tools for indexing forecasts."""

import datetime
import os
from typing import Any, Dict, Sequence, Optional, Union

from absl import logging
from dm_c19_modelling.evaluation import base_indexing
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import dataset_indexing

import numpy as np
import pandas as pd

# Internal imports.


class ForecastIndex(base_indexing.BaseIndex):
  """Manages loading, querying, and adding entries to an index of forecasts."""

  @property
  def _index_type(self):
    return "forecast"

  @property
  def _additional_fields(self):
    return ("last_observation_date", "forecast_id", "cadence", "features_used",)

  def load_file_by_key(self,
                       key: str,
                       validate: bool = True) -> pd.DataFrame:
    """Loads the file contained in the index entry with the given key."""
    entry = self.get_entry(key)
    file_location = entry["file_location"]
    if validate:
      base_indexing.validate_path(file_location)
    logging.info("Loading forecasts from %s", file_location)
    with open(file_location, "r") as fid:
      return pd.read_csv(fid, keep_default_na=False,
                         na_values=[""], dtype={constants.SITE_ID: str})

  def _validate_file_in_entry(self,
                              entry: base_indexing.IndexEntryType) -> None:
    """Validates that contents of forecasts adhere to the expected format."""
    file_location = entry["file_location"]

    with open(file_location, "r") as fid:
      df = pd.read_csv(fid, keep_default_na=False,
                       na_values=[""], dtype={constants.SITE_ID: str})
    required_columns = set([constants.PREDICTION, constants.DATE,
                            constants.SITE_ID, constants.TARGET_NAME])
    if set(required_columns) != set(df.columns):
      raise ValueError(
          f"Forecasts must have columns: {', '.join(sorted(required_columns))}."
          f" Has columns: {', '.join(sorted(df.columns))}")

    if pd.isnull(df[constants.PREDICTION]).any():
      raise ValueError("NaNs founds in forecasts")

    for _, preds_per_site_target in df.groupby(
        [constants.SITE_ID, constants.TARGET_NAME]):
      # Check that the diff in dates for all but the first element is always
      # the same (pandas computes a backwards diff and returns NaN for the first
      # element.
      date_diffs = pd.to_datetime(preds_per_site_target[constants.DATE]).diff()

      if len(date_diffs) > 1 and not (
          date_diffs.iloc[1:] == pd.Timedelta(entry["cadence"], "D")).all():
        raise ValueError("Inconsistent cadence found in forecasts")

    if pd.pivot_table(
        df,
        index=[constants.DATE, constants.SITE_ID, constants.TARGET_NAME],
        dropna=False).isna().any().any():
      raise ValueError("Missing data found in the forecasts: at least one site "
                       "does not have forecasts for all the evaluation dates "
                       "and all of the targets.")

    for target_name in df["target_name"].unique():
      try:
        constants.Targets(target_name)
      except ValueError:
        raise ValueError(f"Invalid target in forecasts: {target_name}")

  def query_by_forecast_id(self, forecast_id: str) -> Union[str, None]:
    """Gets the key in the index corresponding to the given forecast ID."""
    if forecast_id in self._index_dict:
      return forecast_id
    else:
      return None


def build_predictions_df(predictions: np.ndarray, dates: np.ndarray,
                         sites: np.ndarray,
                         target_names: np.ndarray) -> pd.DataFrame:
  """Builds a dataframe of predictions per site, date and target.

  Args:
    predictions: an array of shape (num_forecast_dates, num_sites, num_targets)
      containing model predictions for the evaluation dates.
    dates: an array of shape (num_forecast_dates), specifying the evaluation
      dates.
    sites: an array of shape (num_sites), specifying the site IDs.
    target_names: an array of shape (num_targets), specifying the names of the
      targets which are being predicted.
  Returns:
    A dataframe with columns ("date", "site_id", "target_name", "prediction")
  """

  expected_predictions_shape = (len(dates), len(sites), len(target_names))
  if not np.equal(predictions.shape, expected_predictions_shape).all():
    raise ValueError(f"Predictions have unexpected shape {predictions.shape}. "
                     f"Expected {expected_predictions_shape}")

  # Construct a dataframe of predictions for each target then concatenate them
  target_dfs = []
  for idx, target_name in enumerate(target_names):
    target_df = pd.DataFrame(data=predictions[:, :, idx], columns=sites)
    target_df[constants.DATE] = dates
    target_df = target_df.melt(
        id_vars=constants.DATE,
        value_vars=sites,
        var_name=constants.SITE_ID,
        value_name=constants.PREDICTION)
    target_df[constants.TARGET_NAME] = target_name
    target_dfs.append(target_df)
  df = pd.concat(target_dfs)
  return df


def build_entry(forecast_id: str, file_location: str, dataset_name: str,
                last_observation_date: str, creation_timestamp: str,
                dataset_index_key: str, dataset_location: str, cadence: int,
                extra_info: Dict[str, Any],
                features_used: Optional[Sequence[str]] = None,
                ) -> base_indexing.IndexEntryType:
  """Builds an entry into a forecast index.

  Args:
    forecast_id: the unique identifier of the forecasts.
    file_location: the path to the forecasts on disk.
    dataset_name: the name of the dataset that the forecasts refer to.
    last_observation_date: the last date of ground truth that was used to train
      the model.
    creation_timestamp: the datetime at which the forecasts were created.
    dataset_index_key: the key into the dataset index of the dataset that
      was used to train the model.
    dataset_location: the path to the dataset file that the model was trained
      on.
    cadence: the cadence in days of the predictions. i.e. daily predictions have
      a cadence of 1, weekly predictions have a cadence of 7.
    extra_info: any extra information that is useful to store alongside the
      rest of the forecast metadata. Usually includes the a description of the
      model.
    features_used: the features that were used as inputs to produce the
      forecasts.
  Returns:
    An entry for this forecast that can be added to the forecast index.
  """

  return {
      "forecast_id": forecast_id,
      "file_location": file_location,
      "dataset_name": dataset_name,
      "last_observation_date": last_observation_date,
      "cadence": cadence,
      "creation_timestamp": creation_timestamp,
      "source_data_info": {"dataset_key": dataset_index_key,
                           "dataset_location": dataset_location},
      "features_used": features_used if features_used else "N/A",
      "extra_info": extra_info
  }


def save_predictions_df(predictions_df: np.ndarray,
                        directory: str,
                        last_observation_date: str,
                        forecast_horizon: int,
                        model_description: Optional[Dict[str, str]],
                        dataset_name: str,
                        dataset_index_key: str,
                        cadence: int,
                        extra_info: Optional[Dict[str, str]],
                        features_used: Optional[Sequence[str]] = None) -> str:
  """Saves a formatted predictions dataframe and updates a forecast indexer.

  Args:
    predictions_df: a dataframe of predictions, with columns ['date', 'site_id',
      'prediction', 'target_name']
    directory: the base directory to store indexes and forecasts.
    last_observation_date: the date string corresponding to the last date of
      data that the model had access to during training.
    forecast_horizon: the maximum number of days into the future that the model
      predicts.
    model_description: optional description of the model.
    dataset_name: the name of the dataset.
    dataset_index_key: the unique key into the dataset index that contains the
      training dataset that the model was trained on.
    cadence: the cadence in days of the predictions. i.e. daily predictions have
      a cadence of 1, weekly predictions have a cadence of 7.
    extra_info: a dict of any additional information to store with the
      forecasts.
    features_used: the features that were used as inputs to produce the
      forecasts.

  Returns:
    the unique forecast ID that this forecast is saved under.
  """
  unique_key = base_indexing.get_unique_key()
  forecast_directory = os.path.join(directory, "forecasts")
  if not os.path.exists(forecast_directory):
    os.makedirs(forecast_directory)
  output_filepath = os.path.join(forecast_directory,
                                 f"forecasts_{unique_key}.csv")
  assert not os.path.exists(output_filepath), (
      f"Forecasts already exist at {output_filepath}")

  with open(output_filepath, "w") as fid:
    predictions_df.to_csv(fid, index=False)
  logging.info("Saved model forecasts with forecast ID %s to %s", unique_key,
               output_filepath)

  extra_info = extra_info or {}
  extra_info["forecast_horizon"] = forecast_horizon

  if model_description is not None:
    extra_info["model_description"] = model_description

  current_datetime = datetime.datetime.utcnow()

  dataset_index = dataset_indexing.DatasetIndex(directory, dataset_name)
  dataset_location = dataset_index.get_entry(dataset_index_key)["file_location"]
  entry = build_entry(
      forecast_id=unique_key,
      file_location=output_filepath,
      dataset_name=dataset_name,
      last_observation_date=last_observation_date,
      creation_timestamp=current_datetime.strftime(constants.DATETIME_FORMAT),
      dataset_index_key=dataset_index_key,
      dataset_location=dataset_location,
      cadence=cadence,
      features_used=features_used,
      extra_info=extra_info)

  base_indexing.open_index_and_add_entry(
      directory, dataset_name, index_class=ForecastIndex, key=unique_key,
      entry=entry)

  return unique_key
