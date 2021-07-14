# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Dataset factory."""

import datetime

from typing import Sequence, Tuple

from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import dataset_factory
from dm_c19_modelling.modelling import definitions

import numpy as np


def get_training_dataset(
    **evaluation_dataset_factory_kwargs) -> definitions.TrainingDataset:
  """Gets a dataset."""
  training_dataset = definitions.TrainingDataset.from_dataset(
      dataset_factory.get_dataset(**evaluation_dataset_factory_kwargs))
  training_dataset = _add_site_id_feature(training_dataset)
  return _add_day_of_week_feature(training_dataset)


def _add_day_of_week_feature(dataset: definitions.TrainingDataset):
  """Adds an integer day of the week feature to the data."""
  integer_days_of_the_week = np.array([  # From 0 (Monday) to 6 (Sunday)
      datetime.datetime.strptime(date, constants.DATE_FORMAT).weekday()
      for date in dataset.dates])

  # Broadcast from [num_dates] -> [num_dates, num_sites, 1]
  integer_days_of_the_week = np.tile(
      integer_days_of_the_week[:, None, None], [1, dataset.num_sites, 1])

  return _append_features(
      dataset, integer_days_of_the_week, [definitions.WEEK_DAY_INTEGER])


def _add_site_id_feature(dataset: definitions.TrainingDataset):
  """Adds an integer site id feature to the data."""
  integer_site_ids = np.arange(dataset.num_sites)

  # Broadcast from [num_sites] -> [num_dates, num_sites, 1]
  integer_site_ids = np.tile(
      integer_site_ids[None, :, None], [dataset.num_dates, 1, 1])

  return _append_features(
      dataset, integer_site_ids, [definitions.SITE_ID_INTEGER])


def _append_features(
    dataset: definitions.TrainingDataset, new_features: np.ndarray,
    feature_names: Sequence[str]):
  updated_features = np.concatenate(
      [dataset.features, new_features.astype(dataset.features.dtype)],
      axis=-1)
  updated_feature_names = np.concatenate(
      [dataset.feature_names, feature_names], axis=0)
  return dataset._replace(features=updated_features,
                          feature_names=updated_feature_names)


def remove_validation_dates(
    dataset: definitions.TrainingDataset) -> Tuple[
        definitions.TrainingDataset, np.ndarray]:
  """Generates training and eval datasets.

  Args:
    dataset: `definitions.TrainingDataset` to split.

  Returns:
    Tuple with:
      dataset_without_validation_dates: `definitions.TrainingDataset` where
          the last `num_forecast_dates` worth of data have been removed.
      forecast_targets_validation: targets for the last `num_forecast_dates`
          that have been removed.

  """

  num_forecast_dates = len(dataset.evaluation_dates)

  # We build something that looks like a dataset, but shifted by
  # `num_forecast_dates`, into the past, and keeping the targets for the last
  # `num_forecast_dates` for validation.
  forecast_targets_validation = dataset.targets[-num_forecast_dates:]
  dataset_without_validation_dates = dataset._replace(
      targets=dataset.targets[:-num_forecast_dates],
      features=dataset.features[:-num_forecast_dates],
      dates=dataset.dates[:-num_forecast_dates],
      # As we remove inputs, the index key would no longer be consistent.
      dataset_index_key=None,
      )

  return dataset_without_validation_dates, forecast_targets_validation
