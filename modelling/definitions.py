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
"""Interfaces of datasets and models that can be trained with the framework."""


import abc

from typing import NamedTuple, Optional

from dm_c19_modelling.evaluation import dataset_factory
import numpy as np

WEEK_DAY_INTEGER = "week_day_integer"
SITE_ID_INTEGER = "site_id_integer"
POPULATION = "population"


class TrainingDataset(NamedTuple):
  """A training dataset.

  This is analogous to `dataset_factory.Dataset` without access to the targets
  for the evaluation dates, to avoid information leaking.

  """
  targets: np.ndarray  # (num_dates, num_sites, num_targets)
  features: np.ndarray  # (num_dates, num_sites, num_features)
  sum_past_targets: np.ndarray  # (num_sites, num_targets)
  feature_names: np.ndarray
  target_names: np.ndarray
  dates: np.ndarray  # (num_dates,)
  sites: np.ndarray  # (num_sites,)
  evaluation_dates: np.ndarray  # (num_evaluation_dates,)
  dataset_index_key: Optional[str]
  cadence: int

  @property
  def num_dates(self):
    return self.dates.size

  @property
  def num_sites(self):
    return self.sites.size

  @classmethod
  def from_dataset(cls, dataset):
    if not isinstance(dataset, dataset_factory.Dataset):
      raise TypeError("requires `dataset_factory.Dataset` type")
    return cls(
        targets=dataset.training_targets,
        features=dataset.training_features,
        sum_past_targets=dataset.sum_past_targets,
        target_names=dataset.target_names,
        feature_names=dataset.feature_names,
        dates=dataset.training_dates,
        sites=dataset.sites,
        evaluation_dates=dataset.evaluation_dates,
        cadence=dataset.cadence,
        dataset_index_key=dataset.dataset_index_key)


class Stats(NamedTuple):
  mean: np.ndarray
  std: np.ndarray

  @classmethod
  def from_array(cls, data: np.ndarray, reduce_axes):
    return cls(
        mean=np.nanmean(data, reduce_axes),
        std=np.nanstd(data, reduce_axes),)


class TrainingDatasetSpec(NamedTuple):
  """Specification of a dataset.

  A model trained on a DatasetSpec, should be able to operate on any
  other dataset with the same spec.
  """

  feature_names: np.ndarray
  target_names: np.ndarray
  sites: np.ndarray
  num_forecast_dates: int
  cadence: int
  feature_stats: Stats
  target_stats: Stats

  # Hints about past data, that can be useful for initialization of some models.
  # Population per site.
  population: np.ndarray  # (num_sites,)
  # Estimate of the average target value on the first date.
  initial_targets_hint: np.ndarray  # (num_sites, num_features)
  # Estimate of the summed past targets up to (included) the first date.
  initial_sum_targets_hint: np.ndarray  # (num_sites, num_features)
  # The names of features that have values missing.
  features_with_missing_values: np.ndarray

  @classmethod
  def from_dataset(cls, dataset: TrainingDataset):
    """Builds a `TrainingDatasetSpec` from a `TrainingDataset`."""
    if not isinstance(dataset, (TrainingDataset)):
      raise TypeError("requires `TrainingDataset` type.")

    feature_names = list(dataset.feature_names)
    if POPULATION in feature_names:
      population = dataset.features[0, :, feature_names.index(POPULATION)]
    else:
      population = None

    # Look at the average targets for some initial dates, covering at least a
    # 1 week period. Technically, we could do this for just one step, but
    # but daily data is noisy, so it is better to average.
    num_steps_to_average = int(np.ceil(7 // dataset.cadence))
    initial_targets_hint = dataset.targets[:num_steps_to_average].mean(0)

    initial_sum_targets_hint = (
        dataset.sum_past_targets + dataset.targets[0])

    return cls(
        target_names=dataset.target_names,
        feature_names=dataset.feature_names,
        sites=dataset.sites,
        num_forecast_dates=len(dataset.evaluation_dates),
        cadence=dataset.cadence,
        feature_stats=Stats.from_array(dataset.features, reduce_axes=(0, 1)),
        target_stats=Stats.from_array(dataset.targets, reduce_axes=(0, 1)),
        population=population,
        initial_targets_hint=initial_targets_hint,
        initial_sum_targets_hint=initial_sum_targets_hint,
        features_with_missing_values=np.array(feature_names)[np.any(
            np.isnan(dataset.features), axis=(0, 1))])

  def assert_is_compatible(self, dataset: TrainingDataset):
    # TODO(alvarosg): Maybe make into errors if we decide to keep it.
    assert np.all(dataset.feature_names == self.feature_names)
    assert np.all(dataset.target_names == self.target_names)
    assert np.all(dataset.sites == self.sites)
    assert len(dataset.evaluation_dates) == self.num_forecast_dates
    assert dataset.cadence == self.cadence


class TrainableModel(metaclass=abc.ABCMeta):
  """Base class for trainable models on our training framework."""

  def __init__(self, dataset_spec: TrainingDatasetSpec):
    super().__init__()
    self._dataset_spec = dataset_spec

  def build_training_generator(self, dataset: TrainingDataset):
    """Iteratively yields batches of data given a dataset."""
    self._dataset_spec.assert_is_compatible(dataset)
    return self._build_training_generator(dataset)

  @abc.abstractmethod
  def _build_training_generator(self, dataset: TrainingDataset):
    """See `build_training_generator`."""

  @abc.abstractmethod
  def training_update(self, previous_state, batch, global_step):
    """Updates the model.

    Args:
      previous_state: Previous model state.
      batch: batch of data as generated by `build_training_generator`.
      global_step: global step

    Returns:
      A tuple with (updated_state, scalars_dict).

    """

  def evaluate(self, model_state, dataset: TrainingDataset):
    """Computes a future forecast.

    Args:
      model_state: current model state.
      dataset: input dataset for the future forecast.

    Returns:
      A tuple (predictions, aux_data) with a forecast of shape
      [self._num_forecast_dates, num_sites, num_targets] as well as any
      auxiliary data as a second argument.

    """
    self._dataset_spec.assert_is_compatible(dataset)
    return self._evaluate(model_state, dataset)

  @abc.abstractmethod
  def _evaluate(self, model_state, dataset: TrainingDataset):
    """See `evaluate`."""
