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
"""Utilities for models to use."""

from typing import Generator, Mapping, Sequence, Tuple

from dm_c19_modelling.modelling import definitions

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _get_date_slice(date_indices: np.ndarray, window: int) -> np.ndarray:
  date_slice = date_indices[:, None] + np.arange(window)[None]
  assert date_slice.shape == (date_indices.size,
                              window), "Wrong date slice shape"
  return date_slice


def _get_site_slice(site_indices: np.ndarray, window: int) -> np.ndarray:
  site_slice = np.repeat(site_indices[:, None], window, axis=1)
  assert site_slice.shape == (site_indices.size,
                              window), "Wrong site slice shape"
  return site_slice


def _get_sequences(dataset: definitions.TrainingDataset,
                   date_indices: np.ndarray,
                   site_indices: np.ndarray,
                   window: int) -> Tuple[np.ndarray, np.ndarray]:
  date_slice = _get_date_slice(date_indices, window)  # [batch_size x window]
  site_slice = _get_site_slice(site_indices, window)  # [batch_size x window]
  inputs = dataset.features[date_slice, site_slice]
  targets = dataset.targets[date_slice, site_slice]
  return inputs, targets


def build_training_generator(
    rand: np.random.RandomState, dataset: definitions.TrainingDataset,
    batch_size: int, window: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
  """Yields batches of [inputs, targets] sequences for a dataset."""
  # date_range is the set of dates to predict targets from. Since predictions
  # require a window of input observations, and predict a window of targets,
  # the valid dates start at the end of the earliest input window and end before
  # the latest target window.
  date_range = np.arange(dataset.num_dates - window + 1)
  # site_range is the set of sites to predict over.
  site_range = np.arange(dataset.num_sites)
  while True:
    date_indices = rand.choice(date_range, size=batch_size, replace=True)
    site_indices = rand.choice(site_range, size=batch_size, replace=True)
    yield _get_sequences(dataset, date_indices, site_indices, window)


def prepare_features(features: np.ndarray, feature_names: Sequence[str],
                     feature_stats: definitions.Stats,
                     categorical_features_dict: Mapping[str, int],
                     features_with_missing_values: Sequence[str],
                     dtype=np.float32) -> np.ndarray:
  """Prepares features for the input of a neural network.

  Transforms categorical features into one-hots, and normalizes non-categorical
  features.

  Args:
    features: Array of shape `leading_shape` + [num_features].
    feature_names: sequence of length `num_features`, containing the names of
        the features.
    feature_stats: statistics of the input features.
    categorical_features_dict: Dictionary mapping feature_name -> num_categories
        indicating which features are categorical. Categorical features will
        be prepared as one-hot's.
    features_with_missing_values: List of feature names indicating which
        features have missing values.
    dtype: Type of the prepared features.

  Returns:
    Array of shape `leading_shape` + [num_prepared_features]

  """

  for name in categorical_features_dict.keys():
    if name not in feature_names:
      raise ValueError(
          f"Unrecognized categorical feature '{name}', should be one "
          f"of {feature_names}")

  # TODO(alvaro): Maybe avoid python loop to make it more efficient.
  normalized_features = normalize(features, feature_stats)
  prepared_features = []
  for feature_index, name in enumerate(feature_names):
    if name in categorical_features_dict:
      num_categories = categorical_features_dict[name]
      feature = features[..., feature_index]
      prepared_feature = jax.nn.one_hot(
          feature.astype(np.int32), num_categories, axis=-1)
    else:
      prepared_feature = normalized_features[..., feature_index][..., None]
      if name in features_with_missing_values:
        prepared_feature, missingness_mask = _remove_nans_and_get_mask(
            prepared_feature)
        prepared_features.append(missingness_mask.astype(dtype))
    prepared_features.append(prepared_feature.astype(dtype))

  if not prepared_features:
    raise ValueError("No features available.")

  return jnp.concatenate(prepared_features, axis=-1)


def _get_safe_std(std, threshold=1e-8):
  safe_std = np.array(std)
  mask = np.isclose(std, 0, atol=threshold)
  safe_std[mask] = 1.0
  return safe_std


def normalize(features: np.ndarray, stats: definitions.Stats) -> np.ndarray:
  return (features - stats.mean) / _get_safe_std(stats.std)


def denormalize(normalized_features: np.ndarray,
                stats: definitions.Stats) -> np.ndarray:
  return normalized_features * _get_safe_std(stats.std) + stats.mean


def _remove_nans_and_get_mask(features: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
  """Replaces NaNs in features with 0s and adds mask to indicate missingness."""
  nan_feature_locations = jnp.isnan(features)
  mask = jnp.where(nan_feature_locations, 1., 0.)
  features = jnp.where(nan_feature_locations, 0., features)
  return features, mask


DEFAULT_CONSTANT_FEATURE_NAMES = (definitions.SITE_ID_INTEGER,
                                  definitions.POPULATION)


def rollout_features_with_predictions(
    features: np.ndarray, next_steps_targets: np.ndarray,
    feature_names: Sequence[str], target_names: Sequence[str],
    cadence: int, constant_features: Sequence[str] = ()):
  """Augments input features with target predictions for next steps.

  Args:
    features: array of shape [num_dates, num_sites, num_features]
    next_steps_targets: [num_future_dates, num_sites, num_targets]
    feature_names: names of the input features.
    target_names: names of the target features.
    cadence: cadence for each step in days.
    constant_features: features that can be simply rollout as constants.

  Returns:
    Array of shape `[num_dates + num_future_dates, num_sites, num_features]`
    constructed by concatenating the features, with an augmented version of the
    targets that includes constant features, and features that can be trivially
    rolled-out (e.g. day of the week).

  """

  constant_features = tuple(constant_features) + tuple(
      DEFAULT_CONSTANT_FEATURE_NAMES)

  assert len(feature_names) == features.shape[-1]
  assert len(target_names) == next_steps_targets.shape[-1]

  # Verify that we have enough information to do the rollout.
  missing_features = (
      set(feature_names) -
      set(list(target_names) +
          list(constant_features) +
          [definitions.WEEK_DAY_INTEGER]))

  if missing_features:
    raise ValueError(f"Cannot rollout using features {missing_features} "
                     "which are not constant, neither targets.")

  num_future_dates = next_steps_targets.shape[0]
  rollout_features = []
  for feature_index, name in enumerate(feature_names):
    if name == definitions.WEEK_DAY_INTEGER:
      # Take the last weekday and increment it for each future step.
      last_weekday = features[-1, ..., feature_index].astype(np.int32)
      future_day_index = np.arange(1, num_future_dates + 1) * cadence
      future_weekdays = future_day_index[:, None] + last_weekday[None, :]
      future_weekdays = jnp.mod(future_weekdays, 7)
      rollout_feature = future_weekdays.astype(features.dtype)
    elif name in target_names:
      # Copy the targets.
      rollout_feature = next_steps_targets[..., list(target_names).index(name)]
    elif name in constant_features:
      # Copy features from the last day and broadcast to all dates.
      last_features = features[-1, ..., feature_index]
      rollout_feature = jnp.tile(
          last_features[None], [num_future_dates, 1])
    else:
      # This should never happen, regardless of the inputs, since we already
      # have a check for missing features before the loop.
      raise ValueError(f"Cannot rollout feature {name} which is not constant"
                       "or a target.")
    rollout_features.append(rollout_feature)

  rollout_features = jnp.stack(rollout_features, axis=-1)
  return jnp.concatenate([features, rollout_features], axis=0)


# TODO(alvarosg): Consider removing this (which would simplify the get_optimizer
# method, as if we do not support annelaing, there is no reason for optimizers
# to return auxiliary outputs).
def exponential_annealing(step, start_value, end_value, decay_rate,
                          num_steps_decay_rate):
  """Bridges the gap between start_value and end_value exponentially."""
  progress = decay_rate**(step / num_steps_decay_rate)
  return end_value + (start_value - end_value) * progress


# TODO(alvarosg): Decide if we want to use enums, and use them throughout.
_ANNELING_FNS_MAP = {
    "exponential": exponential_annealing,
}


def get_annealing(global_step, name, **kwargs):
  return _ANNELING_FNS_MAP[name](global_step, **kwargs)


def get_optimizer(name, **kwargs):
  """Returns init_fn, update_fn, aux_outputs for an `optax` optimizer."""
  # We will return the optimizer params, so we can monitor things like
  # annealing of parameters.
  aux_outputs = {name + "_" + k: v for k, v in kwargs.items()}
  return getattr(optax, name)(**kwargs), aux_outputs


def get_optimizer_with_learning_rate_annealing(
    global_step, optimizer_kwargs, annealing_kwargs):

  learning_rate = get_annealing(global_step, **annealing_kwargs)
  optimizer_kwargs = dict(learning_rate=learning_rate, **optimizer_kwargs)
  return get_optimizer(**optimizer_kwargs)


def get_optimizer_params_update_step(loss_fn, optimizer_fn):
  """Returns a jittable fn to update model parameters.

  Args:
    loss_fn: Function that returns the scalar loss with signature:
         loss_fn(trainable_params, non_trainable_state, rng, data) ->
             (scalar_loss, non_trainable_state, aux_outputs)

    optimizer_fn: Function that returns an `optax` optimizer with signature:
         optimizer_fn(global_step) -> (optax_optimizer, aux_outputs)

  Returns:

    Function with signature:
        update_fn(global_step, optimizer_state, trainable_params,
                  non_trainable_state, rng, data) ->
            (updated_optimizer_state, updated_params, loss,
             aux_loss_outputs, aux_optimizer_outputs)

  """

  def update_fn(global_step, optimizer_state, trainable_params,
                non_trainable_state, rng, data):

    # `loss_fn` returns (scalar_loss, non_trainable_state, aux_outputs)
    # but `jax.value_and_grad(loss_fn, has_aux=True)` requires the output to
    # be (scalar_loss, rest). So we apply a small transform to pack it as:
    # (scalar_loss, (non_trainable_state, aux_outputs))

    def loss_fn_with_aux(*args, **kwargs):
      scalar_loss, non_trainable_state, aux_outputs = loss_fn(*args, **kwargs)
      return (scalar_loss, (non_trainable_state, aux_outputs))

    # Compute the loss and gradients.
    (loss, (updated_non_trainable_state, aux_loss_outputs)
     ), grads = jax.value_and_grad(loss_fn_with_aux, has_aux=True)(
         trainable_params, non_trainable_state, rng, data)

    # Get the optimizer fn.
    (_, opt_update_fn), aux_optimizer_outputs = optimizer_fn(global_step)

    # Compute the update params and optimizer state.
    updates, updated_optimizer_state = opt_update_fn(grads, optimizer_state)
    updated_params = optax.apply_updates(trainable_params, updates)

    return (updated_optimizer_state, updated_params,
            updated_non_trainable_state, loss,
            aux_loss_outputs, aux_optimizer_outputs)

  return update_fn
