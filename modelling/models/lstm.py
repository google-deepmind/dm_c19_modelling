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
"""LSTM model."""

import typing
from typing import Any, Dict, Generator, Tuple

from dm_c19_modelling.modelling.models import base_jax
from dm_c19_modelling.modelling.models import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

if typing.TYPE_CHECKING:
  from dm_c19_modelling.modelling.definitions import TrainingDataset


class LSTM(base_jax.LossMinimizerHaikuModel):
  """LSTM model.

  It makes predictions into the future by feeding its own predictions as inputs
  for the next step.

  """

  def __init__(self,
               embedding_size: int,
               hidden_size: int,
               batch_size: int,
               warm_up_steps: int,
               batch_generator_seed: int,
               training_sequence_length: int,
               num_context_dates_evaluation: int,
               **parent_kwargs):  # pylint: disable=g-doc-args
    """Constructor.

    Args:
        embedding_size: Size of linear embedding of the features.
        hidden_size: Size of the LSTM.
        batch_size: Batch size.
        warm_up_steps: Initial number of leading steps for which not loss will
            be computed (to allow the model to look at a few steps of data and
            build up the state before making predictions).
        batch_generator_seed: Seed for the batch generator.
        training_sequence_length: Length of the training sequences.
        num_context_dates_evaluation: Number of steps used to warm up the state
            before making predictions during evaluation.
        **parent_kwargs: Attributes for the parent class.

    """
    super().__init__(**parent_kwargs)

    self._embedding_size = embedding_size
    self._hidden_size = hidden_size
    self._batch_generator_seed = batch_generator_seed
    self._warm_up_steps = warm_up_steps
    self._output_feature_size = len(self._dataset_spec.target_names)
    self._batch_size = batch_size
    self._num_context_dates_evaluation = num_context_dates_evaluation
    self._training_sequence_length = training_sequence_length

  def _build_training_generator(
      self, dataset: "TrainingDataset"
  ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    rand = np.random.RandomState(seed=self._batch_generator_seed)

    # We will generate subsequences for teacher forcing, with inputs and targets
    # shifted by 1 date.
    batch_gen = utils.build_training_generator(
        rand,
        dataset,
        batch_size=self._batch_size,
        window=self._training_sequence_length + 1)

    while True:
      features, targets = next(batch_gen)

      # Leading shape [training_sequence_length, batch_size]
      yield (jnp.swapaxes(features[:, :-1], 0, 1),
             jnp.swapaxes(targets[:, 1:], 0, 1))

  def _build_network(self) -> hk.DeepRNN:
    """Builds network."""
    return hk.DeepRNN([
        self._prepare_features,
        hk.Linear(self._embedding_size),
        jax.nn.relu,
        hk.LSTM(self._hidden_size),
        hk.Linear(self._output_feature_size),
    ])

  def _loss_fn(
      self, batch: Tuple[np.ndarray,
                         np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Computes loss."""
    inputs, targets = batch
    rnn = self._build_network()
    initial_state = rnn.initial_state(self._batch_size)
    normalized_predictions, _ = hk.dynamic_unroll(rnn, inputs, initial_state)

    # Build loss in normalize space.
    normalized_targets = utils.normalize(targets,
                                         self._dataset_spec.target_stats)
    l2_error = (normalized_predictions - normalized_targets)**2

    # Ignore loss for the first steps, until the state is built up.
    l2_error = l2_error[self._warm_up_steps:]

    loss = jnp.mean(l2_error)
    scalars_dict = {}
    return loss, scalars_dict

  def _prepare_predict_fn_inputs(self, dataset):
    # Returns the features.
    return dataset.features

  def _predict_fn(self, inputs: np.ndarray) -> Tuple[np.ndarray, Any]:
    """Makes a prediction using the inputs."""

    rnn = self._build_network()
    initial_state = rnn.initial_state(inputs.shape[1])

    # Build up the state using teacher forcing for
    # `self._num_context_dates_evaluation` steps.
    inputs = inputs[-self._num_context_dates_evaluation:]
    normalized_predictions_teacher_forcing, rnn_state = hk.dynamic_unroll(
        rnn, inputs, initial_state)

    # Everything but the last step, corresponds to one-step predictions for the
    # inputs dates.
    predictions_for_inputs = utils.denormalize(
        normalized_predictions_teacher_forcing[:-1],
        self._dataset_spec.target_stats)

    # Initialize the prediction for the evaluation dates to zeros.
    normalized_predictions = jnp.zeros([
        self._dataset_spec.num_forecast_dates,
        inputs.shape[1],
        self._output_feature_size,], dtype=inputs.dtype)

    # Use last prediction from the teacher forcing phase, which will be the
    # first prediction for the evaluation dates.
    normalized_predictions = jax.lax.dynamic_update_index_in_dim(
        normalized_predictions, normalized_predictions_teacher_forcing[-1],
        index=0, axis=0)

    # Rollout the model for `self._dataset_spec.num_forecast_dates - 1` steps.
    def body(x):
      rnn_state, prev_features, normalized_predictions, step = x

      # Build input features for this step using the features for the last
      # step, and the normalized predictions for the last step.
      features = utils.rollout_features_with_predictions(
          prev_features[None],  # Add a time axis.
          utils.denormalize(
              normalized_predictions[step],
              self._dataset_spec.target_stats)[None],  # Add a time axis.
          self._dataset_spec.feature_names,
          self._dataset_spec.target_names,
          self._dataset_spec.cadence,
      )[-1]  # Remove time axis.

      # Run the model and update the corresponding slice.
      normalized_predictions_step, updated_rnn_state = rnn(
          features, rnn_state)
      normalized_predictions = jax.lax.dynamic_update_index_in_dim(
          normalized_predictions, normalized_predictions_step,
          index=step+1, axis=0)

      return updated_rnn_state, features, normalized_predictions, step + 1

    init_values_loop = rnn_state, inputs[-1], normalized_predictions, 0
    _, _, normalized_predictions, _ = jax.lax.while_loop(
        lambda x: x[-1] < self._dataset_spec.num_forecast_dates - 1,
        body,
        init_values_loop)

    # Denormalize the outputs of the network and return.
    predictions = utils.denormalize(normalized_predictions,
                                    self._dataset_spec.target_stats)

    aux_data = {"predictions_for_inputs": predictions_for_inputs}
    return predictions, aux_data
