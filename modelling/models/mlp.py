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
"""MLP model."""

import typing
from typing import Any, Dict, Generator, Sequence, Tuple

from dm_c19_modelling.modelling.models import base_jax
from dm_c19_modelling.modelling.models import utils
import haiku as hk
import jax.numpy as jnp
import numpy as np

if typing.TYPE_CHECKING:
  from dm_c19_modelling.modelling.definitions import TrainingDataset


class MLPModel(base_jax.LossMinimizerHaikuModel):
  """MLP model."""

  def __init__(self, layer_sizes: Sequence[int], input_window: int,
               batch_size: int, batch_generator_seed: int,
               **parent_kwargs):  # pylint: disable=g-doc-args
    """Constructor."""
    super().__init__(**parent_kwargs)
    self._batch_generator_seed = batch_generator_seed
    self._layer_sizes = layer_sizes
    self._output_feature_size = len(self._dataset_spec.target_names)
    self._input_window = input_window
    self._batch_size = batch_size

  def _build_training_generator(
      self, dataset: "TrainingDataset"
  ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    rand = np.random.RandomState(seed=self._batch_generator_seed)

    batch_gen = utils.build_training_generator(
        rand,
        dataset,
        batch_size=self._batch_size,
        window=self._input_window + self._dataset_spec.num_forecast_dates)

    while True:
      features, targets = next(batch_gen)
      yield features[:, :self._input_window], targets[:, self._input_window:]

  def _build_network(self) -> hk.Sequential:
    """Builds network."""
    initial_reshape = hk.Reshape(output_shape=(-1,))
    mlp = hk.nets.MLP(self._layer_sizes, activate_final=True)
    output_layer = hk.Linear(self._dataset_spec.num_forecast_dates *
                             self._output_feature_size)
    final_reshape = hk.Reshape(
        output_shape=(self._dataset_spec.num_forecast_dates,
                      self._output_feature_size))
    sequential = hk.Sequential(
        [self._prepare_features, initial_reshape, mlp, output_layer,
         final_reshape])
    return sequential

  def _loss_fn(
      self, batch: Tuple[np.ndarray,
                         np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Computes loss."""
    inputs, targets = batch
    network = self._build_network()
    normalized_predictions = network(inputs)

    # Build loss in normalize space.
    normalized_targets = utils.normalize(targets,
                                         self._dataset_spec.target_stats)
    l2_error = (normalized_predictions - normalized_targets)**2
    loss = jnp.mean(l2_error)
    scalars_dict = {"std": jnp.std(l2_error)}  # Customize this for logging.
    return loss, scalars_dict

  def _prepare_predict_fn_inputs(self, dataset):
    # Returns the features for the last input window.
    return dataset.features[-self._input_window:]

  def _predict_fn(self, inputs: np.ndarray) -> Tuple[np.ndarray, Any]:
    """Makes a prediction using the inputs."""

    # [num_dates, num_sites, ...] -> [num_sites, num_dates, ...]
    inputs = jnp.swapaxes(inputs, 0, 1)

    network = self._build_network()
    normalized_predictions = network(inputs)

    # Denormalize the output of the network.
    predictions = utils.denormalize(normalized_predictions,
                                    self._dataset_spec.target_stats)
    # [num_sites, num_dates, ...] -> [num_dates, num_sites, ...]
    aux_data = {}
    return jnp.swapaxes(predictions, 0, 1), aux_data
