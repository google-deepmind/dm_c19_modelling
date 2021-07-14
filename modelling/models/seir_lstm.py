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
"""SEIR LSTM model.

The basic SEIR model's state variables are:

  (S)usceptible_t  Susceptible population as a function of time
  (E)xposed_t  Exposed population as a function of time
  (I)nfectious_t  Infectious population as a function of time
  (R)ecovered_t  Recovered population as a function of time
  (D)eceased_t  Deceased as a function of time

and parameters are:

  S2E   Rate of transmission  (S --> E)
  E2I   Rate of incubation (E --> I)
  I2RD  Rate of infectiousness (I --> R/D)
  ifr   Infection fatality rate [0, 1]

See for reference- https://www.idmod.org/docs/hiv/model-seir.html

This module imposes the following strong assumptions:

* The initial conditions of the model are fitted for each site separately.
* S2E is predicted on a per-day and per-site basis, using an LSTM conditioned
  on daily features, and optionally, the state variables for the previous day.
* E2I, I2RD and ifr, are shared across all sites and dates.

"""

import enum
import typing
from typing import Any, Dict, Generator, Mapping, NamedTuple, Tuple, Union

from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.modelling.models import base_jax
from dm_c19_modelling.modelling.models import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree

if typing.TYPE_CHECKING:
  from dm_c19_modelling.modelling.definitions import TrainingDataset


POPULATION = "population"


class SEIRState(NamedTuple):
  """State of the SEIR differential equation.

  The contents may have different ranks:
  * Scalar floats: Used for initialization.
  * Rank 1 arrays with shape [num_sites] to indicate state per site (shape [1]
    for site broadcasting).
  * Rank 2 arrays with shape [num_dates, num_sites] to indicate a trajectory
    of states per site (shape [num_dates, num_sites] for site broadcasting).
  """

  susceptible: np.ndarray
  exposed: np.ndarray
  infectious: np.ndarray
  recovered: np.ndarray
  deceased: np.ndarray


class SEIRParameters(NamedTuple):
  """Params of the SEIR differential equation. See `SEIRState` for shapes."""
  S2E: np.ndarray
  E2I: np.ndarray
  I2RD: np.ndarray
  ifr: np.ndarray


def _compute_seir_derivatives(
    state: SEIRState, parameters: SEIRParameters) -> SEIRState:
  """Produces time derivatives for each field of the SEIR state."""

  # Susceptible people who get exposed.
  s_to_e = parameters.S2E * state.susceptible * state.infectious
  # Exposed people who become infectious.
  e_to_i = parameters.E2I * state.exposed
  # Infectious people who either recover or die.
  i_to_rd = parameters.I2RD * state.infectious
  # Infectious people who recover.
  i_to_r = i_to_rd * (1.0 - parameters.ifr)
  # Infectious people who die.
  i_to_d = i_to_rd * parameters.ifr

  # Set the derivatives.
  ds = -s_to_e
  de = s_to_e - e_to_i
  di = e_to_i - i_to_rd
  dr = i_to_r
  dd = i_to_d
  return SEIRState(susceptible=ds,
                   exposed=de,
                   infectious=di,
                   recovered=dr,
                   deceased=dd)


def _euler_update(dt: float, state, state_derivative):
  """Integrates a nested state given its derivatives, using Euler integrator."""
  return tree.map_structure(
      lambda field, d_field: field + dt * d_field,
      state, state_derivative)


def _calculate_seir_initialization(dataset_spec,
                                   seir_params_init,
                                   min_exposed_fraction):
  """Infers initial state conditions of the SEIR model from recent stats."""

  # Initialized the SEIR state field by field. Note the SEIR state will be
  # defined as a fraction of the population.
  new_seir_state_init = {}

  # Initialize the cumulative fraction of deceased population.
  new_deceased_index = list(dataset_spec.target_names).index(
      constants.DECEASED_NEW)
  total_num_deceased = (
      dataset_spec.initial_sum_targets_hint[..., new_deceased_index])
  new_seir_state_init["deceased"] = (
      total_num_deceased / dataset_spec.population)

  # Calculate recovered people from the fatality rate.
  ifr = seir_params_init.ifr
  new_seir_state_init["recovered"] = new_seir_state_init["deceased"] / ifr * (
      1 - ifr)

  # Calculate infected proportion from the amount of daily deceased, assuming
  # the value of `seir_params_init.I2RD`
  # Note the targets are accumulated for `dataset_spec.cadence` days, but we
  # need the rate, so we divide by the cadence.
  new_deceased_daily = (
      dataset_spec.initial_targets_hint[..., new_deceased_index] /
      dataset_spec.cadence)
  new_deceased_proportion_daily = (
      new_deceased_daily / dataset_spec.population)
  i2rd = seir_params_init.I2RD
  new_seir_state_init["infectious"] = new_deceased_proportion_daily / ifr / i2rd

  # Set exposed proportion to be the same number as infectious proportion, as
  # these two move together.
  new_seir_state_init["exposed"] = new_seir_state_init["infectious"]

  # However, we still set the exposed proportion to a minimum value, as if it
  # happens to be exactly 0, it would be impossible to leave that state.
  new_seir_state_init["exposed"] = np.maximum(
      new_seir_state_init["exposed"], min_exposed_fraction)
  assert np.all(new_seir_state_init["exposed"] > 0.)

  # The remaining fraction, will be population that are still susceptible.
  new_seir_state_init["susceptible"] = 1 - sum(
      new_seir_state_init.values())
  return SEIRState(**new_seir_state_init)


class ExtrapolationModes(enum.Enum):
  """Extrapolation modes for producing ODE parameters for the horizon period."""

  # Fits a polynomial the last values of the ODE params.
  POLYNOMIAL_FIT_PARAMS = "polynomial_fit_params"

  # Uses the model predictions to produce features for subsequent days.
  # Only possible if only targets and constants are used as features.
  ROLLOUT_FEATURES = "rollout_features"


class SEIRLSTM(base_jax.LossMinimizerHaikuModel):
  """LSTM model.

  It makes predictions into the future by feeding its own predictions as inputs
  for the next step.

  """

  def __init__(self,
               lstm_embedding_size: int,
               lstm_hidden_size: int,
               min_exposed_fraction: float,
               seir_parameters_init: Mapping[str, float],
               condition_lstm_on_ode_state: bool,
               extrapolation_mode: Union[str, ExtrapolationModes],
               param_extrapolation_context_steps: int,
               param_extrapolation_poly_degree: int,
               **parent_kwargs):  # pylint: disable=g-doc-args
    """Constructor.

    Args:
        lstm_embedding_size: Size of linear embedding of the features.
        lstm_hidden_size: Size of the LSTM.
        min_exposed_fraction: Minimum initial fraction of the population exposed
            to the disease. The initial SEIR state is estimated from the initial
            statistics about accumulated deceased. However, if the number is
            zero, the SEIR model would be initialized to 0, and it would not
            be able to leave that state. To avoid this, whe set a lower bound
            on the fraction of exposed population.
        seir_parameters_init: Initialization values for the seir state, packed
            in a dict with `SEIRParameters` fields.
        condition_lstm_on_ode_state: Whether to feed the current ODE state
            as inputs for the LSTM.
        extrapolation_mode: One of `ExtrapolationModes`.
        param_extrapolation_context_steps: Number of steps to fit a polynomial
            to when extrapolating ODE parameters in time. Only for
            `extrapolation_mode = ExtrapolationModes.POLYNOMIAL_FIT_PARAMS`.
        param_extrapolation_poly_degree: Degree of the polynomial to use when
            extrapolating ODE parameters in time. Only for
            `extrapolation_mode = ExtrapolationModes.POLYNOMIAL_FIT_PARAMS`.
        **parent_kwargs: Attributes for the parent class.

    """
    super().__init__(**parent_kwargs)

    self._extrapolation_mode = ExtrapolationModes(extrapolation_mode)
    self._param_extrapolation_poly_degree = param_extrapolation_poly_degree
    self._param_extrapolation_context_steps = param_extrapolation_context_steps

    self._lstm_embedding_size = lstm_embedding_size
    self._lstm_hidden_size = lstm_hidden_size

    # TODO(alvarosg): Maybe support `constants.CONFIRMED_NEW`.
    if tuple(self._dataset_spec.target_names) != (constants.DECEASED_NEW,):
      raise ValueError(f"{constants.DECEASED_NEW} is the only supported target,"
                       f" got {self._dataset_spec.target_names}")

    if POPULATION not in self._dataset_spec.feature_names:
      raise ValueError(
          f"Missing population feature, got {self._dataset_spec.feature_names}")

    self._seir_parameters_init = SEIRParameters(**seir_parameters_init)
    self._condition_lstm_on_ode_state = condition_lstm_on_ode_state

    self._seir_state_init = _calculate_seir_initialization(
        self._dataset_spec, self._seir_parameters_init,
        min_exposed_fraction)

  def _build_training_generator(
      self, dataset: "TrainingDataset"
  ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    while True:
      # Leading shape [num_dates, num_sites]
      yield dataset.features[:-1], dataset.targets[1:]

  def _get_learnable_seir_initial_state(self) -> SEIRState:
    """Returns learnable variables for the initial state of the SEIR model.."""

    num_sites = len(self._dataset_spec.sites)

    logits_state_dict = {}
    for name, init_value in self._seir_state_init._asdict().items():

      logits_state_dict[name] = hk.get_parameter(
          f"log_{name}_initial", shape=[num_sites], dtype=np.float32,
          init=hk.initializers.Constant(np.log(init_value)))

    # Make sure they add up to one.
    logits_state = SEIRState(**logits_state_dict)
    initial_state_array = jax.nn.softmax(
        jnp.stack(tuple(logits_state), axis=0), axis=0)

    return SEIRState(*list(initial_state_array))  # Unstack along first axis.

  def _get_logit_functions(self, param_name):
    if param_name == "ifr":
      # To map the interval [0, 1]
      return jax.scipy.special.logit, jax.nn.sigmoid
    elif param_name in ["E2I", "I2RD", "S2E"]:
      # To map the interval [0, +inf)
      return jnp.log, jnp.exp
    else:
      raise ValueError(f"Param name {param_name}")

  def _get_learnable_seir_params(self) -> SEIRParameters:
    """Returns learnable values for ODE parameters."""
    # Get the fixed values.
    params_dict = {}
    for name, init_value in self._seir_parameters_init._asdict().items():
      if name == "S2E":
        params_dict[name] = None
      else:
        log_fn, exp_fn = self._get_logit_functions(name)
        # Shape [1], to it will be broadcasted to all sites.
        params_dict[name] = exp_fn(hk.get_parameter(
            f"log_{name}_param", shape=[1], dtype=np.float32,
            init=hk.initializers.Constant(log_fn(init_value))))
    return SEIRParameters(**params_dict)

  def _build_rnn(self, name_prefix="S2E") -> hk.DeepRNN:
    """Builds network."""
    return hk.DeepRNN([
        hk.Linear(self._lstm_embedding_size, name=name_prefix + "_encoder"),
        jax.nn.relu,
        hk.LSTM(self._lstm_hidden_size, name=name_prefix + "_lstm"),
        hk.Linear(1, name=name_prefix + "_decoder"),  # Predict `S2E`.
    ])

  def _loss_fn(
      self, batch: Tuple[np.ndarray,
                         np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Computes loss."""
    # features_sequence: [num_training_dates, num_sites, num_features]
    # targets_sequence: [num_training_dates, num_sites, num_targets]
    features_sequence, targets_sequence = batch

    rnn = self._build_rnn()

    seir_state_sequence, _, _ = self._run_model_on_feature_sequence(
        rnn, features_sequence)

    # Get the number of deceased scaled by population. Since population is
    # constant we can just grab it from the first step.
    population = self._get_population(features_sequence[0])
    deceased = seir_state_sequence.deceased * population[None]

    # Go from cumulative deceased to incremental adding a trailing feature axis.
    predictions_sequence = jnp.diff(deceased, axis=0)[..., None]

    # Build loss in normalize space.
    normalized_targets = utils.normalize(targets_sequence,
                                         self._dataset_spec.target_stats)
    normalized_predictions = utils.normalize(predictions_sequence,
                                             self._dataset_spec.target_stats)
    l2_error = (normalized_predictions - normalized_targets) ** 2

    loss = jnp.mean(l2_error)
    scalars_dict = {}
    return loss, scalars_dict

  def _get_population(self, features):
    population_index = list(self._dataset_spec.feature_names).index(POPULATION)
    return features[..., population_index]

  def _run_model_on_feature_sequence(self, rnn, feature_sequence):
    """Runs the model using a sequence of features to feed the S2E RNN."""

    def core(features, state):

      seir_state, rnn_state = state
      # Obtain the SEIR parameters for this step.
      seir_params, updated_rnn_state = self._get_seir_params_for_step_with_rnn(
          rnn, features, rnn_state, seir_state)

      # Forward the SEIR state with the current SEIR params.
      updated_seir_state = self._multi_step_euler_update(
          seir_state, seir_params)

      # Note `hk.dynamic_unroll` expects two outputs.
      # * Elements in the first output will be returned for all steps stacked
      #   along the first axis.
      # * Elements of the second output will be passed to each subsequent
      #   iteration, and ultimately, only the final value will be returned.
      # So in this case we return `updated_seir_state` both in the first output
      # (to be able to get the full trajectory after `hk.dynamic_unroll`) and
      # the second output, to be able to pass the value to the next iteration.
      next_state = updated_seir_state, updated_rnn_state

      return (updated_seir_state, seir_params), next_state

    initial_state_seir = self._get_learnable_seir_initial_state()
    initial_state_rnn = rnn.initial_state(feature_sequence.shape[1])
    ((seir_state_sequence, seir_params_sequence),
     (_, final_rnn_state)) = hk.dynamic_unroll(
         core, feature_sequence,
         initial_state=(initial_state_seir, initial_state_rnn))

    seir_state_sequence = tree.map_structure(
        lambda first, seq: jnp.concatenate([first[None], seq], axis=0),
        initial_state_seir, seir_state_sequence)
    return (seir_state_sequence, seir_params_sequence, final_rnn_state)

  def _multi_step_euler_update(self, seir_state, seir_params):
    # To keep it comparable across values of the cadence, Euler steps will
    # always have a fixed dt=1 day, but we will have `cadence` Euler steps.
    for _ in range(self._dataset_spec.cadence):
      seir_derivatives = _compute_seir_derivatives(seir_state, seir_params)
      seir_state = _euler_update(
          dt=1, state=seir_state, state_derivative=seir_derivatives)
    return seir_state

  def _get_seir_params_for_step_with_rnn(
      self, rnn, features, rnn_state, seir_state):
    """Returns the SEIR state for a step using an RNN for `S2E`."""

    # Get the parameters that are simply learned.
    seir_params = self._get_learnable_seir_params()

    # Build inputs for the rnn model.
    rnn_inputs = self._prepare_features(features)
    if self._condition_lstm_on_ode_state:
      ode_state_features = jnp.stack(tuple(seir_state), axis=1)
      rnn_inputs = jnp.concatenate([rnn_inputs, ode_state_features], axis=1)

    # Add the RNN "S2E" value.
    rnn_output, updated_rnn_state = rnn(rnn_inputs, rnn_state)
    logits_s2e = jnp.squeeze(rnn_output, axis=1)
    _, exp_fn = self._get_logit_functions("S2E")

    # Our prior is that S2E will be close to `self._seir_parameters_init.S2E`,
    # Propertly initialized neural networks produce initial outputs with
    # zero-mean and unit-variance. So on average at the beginning of training
    # the value of S2E will be `exp(logits_s2e) * seir_parameters_init.S2E`
    # for `logits_s2e = 0` will yield `seir_parameters_init.S2E`.
    # This way the LSTM only has to learn a correction to
    # `seir_parameters_init.S2E`
    s2e = exp_fn(logits_s2e) * self._seir_parameters_init.S2E
    seir_params = seir_params._replace(S2E=s2e)
    return seir_params, updated_rnn_state

  def _prepare_predict_fn_inputs(self, dataset):
    # Returns the features.
    return dataset.features

  def _predict_fn(self, inputs: np.ndarray) -> Tuple[np.ndarray, Any]:
    """Makes a prediction using the inputs."""

    # inputs: [num_training_dates, num_sites, num_features]

    num_forecast_dates = self._dataset_spec.num_forecast_dates

    rnn = self._build_rnn()

    # Run LSTM on the input sequence.
    (seir_state_sequence,
     seir_params_sequence,
     rnn_state) = self._run_model_on_feature_sequence(rnn, inputs)

    if self._extrapolation_mode == ExtrapolationModes.POLYNOMIAL_FIT_PARAMS:
      (additional_seir_state_sequence,
       additional_seir_params_sequence
       ) = self._extrapolation_with_polynomial_fit_on_params(
           seir_state=tree.map_structure(lambda x: x[-1], seir_state_sequence),
           seir_params_sequence=seir_params_sequence,
           num_steps=num_forecast_dates - 1)
    elif self._extrapolation_mode == ExtrapolationModes.ROLLOUT_FEATURES:
      (additional_seir_state_sequence,
       additional_seir_params_sequence
       ) = self._extrapolation_with_feature_rollout(
           rnn=rnn,
           seir_state_t=tree.map_structure(
               lambda x: x[-1], seir_state_sequence),
           seir_state_tm1=tree.map_structure(
               lambda x: x[-2], seir_state_sequence),
           rnn_state=rnn_state,
           features_tm1=inputs[-1],
           num_steps=num_forecast_dates - 1)

    # Build the full sequence.
    seir_state_sequence = tree.map_structure(
        lambda a, b: jnp.concatenate([a, b], axis=0),
        seir_state_sequence, additional_seir_state_sequence)
    seir_params_sequence = tree.map_structure(
        lambda a, b: jnp.concatenate([a, b], axis=0),
        seir_params_sequence, additional_seir_params_sequence)

    # Get the number of deceased scaled by population. Since population is
    # constant we can just grab it from the first step.
    population = self._get_population(inputs[0])
    deceased = seir_state_sequence.deceased * population[None]

    # Go from cumulative deceased to incremental adding a trailing feature axis.
    new_deceased = jnp.diff(deceased, axis=0)[..., None]

    # Get the final predictions of interset.
    predictions = new_deceased[-num_forecast_dates:]
    aux_data = {"full_seir_params_sequence": seir_params_sequence,
                "full_seir_state_sequence": seir_state_sequence,
                "predictions_for_inputs": new_deceased[:-num_forecast_dates]}

    return predictions, aux_data

  def _extrapolation_with_polynomial_fit_on_params(
      self, seir_state, seir_params_sequence, num_steps):
    extrapolated_seir_params_sequence = self._extrapolate_seir_params(
        seir_params_sequence, num_steps)

    # Run additional steps, where the initial SEIR state is the last SEIR state
    # from the initial teacher forcing stage.
    return self._run_model_on_seir_params_sequence(
        initial_seir_state=seir_state,
        seir_params_sequence=extrapolated_seir_params_sequence)

  def _run_model_on_seir_params_sequence(
      self, initial_seir_state, seir_params_sequence):
    """Runs the model using a sequence of seir parameters."""

    def core(seir_params, seir_state):
      # Forward the SEIR state with the current SEIR params.
      updated_seir_state = self._multi_step_euler_update(
          seir_state, seir_params)
      return (updated_seir_state, seir_params), updated_seir_state

    (seir_state_sequence, seir_params_sequence), _ = hk.dynamic_unroll(
        core, seir_params_sequence, initial_state=initial_seir_state)

    return seir_state_sequence, seir_params_sequence

  def _extrapolate_seir_params(self, seir_params_sequence, num_steps):
    """Extrapolate SEIR parameters from previous values with polynomial fit."""
    step_index = jnp.arange(
        self._param_extrapolation_context_steps + num_steps,
        dtype=tree.flatten(seir_params_sequence)[0].dtype)

    x_powers = jnp.stack(
        [step_index ** p
         for p in range(self._param_extrapolation_poly_degree + 1)], axis=1)

    # [self._num_context_steps, poly_degree + 1]
    x_context = x_powers[:self._param_extrapolation_context_steps]

    # [num_steps, poly_degree + 1]
    x_extrapolation = x_powers[self._param_extrapolation_context_steps:]

    def fn(param_sequence):

      if param_sequence.shape[0] < self._param_extrapolation_context_steps:
        raise ValueError(
            f"Not enough inputs steps {param_sequence.shape[0]} to extrapolate "
            f"with {self._param_extrapolation_context_steps} steps of context.")

      # [self._num_context_steps, num_sites]
      y_context = param_sequence[-self._param_extrapolation_context_steps:]

      # [poly_degree + 1, num_sites]
      coefficients, _, _, _ = jnp.linalg.lstsq(x_context, y_context)

      # [num_steps, num_sites]
      return jnp.einsum("td,db->tb", x_extrapolation, coefficients)

    return tree.map_structure(fn, seir_params_sequence)

  def _extrapolation_with_feature_rollout(
      self, rnn, seir_state_t, seir_state_tm1, rnn_state,
      features_tm1, num_steps):
    """Rollout own model predictions to produce future model ODE parameters."""

    population = self._get_population(features_tm1)

    def core(unused_step, state):

      features_tm1, seir_state_tm1, seir_state_t, rnn_state = state

      # Compute input features for the next step using the predictions based on
      # previous ODE states.
      # [num_sites]
      new_proportion_deceased = seir_state_t.deceased - seir_state_tm1.deceased
      # [num_sites, num_targets]
      targets = (new_proportion_deceased * population)[..., None]
      # [num_sites, num_feaures]
      features_t = utils.rollout_features_with_predictions(
          features_tm1[None],  # Add a time axis.
          targets[None],  # Add a time axis.
          self._dataset_spec.feature_names,
          self._dataset_spec.target_names,
          self._dataset_spec.cadence,
      )[-1]  # Remove time axis.

      # Obtain the SEIR parameters for this step with the RNN.
      seir_params, updated_rnn_state = self._get_seir_params_for_step_with_rnn(
          rnn, features_t, rnn_state, seir_state_t)

      # Forward the SEIR state with the current SEIR params.
      seir_state_tp1 = self._multi_step_euler_update(
          seir_state_t, seir_params)

      next_state = features_t, seir_state_t, seir_state_tp1, updated_rnn_state

      return (seir_state_tp1, seir_params), next_state

    (seir_state_sequence, seir_params_sequence), _ = hk.dynamic_unroll(
        core, jnp.arange(num_steps),
        initial_state=(
            features_tm1, seir_state_tm1, seir_state_t, rnn_state))

    return seir_state_sequence, seir_params_sequence
