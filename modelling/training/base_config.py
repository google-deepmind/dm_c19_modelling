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
"""Config for c19 experiment."""

from dm_c19_modelling.evaluation import constants
from ml_collections import config_dict

# TODO(peterbattaglia): These are just for reference: remove later.
_ALL_FEATURES = (
    "new_confirmed", "new_deceased", "new_recovered", "new_tested",
    "total_confirmed", "total_deceased", "total_recovered", "total_tested",
    "population", "population_male", "population_female", "rural_population",
    "urban_population", "largest_city_population", "clustered_population",
    "population_density", "human_development_index", "population_age_00_09",
    "population_age_10_19", "population_age_20_29", "population_age_30_39",
    "population_age_40_49", "population_age_50_59", "population_age_60_69",
    "population_age_70_79", "population_age_80_89", "population_age_90_99",
    "population_age_80_and_older", "mobility_retail_and_recreation",
    "mobility_grocery_and_pharmacy", "mobility_parks",
    "mobility_transit_stations", "mobility_workplaces", "mobility_residential")


def get_config(project_directory):
  """Returns the experiment config.."""
  config = config_dict.ConfigDict()

  # Parameters to build the model and dataset.
  config.dataset = _get_dataset_config(project_directory)
  config.model = _get_model_config()

  # Configuration of the training, evaluation, and fine tuning loops.
  config.training = _get_training_config()
  config.eval = _get_eval_config()
  config.fine_tune = _get_fine_tune_config(config.training)

  # Parameters to checkpoint the model during training.
  config.checkpointer = _get_checkpointer_config()
  return config


def _get_dataset_config(project_directory):
  """Keyword arguments to `dataset_factory.get_dataset`."""
  config = config_dict.ConfigDict()
  config.directory = project_directory
  config.dataset_name = "covid_open_data_world"
  # The name(s) of the target to predict.
  config.targets = [constants.Targets.DECEASED_NEW]
  # The names of the features to use to make predictions.
  # TODO(alvarosg): Should we add an option for models to then select subsets of
  # these features only?
  config.features = [
      "new_confirmed",
      "new_deceased",
      "population",
      "mobility_retail_and_recreation", "mobility_grocery_and_pharmacy",
      "mobility_parks", "mobility_transit_stations", "mobility_workplaces",
      "mobility_residential"
  ]
  # The The creation date of the dataset to use for training.
  config.creation_date = "latest"
  # The date to train up to and evaluate from.
  config.last_observation_date = "2020-12-06"
  # The number of dates to use for evaluation. The forecast horizon is equal
  # to num_forecast_dates * cadence.
  config.num_forecast_dates = 28
  #  The cadence in days of the forecasts.
  config.cadence = 1
  # Whether to allow sites to be dropped if any of the requested features aren't
  # defined for that site for at least one training date.
  config.allow_dropped_sites = False
  return config


def _get_model_config():
  """Keyword arguments to `model_factory.get_model`."""
  config = config_dict.ConfigDict()

  # This is a very specific implementation of a config to work with our model
  # factory. Users may change the config and factory design, such as it is
  # compatible with the factory being called as:
  #     `get_model(dataset_spec, **this_config)`

  # Parameters that are shared by all instances of `LossMinimizerHaikuModel`,
  config.init_seed = 42
  config.training_seed = 42
  config.optimizer_kwargs = dict(
      name="adam",
      b1=0.9,
      b2=0.999,
      eps=1e-8,)
  config.learning_rate_annealing_kwargs = dict(
      name="exponential",
      start_value=1e-3,
      end_value=1e-6,
      num_steps_decay_rate=1e5,
      decay_rate=0.1)

  # Model name and additional specific configs for the models we support.
  config.model_name = "mlp"  # One of those below (e.g. mlp, lstm, seir_lstm)
  config.model_specific_kwargs = dict(
      mlp=_get_mlp_model_config(),
      lstm=_get_lstm_model_config(),
      seir_lstm=_get_seir_lstm_model_config(),
      )
  return config


def _get_mlp_model_config():
  """Returns MLP model config."""
  config = config_dict.ConfigDict()
  # Eventually this will probably stop being a model specific parameter
  # and instead be on the final base class.
  config.batch_generator_seed = 42
  config.layer_sizes = (128, 128,)
  config.input_window = 21
  config.batch_size = 128
  return config


def _get_lstm_model_config():
  """Returns LSTM model config."""
  # Note currently this model only works if the model input features are
  # constants (e.g. population), are trivially predictable (e.g. day of week) or
  # can be built from targets (e.g. number of deceased).
  config = config_dict.ConfigDict()
  config.embedding_size = 32
  config.hidden_size = 32
  config.batch_size = 64
  config.warm_up_steps = 14
  config.batch_generator_seed = 42
  config.training_sequence_length = 56
  config.num_context_dates_evaluation = 28
  return config


def _get_seir_lstm_model_config():
  """Returns LSTM model config."""
  config = config_dict.ConfigDict()
  config.lstm_embedding_size = 32
  config.lstm_hidden_size = 32
  config.condition_lstm_on_ode_state = True
  config.min_exposed_fraction = 5e-6
  # One of: ["rollout_features", "polynomial_fit_params"]
  # Note `rollout_features` only works if the model input features are constants
  # (e.g. population), are trivially predictable (e.g. day of week) or can be
  # built from targets (e.g. number of deceased).
  config.extrapolation_mode = "polynomial_fit_params"
  config.param_extrapolation_context_steps = 21
  config.param_extrapolation_poly_degree = 1
  config.seir_parameters_init = dict(
      S2E=0.8,  # An LSTM will be used to modulate this.
      E2I=0.2,
      I2RD=0.5,
      ifr=0.01,
  )
  return config


def _get_training_config():
  """Keyword arguments to `train_loop.train`."""
  config = config_dict.ConfigDict()
  config.training_steps = 10000
  config.log_interval = 100
  config.checkpoint_interval = 1000
  return config


def _get_eval_config():
  """Keyword arguments to `eval_loop.evaluate`."""
  config = config_dict.ConfigDict()
  config.early_stop_metric_to_minimize = "eval_mean_squared_error"
  return config


def _get_fine_tune_config(training_config):
  """Keyword arguments to `train_loop.fine_tune`."""
  config = config_dict.ConfigDict()
  config.fine_tune_steps = 4000

  # By default reuse the same intervals than during training.
  config.log_interval = training_config.get_ref("log_interval")
  config.checkpoint_interval = training_config.get_ref("checkpoint_interval")

  # By default tell the model during fine tuning that the global step
  # is the last training step. Note models sometimes use this global_step for
  # learning rate annealing, so in practice this will cause the fine tuning to
  # happen with the final training learning rate.
  config.global_step_for_model = training_config.get_ref("training_steps")

  return config


def _get_checkpointer_config():
  """Keyword arguments to `checkpointing.Checkpointer`."""
  config = config_dict.ConfigDict()
  config.directory = "/tmp/training_example/"
  config.max_to_keep = 2
  return config
