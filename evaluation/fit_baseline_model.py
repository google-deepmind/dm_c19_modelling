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
"""Script to fit a baseline model on a given dataset."""

from absl import app
from absl import flags

from dm_c19_modelling.evaluation import baseline_models
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import dataset_factory
from dm_c19_modelling.evaluation import forecast_indexing
from dm_c19_modelling.evaluation import forecast_utils

_PROJECT_DIR = flags.DEFINE_string(
    "project_directory", None, "The directory where data and models are saved.")

_DATASET_NAME = flags.DEFINE_string(
    "dataset_name", "covid_open_data_world", "The name of the dataset to use.")

_DATASET_CREATION_DATE = flags.DEFINE_string(
    "creation_date", "latest", "The creation date of the dataset to use for "
    "training")

_TARGET_NAME = flags.DEFINE_enum("target_name", "new_deceased",
                                 [target.value for target in constants.Targets],
                                 "The name of the target to predict.")

_LAST_OBSERVATION_DATE = flags.DEFINE_string(
    "last_observation_date", None, "The date to train up to and evaluate from")

_NUM_FORECAST_DATES = flags.DEFINE_integer(
    "num_forecast_dates", 14, "The number of dates to use for evaluation. "
    "The forecast horizon in days is equal to num_forecast_dates * cadence")

_MODEL_NAME = flags.DEFINE_enum(
    "model_name", "logistic", [model.value for model in constants.Models],
    "The model to fit to the data")

_MODEL_DESCRIPTION = flags.DEFINE_string(
    "model_description", None, "Optional description to associate with the "
    "forecasts output by the model in the forecast index.")

_CADENCE = flags.DEFINE_integer(
    "cadence", 1, "The cadence in days of the predictions. i.e. daily "
    "predictions have a cadence of 1, weekly predictions have a cadence of 7.")

_WEEKLY_CONVERSION_END_DAY = flags.DEFINE_string(
    "weekly_conversion_end_day", None, "Whether to convert predictions to "
    "weekly predictions, and if so, what day the week should end on. e.g. A "
    "value of Sunday would aggregate through normal weeks, while a value of "
    "Saturday would aggregate through epidemiological weeks. This can only be "
    "used with a daily cadence.")

_NUM_CONTEXT_DATES = flags.DEFINE_integer(
    "num_context_dates", None,
    "The number of most recent dates that the baseline will be fitted to. "
    "The context horizon in days is equal to num_context_dates * cadence")

flags.mark_flags_as_required(["project_directory", "last_observation_date"])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  dataset = dataset_factory.get_dataset(
      directory=_PROJECT_DIR.value,
      dataset_name=_DATASET_NAME.value,
      creation_date=_DATASET_CREATION_DATE.value,
      last_observation_date=_LAST_OBSERVATION_DATE.value,
      targets=[constants.Targets(_TARGET_NAME.value)],
      features=[],
      num_forecast_dates=_NUM_FORECAST_DATES.value,
      cadence=_CADENCE.value)

  if _MODEL_NAME.value == constants.Models.LOGISTIC.value:
    model_kwargs = dict(num_context_dates=_NUM_CONTEXT_DATES.value)
    model = baseline_models.Logistic(**model_kwargs)

  elif _MODEL_NAME.value == constants.Models.GOMPERTZ.value:
    model_kwargs = dict(num_context_dates=_NUM_CONTEXT_DATES.value)
    model = baseline_models.Gompertz(**model_kwargs)

  elif _MODEL_NAME.value == constants.Models.LINEAR.value:
    model_kwargs = dict(num_context_dates=_NUM_CONTEXT_DATES.value,
                        polynomial_degree=1)
    model = baseline_models.PolynomialFit(**model_kwargs)

  elif _MODEL_NAME.value == constants.Models.QUADRATIC.value:
    model_kwargs = dict(num_context_dates=_NUM_CONTEXT_DATES.value,
                        polynomial_degree=2)
    model = baseline_models.PolynomialFit(**model_kwargs)

  elif _MODEL_NAME.value == constants.Models.REPEAT_LAST_WEEK.value:
    model_kwargs = {}
    model = baseline_models.RepeatLastWeek(**model_kwargs)

  predictions = model.predict(dataset)
  cadence = _CADENCE.value

  if _WEEKLY_CONVERSION_END_DAY.value:
    if cadence != 1:
      raise ValueError("Only daily cadence predictions can be pooled to "
                       "weekly predictions")
    predictions, evaluation_dates = (
        forecast_utils.pool_daily_forecasts_to_weekly(
            predictions, dataset.evaluation_dates,
            _WEEKLY_CONVERSION_END_DAY.value))
    cadence = 7
  else:
    evaluation_dates = dataset.evaluation_dates

  model_description = {
      "name": _MODEL_NAME.value,
      "model_kwargs": model_kwargs,
      "model_description": _MODEL_DESCRIPTION.value,
  }

  predictions_df = forecast_indexing.build_predictions_df(
      predictions, evaluation_dates, dataset.sites,
      dataset.target_names)

  forecast_indexing.save_predictions_df(
      predictions_df,
      directory=str(_PROJECT_DIR.value),
      last_observation_date=str(_LAST_OBSERVATION_DATE.value),
      forecast_horizon=_NUM_FORECAST_DATES.value,
      model_description=model_description,
      dataset_name=_DATASET_NAME.value,
      dataset_index_key=dataset.dataset_index_key,
      cadence=cadence,
      features_used=[_TARGET_NAME.value],
      extra_info={"first_training_date": dataset.training_dates[0]})


if __name__ == "__main__":
  app.run(main)
