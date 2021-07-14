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
"""Script to run evaluation on forecasts from one on more models."""

from absl import app
from absl import flags
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import evaluation
from dm_c19_modelling.evaluation import forecast_indexing


_PROJECT_DIR = flags.DEFINE_string(
    "project_directory", None, "The directory where datasets and models are "
    "saved.")

_DATASET_NAME = flags.DEFINE_string(
    "dataset_name", "covid_open_data_world", "The name of the dataset to use.")

_FORECAST_IDS = flags.DEFINE_list(
    "forecast_ids", None, "The IDs of the forecasts to evaluate. The forecasts "
    "must be comparable: the models used to generate them must have used the "
    "same training dataset, they must have the same forecast date and "
    "forecast_horizon, provide forecasts for the same sites, and must predict "
    "whatever is specified to be target_name")

_EVAL_DATASET_CREATION_DATE = flags.DEFINE_string(
    "eval_dataset_creation_date", "latest", "The creation date of the dataset "
    "to use for getting the ground truth for the evaluation dates.")

_TARGET_NAME = flags.DEFINE_string(
    "target_name", None, "The name of the target to evaluate.")

_SAVE_METRICS = flags.DEFINE_bool(
    "save_metrics", True, "Whether to save metrics to file.")

_SITES_PERMITTED_TO_DROP = flags.DEFINE_list(
    "sites_permitted_to_drop", [], "A list of sites that may be dropped from "
    "from evaluation if forecasts for that site are not defined in every "
    "forecast being compared")

_NUM_FORECAST_DATES = flags.DEFINE_integer(
    "num_forecast_dates", None, "The number of dates to use for evaluation. "
    "This is optional: if not specified, evaluation will run on the maximum "
    "number of overlapping dates available between the different forecasts.")

flags.mark_flags_as_required(["project_directory", "target_name"])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  del argv  # Unused

  if _FORECAST_IDS.value is None:
    # Forecast IDs must be provided, so instead just list the available ones.
    forecast_index = forecast_indexing.ForecastIndex(_PROJECT_DIR.value,
                                                     _DATASET_NAME.value)
    available_forecast_ids = list(forecast_index._index_dict.keys())  # pylint: disable=protected-access
    print("\nAvailable forecast IDs:")
    print("\n".join(available_forecast_ids))
    return

  evaluation.evaluate(
      directory=_PROJECT_DIR.value,
      dataset_name=_DATASET_NAME.value,
      eval_dataset_creation_date=_EVAL_DATASET_CREATION_DATE.value,
      target_name=constants.Targets(_TARGET_NAME.value),
      forecast_ids=_FORECAST_IDS.value,
      save=_SAVE_METRICS.value,
      sites_permitted_to_drop=_SITES_PERMITTED_TO_DROP.value,
      num_forecast_dates=_NUM_FORECAST_DATES.value)

if __name__ == "__main__":
  app.run(main)
