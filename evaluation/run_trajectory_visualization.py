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
"""Script to visualize trajectories of data and forecasts."""

import datetime
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import dataset_factory
from dm_c19_modelling.evaluation import evaluation
from dm_c19_modelling.evaluation import plot_utils

_PROJECT_DIR = flags.DEFINE_string(
    "project_directory", None, "The directory where datasets and models are "
    "saved.")

_DATASET_NAME = flags.DEFINE_string("dataset_name", "covid_open_data_world",
                                    "The name of the dataset to use.")

_FORECAST_IDS = flags.DEFINE_list(
    "forecast_ids", None, "The IDs of the forecasts to evaluate. The forecasts "
    "must be comparable: the models used to generate them must have used the "
    "same training dataset, they must have the same forecast date and "
    "forecast_horizon, provide forecasts for the same sites, and must predict "
    "whatever is specified to be target_name.")

_EVAL_DATASET_CREATION_DATE = flags.DEFINE_string(
    "eval_dataset_creation_date", "latest", "The creation date of the dataset "
    "to use for getting the ground truth for the evaluation dates.")

_TARGET_NAME = flags.DEFINE_string("target_name", None,
                                   "The name of the target to evaluate.")

_NUM_FORECAST_DATES = flags.DEFINE_integer(
    "num_forecast_dates", None, "The number of dates to use for evaluation. "
    "This is optional: if not specified, evaluation will run on the maximum "
    "number of overlapping dates available between the different forecasts.")

_NUM_SITES = flags.DEFINE_integer(
    "num_sites", 16, "The number of sites to use for evaluation. "
    "This is optional: if not specified, will plot 16 sites.")

_OVERWRITE = flags.DEFINE_boolean(
    "overwrite", False, "Force overwriting of existing images. "
    "This is optional: if not specified, will default to False..")

flags.mark_flags_as_required(
    ["project_directory", "forecast_ids", "target_name"])


def get_forecast_arrays(directory: str, dataset_name: str,
                        target_name: constants.Targets,
                        forecast_ids: Sequence[str]):
  """Get the forecasts from disk."""
  (all_forecast_entries,
   all_forecasts) = evaluation._load_all_entries_and_forecasts(  # pylint: disable=protected-access

       directory, dataset_name, forecast_ids, target_name.value)
  last_observation_date, forecast_cadence = (
      evaluation._get_last_observation_date_and_validate_comparable(  # pylint: disable=protected-access

          all_forecast_entries))
  all_forecast_arrays = evaluation._convert_forecasts_to_arrays(all_forecasts)  # pylint: disable=protected-access

  return (last_observation_date, forecast_cadence, all_forecast_arrays,
          all_forecast_entries)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  del argv  # Unused

  directory = _PROJECT_DIR.value
  dataset_name = _DATASET_NAME.value
  eval_dataset_creation_date = _EVAL_DATASET_CREATION_DATE.value
  last_observation_date = None
  target_name = constants.Targets(_TARGET_NAME.value)
  forecast_ids = _FORECAST_IDS.value
  num_forecast_dates = _NUM_FORECAST_DATES.value
  num_sites = _NUM_SITES.value

  (last_observation_date, forecast_cadence, all_forecast_arrays,
   all_forecast_entries) = get_forecast_arrays(
       directory=directory,
       dataset_name=dataset_name,
       target_name=target_name,
       forecast_ids=forecast_ids,
   )

  dates_to_eval, _, sites_to_drop, _ = (
      evaluation._get_forecast_spec_and_comparable_predictions(  # pylint: disable=protected-access
          all_forecast_arrays, num_forecast_dates))

  if sites_to_drop:
    logging.warn("sites to drop includes: %s", sites_to_drop.join(", "))

  forecast_horizon = (
      datetime.datetime.strptime(max(dates_to_eval), constants.DATE_FORMAT) -
      datetime.datetime.strptime(last_observation_date,
                                 constants.DATE_FORMAT)).days

  eval_dataset = dataset_factory.get_dataset(
      directory=directory,
      dataset_name=dataset_name,
      creation_date=eval_dataset_creation_date,
      last_observation_date=last_observation_date,
      targets=[target_name],
      features=[],
      num_forecast_dates=len(dates_to_eval),
      cadence=forecast_cadence)

  # Get the actual evaluation creation date in case using 'latest'
  eval_dataset_creation_date = evaluation.get_recorded_creation_date(
      directory, dataset_name, eval_dataset.dataset_index_key)

  plot_utils.plot_trajectories_and_save(
      directory=directory,
      forecast_ids=forecast_ids,
      eval_dataset_creation_date=eval_dataset_creation_date,
      forecast_horizon=forecast_horizon,
      save=True,
      target_name=target_name,
      all_forecast_entries=all_forecast_entries,
      all_forecast_arrays=all_forecast_arrays,
      num_sites=num_sites,
      eval_dataset=eval_dataset,
      overwrite=_OVERWRITE.value)


if __name__ == "__main__":
  app.run(main)
