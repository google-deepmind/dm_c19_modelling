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
"""Evaluation loop."""


import time

from absl import logging

from dm_c19_modelling.evaluation import forecast_indexing
from dm_c19_modelling.modelling.training import log_writers
from dm_c19_modelling.modelling.training import train_loop


# Checkpoint names.
LATEST_TRAIN = train_loop.LATEST_TRAIN
LATEST_EVAL = "latest_eval"
BEST_EVAL = "best_eval"


def evaluate(dataset, next_forecast_targets, model, checkpointer, writers,
             training_steps, early_stop_metric_to_minimize):
  """Main evaluation loop."""

  logging.info("Start evaluation")

  # Get the states.
  latest_train_state = checkpointer.get_experiment_state(LATEST_TRAIN)
  latest_eval_state = checkpointer.get_experiment_state(LATEST_EVAL)
  best_eval_state = checkpointer.get_experiment_state(BEST_EVAL)

  # Setup/restore the eval checkpoints, depending on whether `LATEST_EVAL`
  # can be restored, since the existance of LATEST_EVAL guarantees the
  # existence of BEST_EVAL, as the former is saved last.
  # TODO(alvarosg): Consider having a single eval checkpoint, with two high
  # level fields "best", "last", e.g., usages, would be:
  #   `train_loop.assign_state(eval_state.latest, latest_train_state)`
  #   `train_loop.assign_state(eval_state.best, eval_state.latest)`
  # And we would only need a single:
  #   `checkpointer.restore(EVAL)` and `checkpointer.save(EVAL)`
  if not checkpointer.can_be_restored(LATEST_EVAL):
    latest_eval_state.checkpoint_path = None
    best_eval_state.early_stop_metric_value = None
  else:
    checkpointer.restore(BEST_EVAL)
    checkpointer.restore(LATEST_EVAL)

  # Wait until there is a train checkpoint.
  while True:
    if checkpointer.can_be_restored(LATEST_TRAIN):
      break
    else:
      logging.info("Train checkpoint not available, waiting.")
      time.sleep(10)

  while True:
    checkpoint_path = checkpointer.restore_path(LATEST_TRAIN)

    if checkpoint_path == latest_eval_state.checkpoint_path:
      if latest_eval_state.global_step >= training_steps:
        logging.info("Last checkpoint (iteration %d) evaluated, exiting loop.",
                     latest_eval_state.global_step)
        break
      else:
        logging.info(
            "Checkpoint %s already evaluated, waiting.", checkpoint_path)
        time.sleep(10)
        continue

    # Will evaluate the latest train checkpoint available.
    checkpointer.restore(LATEST_TRAIN)

    predictions, _ = model.evaluate(latest_train_state.model_state, dataset)

    # TODO(alvarosg): Add more eval metrics.
    scalar_metrics = {
        "eval_mean_squared_error": (
            (next_forecast_targets - predictions) ** 2).mean(),
        "step": latest_train_state.global_step,
    }

    # Log the eval metrics.
    log_writers.multiple_write(
        writers, latest_train_state.global_step, scalar_metrics)

    # Store the eval metrics in the latest eval checkpoint.
    train_loop.assign_state(latest_eval_state, latest_train_state)
    latest_eval_state.checkpoint_path = checkpoint_path
    latest_eval_state.early_stop_metric_value = scalar_metrics[
        early_stop_metric_to_minimize]

    # Update the best checkpoint if appropriate.
    if (best_eval_state.early_stop_metric_value is None or
        (latest_eval_state.early_stop_metric_value <
         best_eval_state.early_stop_metric_value)):
      if best_eval_state.early_stop_metric_value is None:
        # Initializing best model:
        logging.info("Initializing best model: %s = %g",
                     early_stop_metric_to_minimize,
                     latest_eval_state.early_stop_metric_value)
      else:
        logging.info("Updating best model: %s %g -> %g",
                     early_stop_metric_to_minimize,
                     best_eval_state.early_stop_metric_value,
                     latest_eval_state.early_stop_metric_value)

      train_loop.assign_state(best_eval_state, latest_eval_state)
      checkpointer.save(BEST_EVAL)
    checkpointer.save(LATEST_EVAL)


def submit_final_forecast(dataset, model, checkpointer, forecast_name,
                          directory, dataset_name, checkpoint_name=BEST_EVAL):
  """Submits forecasts from the best checkpoint to the forecast index."""
  state = checkpointer.get_experiment_state(checkpoint_name)
  checkpoint_path = checkpointer.restore_path(checkpoint_name)
  checkpointer.restore(checkpoint_name)
  final_forecast, _ = model.evaluate(state.model_state, dataset)

  model_description = {
      "name": forecast_name,
      "model_factory_kwargs": state.build_info[
          "model_factory_kwargs"],
      "checkpoint_path": checkpoint_path
  }

  extra_info = {
      "first_training_date": dataset.dates[0]
  }

  predictions_df = forecast_indexing.build_predictions_df(
      final_forecast, dataset.evaluation_dates, dataset.sites,
      dataset.target_names)

  # TODO(alvarosg): In case of cadence=1, submit weekly forecasts too, with
  # `forecast_utils.pool_daily_forecasts_to_weekly`.

  logging.info(
      "Submitting final forecast with name '%s' for dataset with index '%s' "
      "for checkpoint at %s.",
      forecast_name, dataset.dataset_index_key, checkpoint_path)

  logging.info("Model description:")
  logging.info(model_description)

  logging.info("Extra info:")
  logging.info(extra_info)

  if forecast_name is None:
    logging.info("Empty forcast name, skipping submission.")
    return

  forecast_indexing.save_predictions_df(
      predictions_df,
      directory=directory,
      last_observation_date=max(dataset.dates),
      forecast_horizon=len(dataset.evaluation_dates),
      model_description=model_description,
      dataset_name=dataset_name,
      dataset_index_key=dataset.dataset_index_key,
      cadence=dataset.cadence,
      features_used=list(dataset.feature_names),
      extra_info=extra_info)
