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
"""Training loop."""

import time

from absl import logging
from dm_c19_modelling.modelling.training import log_writers


# Checkpoint names.
LATEST_TRAIN = "latest_train"
LATEST_FINE_TUNE = "latest_fine_tune"


def train(dataset, model, build_info, checkpointer, writers, training_steps,
          **train_loop_kwargs):
  """Main training loop."""

  logging.info("Start training")

  # Setup/restore the checkpoint.
  state = checkpointer.get_experiment_state(LATEST_TRAIN)
  if not checkpointer.can_be_restored(LATEST_TRAIN):
    state.global_step = 0
    state.model_state = None
    state.build_info = build_info
  else:
    checkpointer.restore(LATEST_TRAIN)

  batch_generator = model.build_training_generator(dataset)

  _train_loop(model, batch_generator, training_steps, state,
              checkpoint_save_fn=lambda: checkpointer.save(LATEST_TRAIN),
              writers=writers, **train_loop_kwargs)


def fine_tune(dataset, model, checkpointer, writers, fine_tune_steps,
              global_step_for_model, initial_checkpoint, **train_loop_kwargs):
  """Fine tuning training loop."""

  logging.info("Start fine-tuning")

  # Setup/restore the fine tune checkpoint.
  state = checkpointer.get_experiment_state(LATEST_FINE_TUNE)
  if not checkpointer.can_be_restored(LATEST_FINE_TUNE):
    # If there is not one yet, we simply copy the initial one.
    initial_state = checkpointer.get_experiment_state(initial_checkpoint)
    checkpointer.restore(initial_checkpoint)
    assign_state(state, initial_state)
    state.global_step = 0
  else:
    checkpointer.restore(LATEST_FINE_TUNE)

  batch_generator = model.build_training_generator(dataset)

  _train_loop(model, batch_generator, fine_tune_steps, state,
              override_global_step_for_model=global_step_for_model,
              checkpoint_save_fn=lambda: checkpointer.save(LATEST_FINE_TUNE),
              writers=writers, **train_loop_kwargs)


def _train_loop(model, batch_generator, training_steps, state,
                log_interval, writers, checkpoint_interval, checkpoint_save_fn,
                override_global_step_for_model=None):
  """Training loop, updating the model state at each iteration."""

  logging.info("Entering training loop")
  prev_timestamp = None
  while  state.global_step < training_steps:
    batch = next(batch_generator)

    if override_global_step_for_model is not None:
      global_step_for_model = override_global_step_for_model
    else:
      global_step_for_model = state.global_step

    state.model_state, scalar_outputs = model.training_update(
        state.model_state, batch, global_step_for_model)

    # Log scalars still using the pre-update global step, as the losses
    # etc. here would correspond to metrics before the model is updated.
    scalar_outputs["step"] = state.global_step

    # Increase the global step before calling the saving model callbacks, so
    # the state of the saved model has the correct global step, e.g.
    # 1 to indicate that model has been trained for 1 iterations.
    state.global_step += 1

    # Write to the loggers periodically.
    if state.global_step % log_interval == 0:
      # Compute steps per second.
      new_timestamp = time.time()
      if prev_timestamp is None:
        scalar_outputs["steps_per_sec"] = float("nan")
      else:
        scalar_outputs["steps_per_sec"] = log_interval / (
            new_timestamp - prev_timestamp)
      log_writers.multiple_write(writers, state.global_step, scalar_outputs)
      prev_timestamp = new_timestamp

    # Checkpointing periodically (should always happens last in the loop).
    if state.global_step % checkpoint_interval == 0:
      checkpoint_save_fn()

  logging.info("Storing checkpoint at end of loop")
  checkpoint_save_fn()


def assign_state(state_dst, state_src):
  for k, v in state_src.items():
    setattr(state_dst, k, v)

