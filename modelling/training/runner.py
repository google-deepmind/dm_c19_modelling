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
"""Executable to train and evaluate models online.

Usage:

```
  runner.py --config=base_config.py:<project_directory> --mode="train"
  runner.py --config=base_config.py:<project_directory> --mode="eval"
```

"""

from absl import app
from absl import flags

from dm_c19_modelling.modelling import definitions
from dm_c19_modelling.modelling.training import checkpointing
from dm_c19_modelling.modelling.training import dataset_factory
from dm_c19_modelling.modelling.training import eval_loop
from dm_c19_modelling.modelling.training import log_writers
from dm_c19_modelling.modelling.training import model_factory
from dm_c19_modelling.modelling.training import train_loop

from ml_collections import config_flags

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
    "config", help_string="Experiment configuration file.")
flags.DEFINE_enum(
    "mode", "train", ["train", "eval"],
    "Execution mode: `train` will run training, `eval` will run evaluation.")

flags.DEFINE_boolean(
    "overfit", False,
    "If True, no data is left out for validation. Useful for debugging models.")

flags.DEFINE_string(
    "forecast_name", None,
    "Forecast name to use for storing final predictions in the forecast index.")


def create_writers(mode):

  writers = []
  writers.append(log_writers.ConsoleWriter(mode))

  return writers


def main(argv):
  del argv
  config = FLAGS.config

  checkpointer = checkpointing.Checkpointer(**config.checkpointer)
  writers = create_writers(FLAGS.mode)
  dataset = dataset_factory.get_training_dataset(**config.dataset)
  dataset_spec = definitions.TrainingDatasetSpec.from_dataset(dataset)
  model = model_factory.get_model(dataset_spec, **config.model)

  build_info = dict(
      dataset_spec=dataset_spec,
      model_factory_kwargs=config.model.to_dict(),
      dataset_factory_kwargs=config.dataset.to_dict(),
  )

  # Get another version of the dataset with some trailing dates left out
  # for validation and early stopping.
  dataset_without_validation_dates, valid_forecast_targets = (
      dataset_factory.remove_validation_dates(dataset))

  if FLAGS.mode == "train":
    train_loop.train(
        dataset if FLAGS.overfit else dataset_without_validation_dates,
        model, build_info, checkpointer, writers, **config.training)
  elif FLAGS.mode.startswith("eval"):
    eval_loop.evaluate(
        dataset_without_validation_dates, valid_forecast_targets, model,
        checkpointer, writers, training_steps=config.training.training_steps,
        **config.eval)

    if config.fine_tune.fine_tune_steps is not None:
      # Fine tune the best eval checkpoint on the whole dataset, including
      # validation dates.
      fine_tune_writers = create_writers("fine_tune")
      train_loop.fine_tune(
          dataset, model, checkpointer, fine_tune_writers,
          initial_checkpoint=eval_loop.BEST_EVAL, **config.fine_tune)
      checkpoint_name_to_submit = train_loop.LATEST_FINE_TUNE
    else:
      checkpoint_name_to_submit = eval_loop.BEST_EVAL

    # Make a final forecast using the full dataset (including validation dates)
    # as inputs.
    if not FLAGS.overfit:
      eval_loop.submit_final_forecast(
          dataset, model, checkpointer, forecast_name=FLAGS.forecast_name,
          directory=config.dataset.directory,
          dataset_name=config.dataset.dataset_name,
          checkpoint_name=checkpoint_name_to_submit)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
