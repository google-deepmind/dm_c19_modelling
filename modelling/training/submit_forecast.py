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
r"""Executable to submit a forecast for the checkpoint of a pretrained model.

Usage:

```
  submit_forecast.py --checkpoint_path=/tmp/checkpoints/latest_fine_tune  \
      --forecast_name="tmp_model_forecast"
```

"""

from absl import app
from absl import flags
from dm_c19_modelling.modelling.training import checkpointing
from dm_c19_modelling.modelling.training import dataset_factory
from dm_c19_modelling.modelling.training import eval_loop
from dm_c19_modelling.modelling.training import model_factory
from dm_c19_modelling.modelling.training import train_loop


FLAGS = flags.FLAGS


flags.DEFINE_string(
    "checkpoint_path", None, "Path to the checkpoint. E.g.. ")

flags.DEFINE_enum(
    "checkpoint_name",
    train_loop.LATEST_FINE_TUNE,
    [train_loop.LATEST_TRAIN,
     eval_loop.BEST_EVAL,
     train_loop.LATEST_FINE_TUNE],
    "Checkpoint name. By default, the fined tuned checkpoint.")

flags.DEFINE_string(
    "forecast_name", None,
    "Forecast name to use for storing predictions in the forecast index.")


def main(argv):
  del argv

  checkpointer = checkpointing.Checkpointer(
      directory=FLAGS.checkpoint_path, max_to_keep=2)

  state = checkpointer.get_experiment_state(FLAGS.checkpoint_name)
  checkpointer.restore(FLAGS.checkpoint_name)

  # Get dataset.
  dataset = dataset_factory.get_training_dataset(
      **state.build_info["dataset_factory_kwargs"])

  # Get model.
  # Note that code for the model must not have changed since training.
  model = model_factory.get_model(
      state.build_info["dataset_spec"],
      **state.build_info["model_factory_kwargs"])

  eval_loop.submit_final_forecast(
      dataset, model, checkpointer,
      forecast_name=FLAGS.forecast_name,
      directory=state.build_info["dataset_factory_kwargs"]["directory"],
      dataset_name=state.build_info["dataset_factory_kwargs"]["dataset_name"],
      checkpoint_name=FLAGS.checkpoint_name)


if __name__ == "__main__":
  app.run(main)
