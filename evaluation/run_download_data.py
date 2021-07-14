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
"""Downloads and saves data from the COVID-19 Open Data repository."""


from absl import app
from absl import flags
from dm_c19_modelling.evaluation import dataset_indexing
from dm_c19_modelling.evaluation import download_data

_PROJECT_DIR = flags.DEFINE_string(
    "project_directory", default=None, help="The output directory where the "
    "dataset index and dataset should be saved.")

_DATASET_NAME = flags.DEFINE_enum(
    "dataset_name", default=None, enum_values=download_data.VALID_DATASETS,
    help="The name of the dataset to download.")

flags.mark_flags_as_required(["project_directory", "dataset_name"])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  del argv  # Unused
  df, source_data_info = download_data.fetch_data(_DATASET_NAME.value)
  dataset_indexing.save_dataset(
      df,
      directory=_PROJECT_DIR.value,
      dataset_name=_DATASET_NAME.value,
      source_data_info=source_data_info)


if __name__ == "__main__":
  app.run(main)
