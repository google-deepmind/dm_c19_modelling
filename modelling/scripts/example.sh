#!/bin/bash
# Copyright 2021 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error.
set -e

# Display commands being run.
set -x

PROJECT_TMP_DIR=`mktemp -d`

# Fetch some data.
python3 -m dm_c19_modelling.evaluation.run_download_data \
    --project_directory=$PROJECT_TMP_DIR \
    --dataset_name="covid_open_data_us_states"


CHECKPOINT_TMP_DIR=`mktemp -d`

# Run training in the background with & .
python3 -m dm_c19_modelling.modelling.training.runner \
    --config=./dm_c19_modelling/modelling/training/base_config.py:$PROJECT_TMP_DIR \
    --config.dataset.dataset_name="covid_open_data_us_states" \
    --config.dataset.allow_dropped_sites=True \
    --config.checkpointer.directory=$CHECKPOINT_TMP_DIR \
    --mode="train" &

# Run eval in parallel with training. Eval will detect when the main training
# has finished, perform final fine-tuning and submit a forecast to the
# forecast index.
python3 -m dm_c19_modelling.modelling.training.runner \
    --config=./dm_c19_modelling/modelling/training/base_config.py:$PROJECT_TMP_DIR \
    --config.dataset.dataset_name="covid_open_data_us_states" \
    --config.dataset.allow_dropped_sites=True \
    --config.checkpointer.directory=$CHECKPOINT_TMP_DIR \
    --mode="eval" \
    --forecast_name="test_forecast"

echo "All data and output forecasts are here: ${PROJECT_TMP_DIR}"
