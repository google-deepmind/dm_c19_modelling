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
"""General constants available in dataset and evaluation modules."""

import enum

DATE = "date"
SITE_ID = "site_id"

DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S"

DECEASED_NEW = "new_deceased"
CONFIRMED_NEW = "new_confirmed"
HOSPITALISED_NEW = "new_hospitalised"

# Column names used in forecasts
PREDICTION = "prediction"
TARGET_NAME = "target_name"


# At least one target needs to be defined in a dataset.
class Targets(enum.Enum):
  DECEASED_NEW = DECEASED_NEW
  CONFIRMED_NEW = CONFIRMED_NEW
  HOSPITALISED_NEW = HOSPITALISED_NEW


# Models that are available
class Models(enum.Enum):
  LOGISTIC = "logistic"  # Fit cumulative targets as a Logistic function of time
  GOMPERTZ = "gompertz"  # Fit cumulative targets as a Gompertz function of time
  LINEAR = "linear"  # Fit targets as a linear function of time
  QUADRATIC = "quadratic"  # Fit targets as a quadratic function of time
  REPEAT_LAST_WEEK = "repeat_last_week"  # Repeat the last week's data
