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

VIRTUAL_ENV_TMP_DIR=`mktemp -d`

virtualenv --python=python3 "${VIRTUAL_ENV_TMP_DIR}/dm_c19_modelling"
source "${VIRTUAL_ENV_TMP_DIR}/dm_c19_modelling/bin/activate"

pip install -r requirements-modelling.txt
pip install nose

cd ..

# Run unittests
nosetests dm_c19_modelling -verbosity=0

# Run integration test.
bash ./dm_c19_modelling/modelling/scripts/example.sh
