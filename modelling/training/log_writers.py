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
"""Utilities."""

import abc
from typing import Any, Dict, Text

from absl import logging
import tree


class BaseWriter(metaclass=abc.ABCMeta):
  """Writer interface for experiment data."""

  def write_scalars(self, global_step, scalars):
    """Writes the scalars returned by experiment's step()."""


class ConsoleWriter(BaseWriter):
  """Writes training data to the log."""

  def __init__(self, name):
    self._name = name

  def write_scalars(self, global_step: int, scalars: Dict[Text, Any]):
    logging.info('%s: global_step: %d, %s',
                 self._name, global_step,
                 tree.map_structure(str, scalars))


def multiple_write(writers, global_step, scalars):
  for writer in writers:
    writer.write_scalars(global_step, scalars)

