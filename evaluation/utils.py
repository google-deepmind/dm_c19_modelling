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
"""Utilities for dataset and evaluation modules."""

import os  # pylint: disable=unused-import
import time

# Internal imports.


class FileLock(object):
  """Creates a file lock."""

  def __init__(self,
               lock_path: str,
               timeout: int,
               retry_interval: int = 1) -> None:
    self._lock_path = lock_path
    self._timeout = timeout
    self._retry_interval = retry_interval

  def acquire(self) -> None:
    time_elapsed = 0
    while os.path.exists(self._lock_path):
      if time_elapsed > self._timeout:
        raise IOError(f"Unable to acquire lock {self._lock_path}.")
      time_elapsed += self._retry_interval
      time.sleep(self._retry_interval)
    open(self._lock_path, "w").close()

  def release(self) -> None:
    os.remove(self._lock_path)
