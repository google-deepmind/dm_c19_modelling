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
"""Tests for checkpointing.py."""

import os
import tempfile
from absl.testing import absltest

from dm_c19_modelling.modelling.training import checkpointing
import numpy as np


class CheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._checkpoint_dir = tempfile.TemporaryDirectory().name

  def test_python_state(self):
    chk = checkpointing.Checkpointer(self._checkpoint_dir, max_to_keep=1)
    state = chk.get_experiment_state('checkpoint1')
    state.int = 10
    state.float = 12.34
    state.string = 'test'
    state.numpy = np.array([1, 2, 3, 4])

    # Save the checkpooint.
    self.assertFalse(chk.can_be_restored('checkpoint1'))
    chk.restore_or_save('checkpoint1')
    self.assertTrue(chk.can_be_restored('checkpoint1'))

    # Restore the checkpoint.
    chk2 = checkpointing.Checkpointer(self._checkpoint_dir, max_to_keep=1)
    state2 = chk2.get_experiment_state('checkpoint1')

    self.assertTrue(chk2.can_be_restored('checkpoint1'))
    chk2.restore_or_save('checkpoint1')

    self.assertEqual(state.int, state2.int)
    self.assertEqual(state.float, state2.float)
    self.assertEqual(state.string, state2.string)
    np.testing.assert_array_equal(state.numpy, state2.numpy)

  def test_restore_path(self):
    chk1 = checkpointing.Checkpointer(
        self._checkpoint_dir,
        max_to_keep=1,
        restore_path=os.path.join(self._checkpoint_dir, 'bad_path'))

    state1 = chk1.get_experiment_state('state1')
    state1.counter = np.random.randint(100)
    self.assertFalse(chk1.can_be_restored('state1'))

    chk1.save('state1')
    self.assertTrue(chk1.can_be_restored('state1'))

    chk2 = checkpointing.Checkpointer(
        self._checkpoint_dir,
        max_to_keep=1,
        restore_path=os.path.join(self._checkpoint_dir, 'state1'))

    state2 = chk2.get_experiment_state('state2')
    self.assertTrue(chk2.can_be_restored('state2'))

    # First restore will override the state with the values from the checkpoint.
    state2.counter = state1.counter + 1
    chk2.restore('state2')
    self.assertEqual(state1.counter, state2.counter)

    # After we save and restore, the original values are lost.
    state2.counter = state1.counter + 1
    chk2.save('state2')

    chk3 = checkpointing.Checkpointer(
        self._checkpoint_dir,
        max_to_keep=1,
        restore_path=chk1.restore_path('state1'))

    # The restore path will be ignored because we have a checkpoint for state2
    # in our main checkpoint directory.
    state3 = chk3.get_experiment_state('state2')
    chk3.restore('state2')
    self.assertEqual(state3.counter, state1.counter + 1)

  def test_restore_path_update(self):
    chk1 = checkpointing.Checkpointer(self._checkpoint_dir, max_to_keep=1)
    state1 = chk1.get_experiment_state('latest')
    state1.counter = np.random.randint(100)
    self.assertIsNone(chk1.restore_path('latest'))

    chk2 = checkpointing.Checkpointer(self._checkpoint_dir, max_to_keep=1)
    state2 = chk2.get_experiment_state('latest')
    self.assertIsNone(chk2.restore_path('latest'))
    self.assertFalse(chk2.can_be_restored('latest'))

    state1.counter += 1
    chk1.save('latest')
    restore_path = chk2.restore_path('latest')
    self.assertIsNotNone(restore_path)
    self.assertTrue(chk2.can_be_restored('latest'))

    state1.counter += 1
    chk1.save('latest')
    new_restore_path = chk2.restore_path('latest')
    self.assertNotEqual(restore_path, new_restore_path)

    chk2.restore('latest')
    self.assertEqual(state2.counter, state1.counter)


if __name__ == '__main__':
  absltest.main()
