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
"""Checkpointing utilities."""

import os
import pickle

from absl import logging
import tensorflow as tf


class _PyWrapper(tf.train.experimental.PythonState):
  """Wraps a Python object for storage in an object-based checkpoint."""

  def __init__(self, obj):
    """Specify an object to wrap.

    Args:
      obj: The object to save and restore (may be overwritten).
    """
    self._obj = obj

  @property
  def object(self):
    return self._obj

  def serialize(self):
    """Callback to serialize the object."""
    return pickle.dumps(self._obj)

  def deserialize(self, string_value):
    """Callback to deserialize the array."""
    self._obj = pickle.loads(string_value)


class _CheckpointState:
  """tf.Train.Checkpoint wrapper ensuring all fields are checkpointable."""

  def __init__(self):
    super().__setattr__('_checkpoint',
                        tf.train.Checkpoint(python_state=_PyWrapper({})))

  @property
  def checkpoint(self):
    return self._checkpoint

  def __setattr__(self, name, value):
    self._checkpoint.python_state.object[name] = value

  def __getattr__(self, name):
    return self._checkpoint.python_state.object[name]

  def keys(self):
    return self._checkpoint.python_state.object.keys()

  def items(self):
    return self._checkpoint.python_state.object.items()


class Checkpointer:
  """Checkpoints python state using tf.train.Checkpoint."""

  def __init__(self, directory, max_to_keep, restore_path=None):
    self._directory = directory
    self._max_to_keep = max_to_keep
    self._first_restore_path = restore_path
    self._experiment_states = {}
    self._checkpoints = {}
    logging.info('Storing checkpoint at: %s', directory)

  def _internal_restore_path(self, checkpoint_name):
    """Returns a path to the checkpoint used for restore, or None."""

    # If we have a checkpoint we own, return that.
    restore_path = self.restore_path(checkpoint_name)

    # Otherwise, check the read-only restore path.
    if restore_path is None and self._first_restore_path is not None:
      # We use the checkpoint metadata (state) to check whether the
      # checkpoint we want actually exists.
      # First restore path can be a directory or a specific checkpoint.
      chk_state = tf.train.get_checkpoint_state(self._first_restore_path)
      if chk_state is not None:
        # The restore path is a directory, get the latest checkpoint from there.
        restore_path = chk_state.model_checkpoint_path
      else:
        # Try with the the parent directory.
        chk_state = tf.train.get_checkpoint_state(
            os.path.dirname(self._first_restore_path))
        if chk_state is not None and (
            self._first_restore_path in chk_state.all_model_checkpoint_paths):
          restore_path = self._first_restore_path
        else:
          restore_path = None

    return restore_path

  def get_experiment_state(self, checkpoint_name):
    """Returns the experiment state."""

    if checkpoint_name not in self._experiment_states:
      assert checkpoint_name not in self._checkpoints
      state = _CheckpointState()
      self._experiment_states[checkpoint_name] = state
      self._checkpoints[checkpoint_name] = tf.train.CheckpointManager(
          state.checkpoint,
          os.path.join(self._directory, checkpoint_name),
          self._max_to_keep,
          checkpoint_name=checkpoint_name)

    return self._experiment_states[checkpoint_name]

  def can_be_restored(self, checkpoint_name):
    """Returns True if the checkpoint with the given name can be restored."""
    return self._internal_restore_path(checkpoint_name) is not None

  def restore(self, checkpoint_name):
    """Restores checkpoint state."""
    save_path = self._internal_restore_path(checkpoint_name)
    assert save_path is not None

    checkpoint_manager = self._checkpoints[checkpoint_name]
    checkpoint_manager.checkpoint.restore(save_path).assert_consumed()
    logging.info('Restored checkpoint from: %s', save_path)

  def restore_or_save(self, checkpoint_name):
    if self.can_be_restored(checkpoint_name):
      self.restore(checkpoint_name)
    else:
      self.save(checkpoint_name)

  def save(self, checkpoint_name):
    """Saves the state to file."""
    self._checkpoints[checkpoint_name].save()
    self._first_restore_path = None
    logging.info('Saved checkpoint at: %s', self.restore_path(checkpoint_name))

  def restore_path(self, checkpoint_name):
    """Returns the restore path for this checkpoint."""
    # Returns None if we didn't create any checkpoint yet.
    chk_state = tf.train.get_checkpoint_state(
        self._checkpoints[checkpoint_name].directory)

    return None if chk_state is None else chk_state.model_checkpoint_path

