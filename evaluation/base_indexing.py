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
"""Base class for indexers that store dataset and forecast metadata."""

import abc
import datetime
import json
import os
import shutil  # pylint: disable=unused-import
from typing import Any, Callable, Dict, Optional, Sequence, Set, Union

from absl import logging
from dm_c19_modelling.evaluation import constants
from dm_c19_modelling.evaluation import utils

# Internal imports.

_JSON_DUMP_KWARGS = dict(sort_keys=True, indent=4, separators=(",", ": "))

IndexEntryType = Dict[str, Union[str, int, Dict[str, Any], Sequence[str]]]


def get_unique_key() -> str:
  """Gets a unique key based on the current time to use in indexing."""
  return str(int(datetime.datetime.utcnow().timestamp() * int(1e6)))


def validate_path(path: str) -> None:
  """Validates that a path exists."""
  if not os.path.exists(path):
    existing = "\n".join(sorted(os.listdir(os.path.dirname(path))))
    raise IOError(f"Path {path} not found. Found:\n{existing}")


def _create_index_path_from_name(directory: str,
                                 dataset_name: str,
                                 index_type: str) -> str:
  """Gets the index's path for a given directory and dataset name."""
  filename = f"{index_type}_index-{dataset_name}.json"
  return os.path.join(directory, filename)


def _load_index(path: str) -> Dict[str, IndexEntryType]:
  """Loads index given its path ."""
  logging.info("Index file found, loading")
  with open(path, "r") as infile:
    index_dict = json.load(infile)
  return index_dict


def _save_index(path: str, index: Dict[str, IndexEntryType]) -> None:
  """Saves an index as JSON to file, creating a backup of the previous index."""

  if os.path.exists(path):
    backup_path = path + ".backup"
    if os.path.exists(backup_path):
      os.remove(backup_path)
    shutil.copy2(path, backup_path)

  with open(path, "w") as outfile:
    json.dump(index, outfile, **_JSON_DUMP_KWARGS)
    logging.info("Saved index dict at %s", path)


def validate_entry(entry: IndexEntryType,
                   required_fields: Set[str],
                   validate_file_location: bool = True) -> None:
  """Checks that an entry's required fields exist and have valid values."""
  if ("creation_timestamp" not in required_fields or
      "file_location" not in required_fields):
    raise ValueError(
        "Required fields must include creation_timestamp and file_location")

  if set(entry) != required_fields:
    raise ValueError(
        f"Entry must have fields: {', '.join(sorted(required_fields))}. "
        f"Has fields: {', '.join(sorted(entry))}")
  try:
    datetime.datetime.strptime(entry["creation_timestamp"],
                               constants.DATETIME_FORMAT)
  except ValueError:
    raise ValueError("Cannot parse creation_timestamp")
  if validate_file_location:
    validate_path(entry["file_location"])


class BaseIndex(metaclass=abc.ABCMeta):
  """Manages loading, querying, and adding entries to an index.

  Minimum required entry fields:

  "file_location": `str` with absolute file path/url of the file
  "source_data_info": `dict` with info about the file's source data
  "creation_timestamp": `str` that can be parsed with "%Y-%m-%d_%H:%M:%S"
  "extra_info": `dict` with any additional relevant information

  Indexes may have additional required fields, specified in '_additional_fields'
  """

  def __init__(self, directory: str, dataset_name: str, read_only: bool = True):
    self._index_path = _create_index_path_from_name(
        directory, dataset_name, self._index_type)
    if not read_only:
      self._lock = utils.FileLock(f"{self._index_path}.lock", timeout=100)
    else:
      self._lock = None
    self._index_dict = None
    if read_only:
      self._index_dict = self._load_index()

  def __repr__(self):
    return json.dumps(self._index_dict, **self._json_dump_kwargs)

  def __enter__(self):
    if self._lock:
      self._lock.acquire()
      self._index_dict = self._load_index()
    return self

  def __exit__(self, *args):
    if self._lock:
      self._store()
      self._lock.release()

  @property
  def _fields(self):
    """The required fields in every index entry."""
    return set(self._additional_fields) | {
        "dataset_name", "file_location", "source_data_info",
        "creation_timestamp", "extra_info"
    }

  @abc.abstractproperty
  def _index_type(self):
    """The name of the type of index."""

  @abc.abstractproperty
  def _additional_fields(self):
    """The names of additional fields specific to the index."""

  @abc.abstractmethod
  def load_file_by_key(self, key: str) -> Any:
    """Loads the file contained in the index entry with the given key."""

  @abc.abstractmethod
  def _validate_file_in_entry(self, entry: IndexEntryType) -> None:
    """Checks that the file in the entry loads / is formatted correctly."""

  def _load_index(self) -> Dict[str, IndexEntryType]:
    """Loads an index from file."""
    try:
      validate_path(self._index_path)
    except IOError:
      logging.info("No existing index found, creating a new one.")
      return {}
    return _load_index(self._index_path)

  def _store(self) -> None:
    """Stores an index at index_path."""
    if not self._lock:
      raise IOError(
          "Attempting to write to the index when it is in read-only mode.")
    if self._index_dict is None:
      raise IOError(
          "Index has not been loaded. The index should be used as a context "
          "when not in read-only mode")
    _save_index(self._index_path, self._index_dict)

  def add_entry(self,
                key: Optional[str],
                entry: IndexEntryType,
                validate_file_location: bool = True,
                validate_file_in_entry: bool = True) -> None:
    """Adds an entry to an index, validating its fields."""
    key = key or get_unique_key()
    if not self._lock:
      raise IOError(
          "Attempting to write to the index when it is in read-only mode.")
    if self._index_dict is None:
      raise IOError(
          "Index has not been loaded. The index should be used as a context "
          "when not in read-only mode")
    if key in self._index_dict:
      raise ValueError("Found entry for given key. Index keys must be unique.")

    validate_entry(entry, self._fields, validate_file_location)
    if validate_file_in_entry:
      self._validate_file_in_entry(entry)
    self._index_dict[key] = entry

  def remove_entry(self, key: str) -> None:
    """Removes an entry from the index given its key."""
    if not self._lock:
      raise IOError(
          "Attempting to modify the index when it is in read-only mode.")
    if self._index_dict is None:
      raise IOError(
          "Index has not been loaded. The index should be used as a context "
          "when not in read-only mode")
    if key not in self._index_dict:
      raise ValueError(f"Could not find entry with key {key} to delete.")
    del self._index_dict[key]

  def get_entry(self, key: str) -> IndexEntryType:
    """Gets an entry from the index given a key."""
    if key not in self._index_dict:
      raise ValueError(f"Invalid key {key} not found in index.")
    return self._index_dict[key]


def open_index_and_add_entry(directory: str,
                             dataset_name: str,
                             index_class: Callable[..., BaseIndex],
                             key: Optional[str],
                             entry: IndexEntryType,
                             validate_file_location: bool = True,
                             validate_file_in_entry: bool = False) -> None:
  """Opens an index and adds an entry into it.

  Args:
    directory: the directory where the index is or will be stored.
    dataset_name: the name given to the dataset to which the index relates.
    index_class: the class of index to add to.
    key: optional unique identifier.
    entry: the entry into the index, returned from build_entry.
    validate_file_location: whether to validate that the file location in the
      entry exists on disk.
    validate_file_in_entry: whether to validate that the file referenced in the
      entry adheres to specific formatting requirements.
  """
  # Create a unique key into the index based on the current time.
  key = key or get_unique_key()

  with index_class(directory, dataset_name, read_only=False) as index:
    index.add_entry(key, entry, validate_file_location=validate_file_location,
                    validate_file_in_entry=validate_file_in_entry)
