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
"""Tests for dm_c19_modelling.evaluation.dataset_indexing."""

import os

from absl.testing import absltest
from dm_c19_modelling.evaluation import dataset_indexing
import pandas as pd

_TEST_DATASET = "test_dataset"
_TEST_DATASET_FILE = "test_dataset.csv"


def _get_test_entry(directory):
  return {
      "file_location": os.path.join(directory, _TEST_DATASET_FILE),
      "source_data_info": ["test_source_1", "test_source_2"],
      "creation_timestamp": "2020-06-07_12:43:02",
      "dataset_name": _TEST_DATASET,
      "creation_date": "2020-06-07",
      "extra_info": {}
  }


def _create_dataset(file_location):
  df = pd.DataFrame({"site_id": ["A"], "date": ["2020-05-07"],
                     "new_deceased": [0], "new_confirmed": [0]})
  df.to_csv(file_location, index=False)


class DatasetIndexingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._test_dir = absltest.get_default_test_tmpdir()
    os.makedirs(self._test_dir, exist_ok=True)
    self._key = "12345"
    self._entry = _get_test_entry(self._test_dir)
    self._index_path = os.path.join(
        self._test_dir, f"dataset_index-{_TEST_DATASET}.json")
    _create_dataset(self._entry["file_location"])
    self._remove_index_if_exists()

  def _remove_index_if_exists(self):
    if os.path.exists(self._index_path):
      os.remove(self._index_path)

  def test_write_operation_not_in_read_only(self):
    """Test that opening the index in read-only mode prevents writing."""
    index = dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET)
    with self.assertRaisesWithLiteralMatch(
        IOError,
        "Attempting to write to the index when it is in read-only mode."):
      index.add_entry(self._key, {})

  def test_write_operation_not_in_context(self):
    """Tests that the index can't be used in write mode outside of a context."""
    index = dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET,
                                          read_only=False)
    with self.assertRaisesWithLiteralMatch(
        IOError, ("Index has not been loaded. The index should be used as a "
                  "context when not in read-only mode")):
      index.add_entry(self._key, {})

  def test_create_new_index_and_add_entry(self):
    """Tests that an index can get created and an entry added."""
    with dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET,
                                       read_only=False) as index:
      index.add_entry(self._key, self._entry)
    assert os.path.exists(self._index_path)

  def test_create_new_index_add_entry_with_missing_field(self):
    """Tests that adding an entry with missing fields fails."""
    del self._entry["creation_timestamp"]
    with dataset_indexing.DatasetIndex(
        self._test_dir, _TEST_DATASET, read_only=False) as index:
      with self.assertRaisesRegex(ValueError, "Entry must have fields *"):
        index.add_entry(self._key, self._entry)

  def test_add_duplicate_entry(self):
    """Tests that adding an entry with a duplicated key fails."""
    with dataset_indexing.DatasetIndex(
        self._test_dir, _TEST_DATASET, read_only=False) as index:
      index.add_entry(self._key, self._entry)
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          ("Found entry for given key. Index keys must be unique.")):
        index.add_entry(self._key, self._entry)

  def test_create_new_index_add_invalid_creation_timestamp(self):
    """Tests creation timestamp format validation."""
    self._entry["creation_timestamp"] = "2020-06-07"
    with dataset_indexing.DatasetIndex(
        self._test_dir, _TEST_DATASET, read_only=False) as index:
      with self.assertRaisesWithLiteralMatch(ValueError,
                                             "Cannot parse creation_timestamp"):
        index.add_entry(self._key, self._entry)

  def test_create_new_index_add_non_existent_file(self):
    """Tests filepath validation."""
    bad_file_location = os.path.join(self._test_dir, "bad_file")
    self._entry["file_location"] = bad_file_location
    with dataset_indexing.DatasetIndex(
        self._test_dir, _TEST_DATASET, read_only=False) as index:
      with self.assertRaisesRegex(IOError,
                                  f"Path {bad_file_location} not found *"):
        index.add_entry(self._key, self._entry)

  def test_add_to_existing_index(self):
    """Tests that an entry can be added to an existing index."""
    entry_2 = self._entry.copy()
    entry_2["creation_date"] = "2020-06-08"
    key_2 = "123456"
    with dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET,
                                       read_only=False) as index:
      index.add_entry(self._key, self._entry)
    with dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET,
                                       read_only=False) as index:
      index.add_entry(key_2, entry_2)

    read_index = dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET)
    self.assertIsNotNone(read_index.query_by_creation_date("2020-06-07"))
    self.assertIsNotNone(read_index.query_by_creation_date("2020-06-08"))

  def test_get_latest_creation_date(self):
    """Tests that querying 'latest' creation date returns the correct key."""
    with dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET,
                                       read_only=False) as index:
      index.add_entry(self._key, self._entry)
    read_index = dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET)
    self.assertEqual(read_index.query_by_creation_date("latest"), self._key)

  def test_query_by_creation_date_duplicates(self):
    """Tests that querying a duplicated creation date gets the latest entry."""
    entry_2 = self._entry.copy()
    key_2 = "123456"
    entry_2["creation_timestamp"] = "2020-06-07_16:43:02"
    with dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET,
                                       read_only=False) as index:
      index.add_entry(self._key, self._entry)
      index.add_entry(key_2, entry_2)
    read_index = dataset_indexing.DatasetIndex(self._test_dir, _TEST_DATASET)
    self.assertEqual(read_index.query_by_creation_date("latest"), key_2)


if __name__ == "__main__":
  absltest.main()
