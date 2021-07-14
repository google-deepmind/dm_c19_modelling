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
"""Tools for indexing datasets."""

import datetime
import os
from typing import Any, Dict, Optional, Sequence

from absl import logging
from dm_c19_modelling.evaluation import base_indexing
from dm_c19_modelling.evaluation import constants
import pandas as pd

# Internal imports.


class DatasetIndex(base_indexing.BaseIndex):
  """Manages loading, querying, and adding entries to an index of datasets."""

  @property
  def _index_type(self):
    return "dataset"

  @property
  def _additional_fields(self):
    return ("creation_date",)

  def query_by_creation_date(
      self, creation_date: str) -> str:
    """Gets the key in the index corresponding to the given creation_date.

    Args:
      creation_date: The required creation date. May be 'latest' which
      defaults to the most recent creation date available.

    Returns:
      The most recent entry in the index with the required creation date, if
      available. None if no match is found.
    """
    if creation_date == "latest":
      # Get the maximum creation date in the dataset
      creation_date = max(
          [entry["creation_date"] for entry in self._index_dict.values()])

    # Get the entry with the required creation date. If there are duplicates
    # then take the entry that was most recently created.
    matches = {
        key: entry["creation_timestamp"]
        for key, entry in self._index_dict.items()
        if entry["creation_date"] == creation_date
    }
    if matches:
      key, _ = max(matches.items(), key=lambda elt: elt[1])
      return key
    else:
      raise ValueError(
          f"Unable to find a dataset with creation date: {creation_date}."
      )

  def load_file_by_key(self,
                       key: str,
                       validate: bool = True) -> pd.DataFrame:
    """Loads the file contained in the index entry with the given key."""
    entry = self.get_entry(key)
    file_location = entry["file_location"]
    if validate:
      base_indexing.validate_path(file_location)
    logging.info("Loading dataset from %s", file_location)
    return pd.read_csv(open(file_location, "r"), keep_default_na=False,
                       na_values=[""], dtype={constants.SITE_ID: str})

  def _validate_file_in_entry(self,
                              entry: base_indexing.IndexEntryType) -> None:
    # Avoid parsing NaNs from strings
    file_location = entry["file_location"]
    df = pd.read_csv(open(file_location, "r"), keep_default_na=False,
                     na_values=[""], dtype={constants.SITE_ID: str})
    target_names = [target.value for target in constants.Targets]
    target_columns = [col for col in df.columns if col in target_names]
    if not target_columns:
      raise ValueError(
          f"No column found for any of the targets: {target_names}")
    required_columns = (
        constants.DATE,
        constants.SITE_ID,
    ) + tuple(target_columns)

    # Validate that all required fields are present and fully defined.
    for required_column in required_columns:
      if required_column not in df.columns:
        raise ValueError(
            f"{required_column} missing from dataset at {file_location}")
      if df[required_column].isna().any():
        raise ValueError(
            f"NaNs found in {required_column} for dataset at {file_location}"
        )
    for site_id, site_df in df.groupby(constants.SITE_ID):
      # Check that the diff in dates for all but the first element is always
      # 1 day (pandas computes a backwards diff and returns NaN for the first
      # element.
      if not (pd.to_datetime(
          site_df[constants.DATE]).diff().iloc[1:] == pd.Timedelta(
              1, "D")).all():
        raise ValueError(f"Non-daily cadence found in data for {site_id}")


def build_entry(
    file_location: str, dataset_name: str, creation_date: str,
    creation_timestamp: str, source_data_info: Sequence[str],
    extra_info: Dict[str, Any]) -> base_indexing.IndexEntryType:
  """Builds an entry into a dataset index.

  Args:
    file_location: the path to the dataset (may be a URL).
    dataset_name: the name of the dataset.
    creation_date: the date upon which the data was reported.
    creation_timestamp: the datetime at which the dataset was created.
    source_data_info: a list of the sources used to create this dataset.
    extra_info: any extra information that is useful to store alongside the
      rest of the dataset metadata.
  Returns:
    An entry for this dataset that can be added to the dataset index.
  """
  return {
      "file_location": file_location,
      "dataset_name": dataset_name,
      "creation_date": creation_date,
      "creation_timestamp": creation_timestamp,
      "source_data_info": source_data_info,
      "extra_info": extra_info
  }


def save_dataset(df: pd.DataFrame,
                 directory: str,
                 dataset_name: str,
                 source_data_info: Sequence[str],
                 creation_date: Optional[str] = None) -> None:
  """Saves the dataset and updates the dataset indexer with its metadata."""

  # Create a unique key into the index based on the current time.
  index_key = base_indexing.get_unique_key()
  datasets_directory = os.path.join(directory, "datasets")
  current_datetime = datetime.datetime.utcnow()
  if not creation_date:
    creation_date = current_datetime.strftime(constants.DATE_FORMAT)
  output_filepath = os.path.join(
      datasets_directory, f"{dataset_name}_{creation_date}_{index_key}.csv")

  if not os.path.exists(datasets_directory):
    os.makedirs(datasets_directory)

  assert not os.path.exists(output_filepath), (
      f"A dataset already exists at {output_filepath}.")

  df.to_csv(open(output_filepath, "w"), index=False)
  logging.info("Saved dataset to %s", output_filepath)
  extra_dataset_info = {
      "first_data_date": df[constants.DATE].min(),
      "late_data_date": df[constants.DATE].max(),
      "number_of_sites": len(df[constants.SITE_ID].unique())
  }

  entry = build_entry(
      file_location=output_filepath,
      dataset_name=dataset_name,
      creation_date=creation_date,
      creation_timestamp=current_datetime.strftime(constants.DATETIME_FORMAT),
      source_data_info=source_data_info,
      extra_info=extra_dataset_info
  )

  base_indexing.open_index_and_add_entry(
      directory, dataset_name, index_class=DatasetIndex, key=index_key,
      entry=entry)
