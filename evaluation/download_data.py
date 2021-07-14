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
"""Library to download and save data from the COVID-19 Open Data repository."""

import functools
import itertools
from typing import Dict, List, Tuple

from absl import logging
from dm_c19_modelling.evaluation import constants
import pandas as pd

_WORLD_DATASET_NAME = "covid_open_data_world"
_US_STATES_DATASET_NAME = "covid_open_data_us_states"
VALID_DATASETS = [_WORLD_DATASET_NAME, _US_STATES_DATASET_NAME]

_BASE_PATH = "https://storage.googleapis.com/covid19-open-data/v2/"
_TABLES_OF_INTEREST = ("index", "epidemiology", "demographics", "mobility")

_TARGETS = [constants.Targets.DECEASED_NEW, constants.Targets.CONFIRMED_NEW]

# Start date for which target data will be kept.
_FIRST_DATE = {
    _WORLD_DATASET_NAME: "2020-01-05",
    _US_STATES_DATASET_NAME: "2020-01-22"
}

# Up to this date any missing target data is assumed to be zero.
# This is a grace period to be able to align in time sites that start
# reporting later than the first date, for which our investigations indicate
# the lack of reporting is related to the lack of cases.
_END_GRACE_PERIOD_DATE = {
    _WORLD_DATASET_NAME: None,
    _US_STATES_DATASET_NAME: "2020-03-15"
}


_DATASET_FILTERS = {
    _WORLD_DATASET_NAME: lambda df: df.query("aggregation_level == 0"),
    _US_STATES_DATASET_NAME: (
        lambda df: df.query("aggregation_level == 1 and country_code == 'US'")
    )
}

_SITES_FORCED_DROPPED = {
    _WORLD_DATASET_NAME: [],
    # Drop US territories. This leaves 50 states + the District of Columbia.
    _US_STATES_DATASET_NAME: ["AS", "PR", "VI", "GU", "MP"],
}


def _load_table_data(table_name: str) -> pd.DataFrame:
  """Loads a dataframe from the COVID-19 Open Data repository."""
  # Avoid parsing NaNs from strings
  df = pd.read_csv(
      _BASE_PATH + table_name + ".csv", keep_default_na=False, na_values=[""])
  return df


def _get_column_rename_map(dataset_name: str) -> Dict[str, str]:
  rename_map = {"date": constants.DATE}
  if dataset_name == _WORLD_DATASET_NAME:
    rename_map["country_code"] = constants.SITE_ID
  elif dataset_name == _US_STATES_DATASET_NAME:
    rename_map["subregion1_code"] = constants.SITE_ID
  else:
    raise ValueError(f"Unknown dataset name {dataset_name}")
  return rename_map


def _join_two(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
  """Merges dataframes on the "key" field, and the "date" field if present."""
  join_fields = ["key"]
  if "date" in df1 and "date" in df2:
    join_fields.append("date")
  return pd.merge(df1, df2, on=join_fields, how="outer")


def _add_missing_grace_dates_rows(
    input_df, first_required_date, end_grace_period_date):
  """Adds potentially missing rows for the grace period."""
  all_keys = input_df["key"].unique()
  grace_dates = list(
      pd.date_range(first_required_date, end_grace_period_date).strftime(
          constants.DATE_FORMAT))
  primary_key_columns = ["key", constants.DATE]
  grace_dates_df = pd.DataFrame(
      itertools.product(all_keys, grace_dates), columns=primary_key_columns)
  merged = input_df.merge(grace_dates_df, how="outer", on=primary_key_columns)
  return merged.sort_values(constants.DATE)


def _drop_sites_with_insufficient_data(df: pd.DataFrame,
                                       dataset_name: str,
                                       target_name: str) -> pd.DataFrame:
  """Drops sites from the dataset due to missing or insufficient data.

  Sites are dropped if they meet any of the following:
  1. The target_name feature is always missing for that site.
  2. The target_name feature isn't defined from _FIRST_DATE (
  or _END_GRACE_PERIOD_DATE if applicable) for that site.
  3. The target_name feature is missing at any point in the range where it is
     defined.

  Args:
    df: dataframe, the merged data table.
    dataset_name: the name of the dataset.
    target_name: str, the name of a feature which may be used as a prediction
      target.
  Returns:
    df: dataframe, with sites with insufficient data dropped, and dates
      truncated to the valid range.
  """
  first_date = _FIRST_DATE[dataset_name]
  sites_forced_dropped = _SITES_FORCED_DROPPED[dataset_name]

  sites_to_drop_and_reason = {}
  max_dates_with_target = []
  for site_id, site_df in df.groupby(constants.SITE_ID):
    if site_id in sites_forced_dropped:
      sites_to_drop_and_reason[site_id] = "forced removed"
      continue

    if site_df[target_name].isna().all():
      sites_to_drop_and_reason[site_id] = "all nans"
      continue
    min_date_with_target = min(
        site_df[~site_df[target_name].isna()][constants.DATE])
    max_date_with_target = max(
        site_df[~site_df[target_name].isna()][constants.DATE])
    if min_date_with_target > first_date:
      sites_to_drop_and_reason[site_id] = "missing date rows before first date"
      continue
    site_df_valid = site_df.query(
        f"date >= '{first_date}' and date <= '{max_date_with_target}'"
    )
    if site_df_valid[target_name].isna().any():
      sites_to_drop_and_reason[site_id] = "nan target values"
      continue
    # Verify that there is exactly one row for each day in the available range
    if not (pd.to_datetime(
        site_df_valid[constants.DATE]).diff().iloc[1:] == pd.Timedelta(
            1, "D")).all():
      sites_to_drop_and_reason[site_id] = "non-daily cadence"
      continue
    max_dates_with_target.append(max_date_with_target)

  if sites_to_drop_and_reason:
    logging.info("Removing the following sites due to insufficient data for "
                 "target %s: %s", target_name, sites_to_drop_and_reason)
  sites_to_drop = list(sites_to_drop_and_reason.keys())
  df = df.query(f"{constants.SITE_ID} not in {sites_to_drop}")

  if not max_dates_with_target:
    raise ValueError("All sites have been dropped.")

  max_available_date = min(max_dates_with_target)
  df = df.query(
      f"date >= '{first_date}' and date <= '{max_available_date}'")
  if df[target_name].isna().any():
    raise ValueError(f"NaNs found for {target_name}.")

  return df


def _maybe_set_zero_epidemiology_targets_in_grace_period(df, dataset_name):
  """Sets epidemiology targets to 0 for grace period."""
  first_date = _FIRST_DATE[dataset_name]
  end_grace_period_date = _END_GRACE_PERIOD_DATE[dataset_name]

  if (end_grace_period_date is not None and
      first_date <= end_grace_period_date):

    # Add missing row combinations of dates and sites for the grace period.
    df = _add_missing_grace_dates_rows(
        df, first_date, end_grace_period_date)

    # Replace any nan targets by zeros in between the first target date, and
    # the end of the grace period.
    for target in _TARGETS:
      mask = (df[target.value].isna() &
              (df[constants.DATE] >= first_date) &
              (df[constants.DATE] <= end_grace_period_date))
      df.loc[mask, target.value] = 0
  return df


def fetch_data(dataset_name: str) -> Tuple[pd.DataFrame, List[str]]:
  """Download and process data from the COVID-19 Open Data repository.

  Args:
    dataset_name: The name of the dataset to download and process. Valid options
      are 'covid_open_data_world' and 'covid_open_data_us_states'
  Returns:
    A tuple of (dataframe, dataset_name, dataset_sources.) The dataframe has
    target, feature, site and date columns. No target entry may be missing for
    any target or feature. The dataset_sources indicate where the data was
    downloaded from.
  """
  # Filter the table according to the dataset requirements.
  if dataset_name not in VALID_DATASETS:
    raise ValueError(f"Unrecognised dataset name {dataset_name} specified. "
                     f"Valid dataset names are {VALID_DATASETS}")

  tables = {name: _load_table_data(name) for name in _TABLES_OF_INTEREST}

  # Get the keys that need to be filtered, and filter all tables to those.
  keys_to_keep = _DATASET_FILTERS[dataset_name](tables["index"])["key"]
  for name in tables.keys():
    table = tables[name]
    tables[name] = table[table["key"].isin(keys_to_keep)]

  # Handle initial grace period for missing epidemiology targets.
  tables["epidemiology"] = _maybe_set_zero_epidemiology_targets_in_grace_period(
      tables["epidemiology"], dataset_name)

  df = functools.reduce(_join_two, tables.values())
  df.rename(columns=_get_column_rename_map(dataset_name), inplace=True)

  df = _DATASET_FILTERS[dataset_name](df)

  # Drop rows without population or date data
  df = df[~df.population.isna()]
  df = df[~df[constants.DATE].isna()]

  # Drop sites with insufficient data for the possible prediction targets
  for target in _TARGETS:
    df = _drop_sites_with_insufficient_data(df, dataset_name, target.value)
  source_data = [_BASE_PATH + table for table in _TABLES_OF_INTEREST]
  return df, source_data
