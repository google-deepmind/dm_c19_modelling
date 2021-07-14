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
"""Utilities for plotting metrics."""

import pathlib
from typing import Any, List, Sequence

from absl import logging
from dm_c19_modelling.evaluation import base_indexing
from dm_c19_modelling.evaluation import constants
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import pandas as pd


def plot_metrics(metrics_df: pd.DataFrame, target_name: str,
                 last_observation_date: str, eval_dataset_creation_date: str,
                 forecast_horizon: int,
                 forecast_index_entries: Sequence[base_indexing.IndexEntryType],
                 num_dates: int, num_sites: int, cadence: int,
                 dropped_sites: np.ndarray) -> plt.Figure:
  """Plots metrics dataframe as a series of bar charts.

  Args:
    metrics_df: Dataframe of metrics, with columns [forecast_id, metric_name,
      metric_value, target_name].
    target_name: the target being predicted.
    last_observation_date: the last date in the training data.
    eval_dataset_creation_date: the creation date of the dataset used for
      evaluation.
    forecast_horizon: the number of days into the future that the forecasts
      extend to.
    forecast_index_entries: the entries in the forecast index for each of the
      forecasts that are included in the metrics dataframe.
    num_dates: the number of dates included in this evaluation.
    num_sites: the number of sites included in this evaluation.
    cadence: the cadence of the forecasts i.e. a cadence of 1 corresponds to
      daily forecasts, a cadence of 7 corresponds to weekly forecasts.
    dropped_sites: optional list of sites that were dropped during evaluation
      from at least one forecast to ensure that all forecasts are for the same
      sites.

  Returns:
      A series of bar plots, one for each metric calculated in the dataframe,
      evaluating different forecasts against each other.
  """
  fig = plt.figure(figsize=(4, 3))
  plot_width = 2
  offset = 0
  column_width = 0.8

  axes = []
  metric_names = metrics_df.metric_name.unique()
  for _ in metric_names:
    ax = fig.add_axes([offset, 0.1, plot_width, 1.])
    ax.grid(axis="y", alpha=0.3, which="both", zorder=0)
    axes.append(ax)
    offset += plot_width * 1.2

  colour_map = plt.get_cmap("tab20c")(
      np.linspace(0, 1.0, len(forecast_index_entries)))
  x_centers = np.arange(len(forecast_index_entries))

  for ax_idx, metric_name in enumerate(metric_names):
    x_offset = ax_idx * column_width - plot_width / 2 + column_width / 2
    x_values = x_centers + x_offset
    ax = axes[ax_idx]
    for bar_idx, forecast_entry in enumerate(forecast_index_entries):
      forecast_id = forecast_entry["forecast_id"]
      row = metrics_df.query(
          f"forecast_id=='{forecast_id}' and metric_name=='{metric_name}'")
      assert len(row) == 1, (
          "Duplicate entries found in metrics dataframe. "
          f"Found {len(row)} entries for {forecast_id} and {metric_name}")
      row = row.iloc[0]
      metric_value = row.metric_value

      ax.bar(
          x_values[bar_idx],
          metric_value,
          width=column_width,
          zorder=2,
          color=colour_map[bar_idx],
          label=_get_model_label(forecast_entry))
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_ylabel(metric_name)

  axes[0].legend(
      ncol=len(forecast_index_entries),
      loc="center left",
      bbox_to_anchor=[0., 1.07],
      frameon=False)
  fig.text(0, 0, _get_plot_footnote(num_sites, num_dates, dropped_sites,
                                    cadence))
  fig.suptitle(
      _get_plot_title(target_name, last_observation_date,
                      eval_dataset_creation_date, forecast_horizon),
      y=1.35,
      x=1)
  return fig


def _get_model_label(forecast_entry: base_indexing.IndexEntryType) -> str:
  """Gets a description of a model from its entry in the forecast index."""
  description = str(forecast_entry["forecast_id"])
  if "model_description" in forecast_entry["extra_info"]:
    description += f": {forecast_entry['extra_info']['model_description']}"
  return description


def _get_plot_title(target_name: str, last_observation_date: str,
                    eval_dataset_creation_date: str,
                    forecast_horizon: int) -> str:
  """Gets the title of the plot."""
  return (
      f"Comparison of metrics for predicting {target_name}. Forecast date: "
      f"{last_observation_date}, forecast horizon: {forecast_horizon} days, "
      f"evaluation reporting date: {eval_dataset_creation_date}.")


def _get_plot_footnote(num_sites: int, num_dates: int,
                       dropped_sites: np.ndarray, cadence: int):
  """Gets the footnote to be added to the plot."""
  footnote = (
      f"Forecasts evaluated in this plot have a cadence of {cadence} days. "
      f"{num_dates} dates and {num_sites} sites were included in the "
      "evaluation that produced this plot.")
  if dropped_sites.size:
    footnote += (
        "Note that the following sites were dropped from some forecasts during "
        f"evaluation to achieve an overlapping set of sites: {dropped_sites}")
  return footnote


def _plot_trajectories(
    all_forecast_entries: List[Any],
    all_forecast_arrays: List[Any],
    target_name: constants.Targets,
    num_sites: int,
    eval_dataset: Any = None
) -> plt.Figure:
  """Plots trajectories.

  Args:
    all_forecast_entries: TODO
    all_forecast_arrays: TODO
    target_name: the target being predicted.
    num_sites: number of sites to plot
    eval_dataset: evaluation dataset

  Returns:
    Figure.
  """
  fig = plt.figure(figsize=(16, 16))

  dates = all_forecast_arrays[0].dates_array
  num_dates = len(dates)
  forecast_x = np.arange(num_dates)
  x = forecast_x.copy()
  x_stride = 14  # Weekly x tick strides.
  previous_x = None

  avg_values = []
  for fa in all_forecast_arrays:
    avg_values.append(np.squeeze(fa.data_array, axis=2).mean(axis=0))
  site_indices = np.argsort(np.max(avg_values, axis=0))[::-1][:num_sites]
  site_names = all_forecast_arrays[0].sites_array[site_indices]
  n = len(site_names)
  nrows = int(np.ceil(np.sqrt(n)))
  ncols = int(np.ceil(n / nrows))
  axes = fig.subplots(nrows, ncols)
  fig.subplots_adjust(hspace=0.35)
  flat_axes = sum(map(list, axes), [])
  for _ in range(nrows * ncols - n):
    ax = flat_axes.pop()
    fig.delaxes(ax)
  num_colors = len(all_forecast_entries) + 1
  colormap = plt.get_cmap("tab20")
  colors = [colormap(i / num_colors) for i in range(num_colors)]

  if eval_dataset is not None:
    num_previous_dates = num_dates
    previous_dates = eval_dataset.training_dates[-num_previous_dates:]
    previous_x = np.arange(num_previous_dates)
    previous_true_ys = eval_dataset.training_targets[-num_previous_dates:, :, 0]
    forecast_true_ys = eval_dataset.evaluation_targets[-num_previous_dates:, :,
                                                       0]

    forecast_x += num_previous_dates
    dates = np.concatenate([previous_dates, dates])
    x = np.concatenate([previous_x, forecast_x])
    num_dates = len(dates)

  x_idx = np.arange(num_dates)[::-1][::x_stride][::-1]
  # Center the x axis date ticks around the forecast date.
  diffs = x_idx - forecast_x[0]
  smallest_diff = np.argmin(np.abs(diffs))
  x_idx -= diffs[smallest_diff]
  x_idx = np.clip(x_idx, 0, len(x) - 1)

  for ax, site_name in zip(flat_axes, site_names):
    title = f'site_name="{site_name}"'
    ax.set_title(title)
    site_idx = all_forecast_arrays[0].sites_array.tolist().index(site_name)
    if previous_x is not None:
      previous_true_y = previous_true_ys[:, site_idx]
      forecast_true_y = forecast_true_ys[:, site_idx]
      # Plot vertical forecast date line.
      combined_y = np.concatenate([previous_true_y, forecast_true_y])
      mn = np.min(combined_y)
      mx = np.max(combined_y)
      ax.plot(
          [forecast_x[0] - 0.5] * 2, [mn, mx],
          color=(0.5, 0.5, 0.5),
          linestyle="--",
          label=f"(forecast date={dates[forecast_x[0]]})")
      # Plot past and future true data.
      ax.plot(previous_x, previous_true_y, color="k")
      ax.plot(forecast_x, forecast_true_y, color="k", label="true_data")

    # Plot the forecast trajectories.
    ax.axes.set_prop_cycle(color=colors)  # Color forecast curves differently.
    for forecast_entry, forecast_array in zip(all_forecast_entries,
                                              all_forecast_arrays):
      y = forecast_array.data_array[:, site_idx, 0]
      ax.plot(
          forecast_x, y, label=f"forecast_id={forecast_entry['forecast_id']}")
      ax.set_xticks(x[x_idx])
      ax.set_xticklabels(dates[x_idx], rotation=30)
      if ax.is_last_row():
        ax.set_xlabel("Date")
      if ax.is_first_col():
        ax.set_ylabel(target_name.value)
      if ax.is_first_col() and ax.is_first_row():
        ax.legend(loc="upper left")

  return fig


def plot_trajectories_and_save(directory: str, forecast_ids: Sequence[str],
                               eval_dataset_creation_date: str,
                               forecast_horizon: int, save: bool,
                               target_name: constants.Targets,
                               all_forecast_entries: List[Any],
                               all_forecast_arrays: List[Any],
                               num_sites: int = 16,
                               eval_dataset: Any = None,
                               overwrite: bool = False) -> None:
  """Plots trajectories and saves them to file."""
  fig = _plot_trajectories(all_forecast_entries, all_forecast_arrays,
                           target_name, num_sites, eval_dataset=eval_dataset)

  if save:
    trajectories_dir = pathlib.Path(directory) / "trajectories"
    filename_base = (
        f"trajectories_{'_'.join(forecast_ids)}_{eval_dataset_creation_date}_"
        f"{forecast_horizon}d")
    plot_filepath = trajectories_dir / f"{filename_base}.png"

    if not trajectories_dir.exists():
      trajectories_dir.mkdir(parents=True)
    if not overwrite and plot_filepath.exists():
      raise IOError(f"Trajectories already exist at {plot_filepath}")
    logging.info("Saving trajectory plots to %s", plot_filepath)
    fig.savefig(plot_filepath, format="png", bbox_inches="tight")
