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
"""Functions to calculate metrics on forecasts."""

import numpy as np


def check_shape_wrapper(func):
  """Wrapper that checks the shapes of predictions and ground truth."""
  def wrapped_func(predictions, ground_truth):
    assert predictions.shape == ground_truth.shape, (
        f"Predictions array has shape {predictions.shape}, ground truth has "
        f"shape {ground_truth.shape}")
    assert predictions.ndim == 3, (
        "Metrics calculation expects rank 3 predictions and ground truth.")
    assert predictions.shape[-1] == 1, (
        "Metrics calculation expects a single target")
    return func(predictions, ground_truth)
  wrapped_func.__name__ = func.__name__
  return wrapped_func


@check_shape_wrapper
def rmse(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
  """Gets the RMSE averaged over time and sites for the given predictions."""
  squared_error = (predictions - ground_truth) ** 2
  return np.sqrt(np.mean(squared_error))


@check_shape_wrapper
def mae(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
  """Gets MAE averaged over time and sites for the given predictions."""
  return np.mean(np.abs(predictions - ground_truth))
