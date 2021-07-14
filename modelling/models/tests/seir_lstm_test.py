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
"""Tests for `seir_lstm.py`."""

from unittest import mock
from absl.testing import absltest

from dm_c19_modelling.modelling.models import seir_lstm

import numpy as np


class SEIRLSTMTest(absltest.TestCase):

  def test_polynomial_extrapolation(self):
    mock_seir_lstm_instance = mock.Mock()
    mock_seir_lstm_instance._param_extrapolation_poly_degree = 1
    mock_seir_lstm_instance._param_extrapolation_context_steps = 2

    params_sequence = np.array(
        [[0., -3.],  # step 0 (site 0, site 1)  # Should be ignored.
         [7., 9.],  # step 1 (site 0, site 1)
         [8., 7.]],  # step 2 (site 0, site 1)
        )

    actual = seir_lstm.SEIRLSTM._extrapolate_seir_params(
        mock_seir_lstm_instance,
        params_sequence,
        num_steps=3)

    expected = np.array(
        [[9., 5.],  # step 3 (site 0, site 1)
         [10., 3.],  # step 4 (site 0, site 1)
         [11., 1.]],  # step 5 (site 0, site 1)
        )

    np.testing.assert_allclose(actual, expected, rtol=1e-06)


if __name__ == "__main__":
  absltest.main()
