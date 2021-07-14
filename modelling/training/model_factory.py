# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Model factory."""

from dm_c19_modelling.modelling.models import lstm
from dm_c19_modelling.modelling.models import mlp
from dm_c19_modelling.modelling.models import seir_lstm


_MODEL_MAPPING = {
    "mlp": mlp.MLPModel,
    "lstm": lstm.LSTM,
    "seir_lstm": seir_lstm.SEIRLSTM,
}


def get_model(dataset_spec, model_name, model_specific_kwargs, **kwargs):
  """Model factory."""

  # This factory is called by the training script via
  #     `get_model(dataset_spec, **config.model)`

  # This is a very specific implementation of a factory that is setup to build
  # different models based on a specific config structure containing
  # `model_name` and `model_specific_kwargs`. Users implementing new models may
  # fully adjust this factory to use the model config in a different way.

  # Note the `dataset_spec` passed here includes fields such as statistics of
  # the data used for normalization. If reloading a pretrained model care should
  # be taken to pass the same `dataset_spec` as that used at training, when
  # building the model.
  model_kwargs = dict(dataset_spec=dataset_spec, **kwargs)

  # Add model specific options.
  model_kwargs.update(model_specific_kwargs[model_name])

  # Instantiate the correct class.
  model_class = _MODEL_MAPPING[model_name]
  return model_class(**model_kwargs)
