# DeepMind COVID-19 Modelling Toolkit

Contact [dm_c19_modelling@google.com](mailto:dm_c19_modelling@google.com?subject=[C19%20Evaluation])
for comments and questions.

This toolkit provides a starting point for developing and training machine
learning COVID-19 forecasting models. It is built on top of the DeepMind
COVID-19 Evaluation Toolkit ([README](../evaluation/README)). Specifically,
models are trained on datasets indexed by the
[`DatasetIndex`](../evaluation/README.md#dataset-index) protocol, and format and
index their forecasts using the
[`ForecastIndex`](../evaluation/README.md#forecast-index) protocol.

While the evaluation toolkit already provides some baseline models, this tookit
focuses on models that require iterative training, for example, models trained
with mini-batch stochastic gradient descent. A key purpose of this toolkit is to
provide base models which can be trained on various public datasets, for people
to extend and build on top of.

## Components

### Training datasets

Models are trained on a `definitions.TrainingDataset` object. This is
similar to the `Dataset` provided by the evaluation toolkit, however it
deliberately does not have targets for the forecast dates, to prevent
information leak during the modelling phase.

`training/dataset_factory.py` provides a light wrapper of the evaluation
framework dataset factory to produce a `definitions.TrainingDataset` for any
`Dataset` in the `DatasetIndex` via the `get_training_dataset` function.

For the rest of the modelling toolkit we simply refer to
`definitions.TrainingDataset` as the "training dataset" or simply the "dataset".

Any `TrainingDataset` can be described by its
`definitions.TrainingDatasetSpec` which contains information about important
properties of a dataset, such as names of the sites, names of input and target
features, temporal cadence of the observations, or the number of required forecast dates. The
`TrainingDatasetSpec.from_dataset` named constructor extracts a spec from a
dataset.

Finally, it is also possible to use `dataset_factory.remove_validation_dates`
to, given a `TrainingDataset`, obtain another `TrainingDataset` with same
`TrainingDatasetSpec` for which a full forecast worth of data has been removed
for the trailing dates (e.g. the last 4 weeks of data prior to the last
observation date for a 4-week prediction task). This way models can be validated
and early-stopped based on performance against this left-out data, and then
later fine tuned on the full dataset.

### Models.

Models trainable with this toolkit must implement the abstract base class
`definitions.TrainableModel` by inheriting from it. `TrainableModel`s receive
a `TrainingDatasetSpec` at construction time which specifies the task.
Child classes must implement the following methods:

* `def _build_training_generator(self, dataset: TrainingDataset)`: Returns an
iterator that yields batches of data for the passed `TrainingDataset`.

* `def training_update(self, previous_state, batch, global_step)`: Updates the
model state, using a batch of data, and returns the `updated_state`. This is the
equivalent of a training step update, and the state should contain everything
that changes as a result of a training iteration (e.g. model parameters
or batch normalization averages). `previous_state` is set to `None` for the
first call, and the returned value can be anything that can be serialized with
`pickle`. Models should not rely on any additional internal stateful information
that is not passed/returned on this model state, as this is the only part of the
model that will be stored in the checkpoints.

* `def _evaluate(self, model_state, dataset: TrainingDataset)`: Returns a
forecast for the input dataset.

Models in the `models` directory implement this interface.

#### JAX/Haiku models

`definitions.TrainableModel` does not make any assumptions about the ML library
used for the model. However, we also provide an abstract base class specific
for training loss-based models with [JAX](https://github.com/google/jax) and
[Haiku](https://github.com/deepmind/dm-haiku)
at `models/base_jax.LossMinimizerHaikuModel`.

This base class inherits from `definitions.TrainableModel`, but instead of
requiring a `training_update` method it requires child classes to directly
specify a loss function.

We provide some example model implementations based on this base class:

* `models/mlp.MLPModel`: Uses a multilayer perceptron (MLP) to make direct
  predictions for all forecast dates conditioned on a fixed window of data
  prior to the forecast dates.

* `models/lstm.LSTM`: Uses a Long Short-Term Memory network (LSTM) to model
  the trajectory of targets by conditioning on input data date by date, and
  predicting iteratively into the future.

* `models/seir_lstm.SEIRLSTM`: Similar to the LSTM model, but instead of
  using the LSTM to produce targets, it uses it to produce a daily `beta`
  parameter of a specialized
  [SEIR model](https://www.idmod.org/docs/hiv/model-seir.html).

### Model factory

The model factory `training/model_factory.py` returns instances of models that
implement that interface. After implementing a new model, the factory should
be modified to specify how to import and return the new model.

### Training

Training is configured via a config file, such as `base_config.py`, or a config
derived of that. This config specifies arguments to the dataset and model
factories, as well as additional training parameters.

Training occurs via the `training/runner.py` script. This script is designed to
run two separate jobs in parallel, the "train" and the "eval" job:

* The "train" job (`--mode="train"`) trains the model on a version of the
  dataset for which some validation dates have been removed from the end. This
  training happens for a fixed number of steps and the model state is
  periodically stored in a checkpoint.

* The "eval" job (`--mode="eval"`) evaluates the performance of the model for
  the "train" job checkpoints on the validation dates, and uses this information
  to save the model with the best validation performance accross the training
  trajectory (i.e. early stopping). Once the "train" job finishes, the "eval"
  job takes the model checkpoint with the best validation performance and
  fine-tunes that model on the full training dataset for a fixed number of
  steps.

The eval job can be configured to submit a forecast to the `ForecastIndex` at
the end of fine tuning.

## Running example

We provide an example bash script to download data and run training and
evaluation:

```shell
bash dm_c19_modelling/modelling/scripts/example.sh
```

The steps within the example script include the following.

### Pre-requisites

As a pre-requisite it is necessary to generate a dataset. Using the evaluation
toolkit, run:

```shell
cd dm_c19_modelling/evaluation
python3 run_download_data.py --project_directory=<project_directory> \
--dataset_name=<dataset_name>
```

See evaluation toolkit ([README](../evaluation/README)) for more details.

### Training the model

Running an instance of the training and the evaluation jobs, involves running
the following two commands.

```shell
cd dm_c19_modelling/modelling
python3 training/runner.py \
    --config=training/base_config.py:<project_directory> \
    --config.dataset.dataset_name=<dataset_name> \
    --config.model.model_name=<model_name> \
    --config.checkpointer.directory=<checkpoint_directory> \
    --mode="train"
```

```shell
cd dm_c19_modelling/modelling
python3 training/runner.py \
    --config=training/base_config.py:<project_directory> \
    --config.dataset.dataset_name=<dataset_name> \
    --config.model.model_name=<model_name> \
    --config.checkpointer.directory=<checkpoint_directory> \
    --mode="eval"
```


The two jobs should be run simultaneously and share a file system, because they
communicate via the `<checkpoint_directory>`. The training job runs
independently from beginning to end, and the evaluation job monitors the
progress of the training job. One simple option is to start them in separate
consoles, or set the training job in the background by adding ` &` at the end,
and then start the evaluation job on the same console. They can also be run on
separate machines so long as they have access to a shared filesystem.

Training and evaluation are fully checkpointed, and the processes can be stopped
and restarted at any time, and they should pick where they left off, so long
as the same `<checkpoint_directory>` is passed. Note: the training job should
not be run for long periods while the eval job is not running because that would
systematically prevent the eval job from evaluating a large number of "train"
checkpoints.

In the example we are overriding config parameters from the command line,
but it is also possible to directly override parameters directly in
`base_config.py` or create separate child configs for each experiment:

```python
# some_experiment_specific_config.py

from dm_c19_modelling.training import base_config

PROJECT_DIRECTORY = "/tmp/my_evaluation_framework"

def get_config():
  config = base_config.get_config(PROJECT_DIRECTORY):
  config.dataset.dataset_name = "covid_open_data_us_states"
  config.model.model_name = "seir_lstm"
  config.checkpointer.directory = "/tmp/model_checkpoint"
  return config

```

and then pass that config at launch time
`--config=some_experiment_specific_config.py`.

### Submitting the forecast

If a forecast name is passed to the `eval` job via
`--forecast_name=<forecast_name>` it will automatically submit a forecast for
the final fine-tuned model to the `ForecastIndex` of the evaluation framework
at the end of training.

Alternatively, the forecast can be submitted post-training by running:

```shell
python3 submit_forecast.py \
    --checkpoint_path=<checkpoint_directory>/latest_fine_tune  \
    --forecast_name=<forecast_name>
```

### Advanced analysis of the model

The evaluation toolkit provides model-agnostic visualization and comparison
of forecasts, however, models can also provide additional auxiliary outputs for
advanced model interpretation. For example, the `seir_lstm` model provides
a full fit of the curves for the full length of inputs dates, as well as
values for the parameters and state of the SEIR differential equation as
function of time.

`demos/load_pre_trained.ipynb` shows an example of how to load a pretrained
model in a `jupyter notebook`, and perform additional inspection of these extra
outputs.

The evaluation toolkit's visualization utilities can also be used to plot error
bars and forecast trajectories.

## Setup

### Python dependencies

We specify dependencies in `requirements-modelling.txt`.
