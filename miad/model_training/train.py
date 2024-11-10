"""Script that serves as the entry point for model training"""

import math
from pathlib import Path
from typing import Dict, Any, Tuple

import mlflow
import mlflow.tensorflow
import pandas as pd
import tensorflow as tf
import typer
import yaml
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
	Activation,
	Bidirectional,
	Dense,
	Dropout,
	LSTM,
	Lambda,
	TextVectorization
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from miad.model_training.layers import OneHotLayer

from miad.utils.logs import create_logger

logger = create_logger('TRAIN')

app = typer.Typer()

mlflow.tensorflow.autolog(log_models=True)


def load_config(config_path: str) -> Dict[str, Any]:
	"""
	Load configuration from a YAML file.

	Args:
		config_path (str): Path to the configuration file.

	Returns:
		Dict[str, Any]: Loaded configuration.

	Raises:
		FileNotFoundError: If the config file is not found.
		yaml.YAMLError: If there's an error parsing the YAML file.
	"""
	try:
		with open(config_path, 'r') as conf_file:
			return yaml.safe_load(conf_file)
	except FileNotFoundError:
		raise FileNotFoundError(f"Configuration file not found: {config_path}")
	except yaml.YAMLError as e:
		raise yaml.YAMLError(f"Error parsing YAML file: {e}")

def load_and_split(base_path: str, file_name: str) -> Tuple[pd.Series, pd.Series]:
	"""
	Load data from a CSV file and split it into two Series.

	Args:
		base_path (str): Base directory path
		file_name (str): Name of the file to load

	Returns:
		Tuple[pd.Series, pd.Series]: Features and target series
	"""
	data_path = Path(base_path) / file_name
	try:
		data = pd.read_csv(
				data_path,
				usecols=[0, 1],
				header=None,
				names=['first', 'second']
		)
	except FileNotFoundError:
		raise FileNotFoundError(f"The file at {data_path} does not exist.")
	except pd.errors.EmptyDataError:
		raise pd.errors.EmptyDataError(f"The file at {data_path} is empty.")

	if data.shape[1] != 2:
		raise ValueError(f"The CSV file at {data_path} should have two columns.")

	return data.iloc[:, 1].values, data.iloc[:, 0].values

def create_model(
		input_neurons: int,
		input_dropout: float,
		glossary: list,
		max_len: int
) -> Sequential:
	model = Sequential([
		TextVectorization(
				split='character',
				output_mode='int',
				output_sequence_length=max_len,
				vocabulary=glossary
		),
		OneHotLayer(len(glossary) + 2),
		Bidirectional(
				LSTM(input_neurons, return_sequences=True),
				backward_layer=LSTM(
						input_neurons,
						return_sequences=True,
						go_backwards=True
				)
		),
		Dropout(input_dropout),
		Bidirectional(LSTM(input_neurons)),
		Dropout(input_dropout),
		Dense(1),
		Activation('sigmoid')
	])
	return model

def compile_and_fit_model(
		model: Sequential,
		x_train: pd.Series,
		y_train: pd.Series,
		x_test: pd.Series,
		y_test: pd.Series,
		params: Dict
) -> Sequential:
	lr_schedule = ExponentialDecay(
			initial_learning_rate=params['training']['initial_learning_rate'],
			decay_steps=params['training']['epochs'] *
						math.ceil(len(x_train) / params['training']['batch_size']),
			decay_rate=params['training']['decay_rate']
	)
	opt = AdamW(
			learning_rate=lr_schedule,
			weight_decay=params['training']['weight_decay']
	)

	model.compile(
			optimizer=opt,
			loss='binary_crossentropy',
			metrics=['accuracy']
	)

	early_stopping_cb = tf.keras.callbacks.EarlyStopping(
			monitor='val_accuracy',
			min_delta=params['training']['early_stopping_delta'],
			patience=params['training']['early_stopping_patience'],
			mode='max'
	)

	model.fit(
			x_train,
			y_train,
			validation_data=(x_test, y_test),
			epochs=params['training']['epochs'],
			batch_size=params['training']['batch_size'],
			callbacks=[early_stopping_cb],
			verbose=2
	)

	return model

@app.command()
def main(
		config_path: Path = typer.Option(..., help="Path to the configuration file"),
		experiment_name: str = typer.Option("default", help="MLflow experiment name")
):
	"""Main function to train the model with MLflow tracking."""
	# Set up MLflow
	mlflow.set_experiment(experiment_name)

	# Start MLflow run
	with mlflow.start_run() as run:
		logger.info(f"Started MLflow run: {run.info.run_id}")

		try:
			# Load parameters
			params = load_config(str(config_path))

			# Log parameters to MLflow
			mlflow.log_params(
					{
						"input_neurons": params['model']['n_neurons'],
						"dropout_rate": params['model']['dropout'],
						"max_len": params['model']['max_len'],
						"initial_learning_rate": params['training']['initial_learning_rate'],
						"batch_size": params['training']['batch_size'],
						"epochs": params['training']['epochs'],
						"decay_rate": params['training']['decay_rate'],
						"weight_decay": params['training']['weight_decay']
					}
			)

			logger.info('Load and split the dataset')
			x_train, y_train = load_and_split(
					params['data']['train_path'],
					params['data']['train_file']
			)
			x_test, y_test = load_and_split(
					params['data']['test_path'],
					params['data']['test_file']
			)

			logger.info('Build and fit model')
			glossary = list('abcdefghijklmnopqrstuvwxyz ')
			max_len = params['model']['max_len']

			model = create_model(
					params['model']['n_neurons'],
					params['model']['dropout'],
					glossary,
					max_len
			)

			model = compile_and_fit_model(
					model,
					x_train,
					y_train,
					x_test,
					y_test,
					params
			)

			logger.info('Persist model')
			model_output_path = Path(params['model']['output_dir']) / 'model.keras'
			model_output_path.parent.mkdir(parents=True, exist_ok=True)
			model.save(str(model_output_path))

			# Log the model to MLflow
			mlflow.tensorflow.log_model(
					model,
					"model",
					registered_model_name="sex_classifier"
			)

		except Exception as e:
			logger.error(f"An error occurred during model training: {e}")
			mlflow.log_param("error", str(e))
			raise

if __name__ == '__main__':
	app()
