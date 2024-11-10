"""Script that serves for model evaluation"""

import tensorflow as tf
import json
import pandas as pd
from pathlib import Path

import typer
from sklearn.metrics import confusion_matrix, roc_auc_score
from typing import Dict
import yaml

from miad.model_training.perfomance_plots import generate_confusion_matrix
from miad.utils.logs import create_logger

app = typer.Typer()

def load_config(config_path: Path) -> Dict:
	"""Load configuration from YAML file.

	Args:
		config_path: Path to configuration file
	Returns:
		Dictionary containing configuration
	"""
	with open(config_path) as conf_file:
		return yaml.safe_load(conf_file)

@app.command()
def main(config_path: Path = typer.Option(..., help="Path to the configuration file")) -> None:
	"""Evaluate model.
	Args:
		config_path {Text}: path to config
	"""

	config = load_config(config_path)

	logger = create_logger('EVALUATE', log_level=config['base']['log_level'])

	logger.info('Load model')
	model_path = Path(config['model']['output_dir']) / "model.keras"
	model = tf.keras.models.load_model(model_path)

	logger.info('Load test dataset')
	test_df = pd.read_csv(
			Path(config['data_split']['output_path'])  / config['data_split']['output_name_val']
	)

	logger.info('Evaluate (build report)')
	y_test = test_df.iloc[:, 0].values
	X_test = test_df.iloc[:, 1].values

	prediction = model.predict(X_test)
	f1 = roc_auc_score(y_test, prediction)

	report = {
		'f1': f1,
		'actual': y_test,
		'predicted': prediction
	}

	logger.info('Save metrics')
	# save f1 metrics file
	reports_folder = Path(config['evaluate']['reports_dir']) / "figures"
	metrics_path = reports_folder / config['evaluate']['metrics_file']

	json.dump(
			obj={'f1_score': report['f1']},
			fp=open(metrics_path, 'w')
	)

	logger.info(f'F1 metrics file saved to : {metrics_path}')

	logger.info('Save confusion matrix')
	# save confusion_matrix.png
	plt = generate_confusion_matrix(
			y_true=y_test,
			y_proba=prediction,
			threshold=0.5
	)
	confusion_matrix_png_path = reports_folder / config['evaluate']['confusion_matrix_image']
	plt.savefig(confusion_matrix_png_path)
	logger.info(f'Confusion matrix saved to : {confusion_matrix_png_path}')


if __name__ == '__main__':
	app()
