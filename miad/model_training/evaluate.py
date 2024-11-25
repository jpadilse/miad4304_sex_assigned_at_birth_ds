"""Script that serves for model evaluation"""

import tensorflow as tf
import json
import pandas as pd
from pathlib import Path

import typer
from typing import Dict
import yaml

from miad.model_training.metrics import binary_classifier_metrics, max_f1_score
from miad.model_training.perfomance_plots import (
	generate_calibration_curve, generate_confusion_matrix, generate_gains_curve,
	generate_ks_curve, generate_lift_curve, generate_precision_recall_curve,
	generate_roc_curve
)
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

	f1_thr, _ = max_f1_score(y_test, prediction)
	report = binary_classifier_metrics(y_test, prediction, threshold=f1_thr)

	logger.info('Save metrics')
	reports_folder = Path(config['evaluate']['reports_dir']) / "figures"
	metrics_path = reports_folder / config['evaluate']['metrics_file']

	json.dump(obj=report, fp=open(metrics_path, 'w'))
	logger.info(f'Binary metrics file saved to : {metrics_path}')

	logger.info('Save Calibration curve')
	plt = generate_calibration_curve(y_true=y_test, y_proba=prediction)
	image_path = reports_folder / config['evaluate']['calibration_curve_image']
	plt.savefig(image_path)
	logger.info(f'Calibration curve saved to: {image_path}')

	logger.info('Save ROC curve')
	plt = generate_roc_curve(y_true=y_test, y_proba=prediction)
	image_path = reports_folder / config['evaluate']['roc_curve_image']
	plt.savefig(image_path)
	logger.info(f'ROC curve saved to: {image_path}')

	logger.info('Save Precision-Recall curve')
	plt = generate_precision_recall_curve(y_true=y_test, y_proba=prediction)
	image_path = reports_folder / config['evaluate']['pr_curve_image']
	plt.savefig(image_path)
	logger.info(f'Precision-Recall curve saved to: {image_path}')

	logger.info('Save KS curve')
	plt = generate_ks_curve(y_true=y_test, y_proba=prediction)
	image_path = reports_folder / config['evaluate']['ks_curve_image']
	plt.savefig(image_path)
	logger.info(f'KS curve saved to: {image_path}')

	logger.info('Save Gains curve')
	plt = generate_gains_curve(y_true=y_test, y_proba=prediction)
	image_path = reports_folder / config['evaluate']['gains_curve_image']
	plt.savefig(image_path)
	logger.info(f'Gains curve saved to: {image_path}')

	logger.info('Save Lift curve')
	plt = generate_lift_curve(y_true=y_test, y_proba=prediction)
	image_path = reports_folder / config['evaluate']['lift_curve_image']
	plt.savefig(image_path)
	logger.info(f'Gains Lift saved to: {image_path}')

	logger.info('Save confusion matrix')
	plt = generate_confusion_matrix(
			y_true=y_test,
			y_proba=prediction,
			threshold=f1_thr
	)
	image_path = reports_folder / config['evaluate']['confusion_matrix_image']
	plt.savefig(image_path)
	logger.info(f'Confusion matrix saved to: {image_path}')


if __name__ == '__main__':
	app()
