"""Script that serves for data loading"""

import typer
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

from miad.utils.logs import create_logger

app = typer.Typer()

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

def process_dataset(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
	"""
	Process the dataset.

	Args:
		data (pd.DataFrame): Input dataset.
		target_column (str): Name of the target column.

	Returns:
		pd.DataFrame: Processed dataset.
	"""
	return (
		data
		.rename(columns={'sex': target_column})
		.assign(
				first_name=lambda df: (
					df['first_name']
					.fillna('')
					.str.lower()
					.str.normalize("NFKD")
					.str.encode("ascii", errors="ignore")
					.str.decode("utf-8")
				),
				second_name=lambda df: (
					df['second_name']
					.fillna('')
					.str.lower()
					.str.normalize("NFKD")
					.str.encode("ascii", errors="ignore")
					.str.decode("utf-8")
				),
				full_name = lambda x: (x['first_name'] + ' ' + x['second_name']).str.strip()
		)
		.filter(items=[target_column, 'full_name'])
	)

def load_data(file_path: Path) -> pd.DataFrame:
	"""
	Load data from a CSV file.

	Args:
		file_path (Path): Path to the CSV file.

	Returns:
		pd.DataFrame: Loaded data.

	Raises:
		FileNotFoundError: If the CSV file is not found.
	"""
	try:
		return pd.read_csv(file_path)
	except FileNotFoundError:
		raise FileNotFoundError(f"Data file not found: {file_path}")

def save_data(data: pd.DataFrame, file_path: Path) -> None:
	"""
	Save data to a CSV file.

	Args:
		data (pd.DataFrame): Data to save.
		file_path (Path): Path to save the CSV file.
	"""
	data.to_csv(file_path, index=False, header=False)

@app.command()
def data_load(config_path: Path = typer.Option(..., help="Path to the configuration file")):
	"""
	Load raw data, process it, and save the result.
	"""
	config = load_config(str(config_path))
	logger = create_logger('DATA_LOAD', log_level=config['base']['log_level'])

	try:
		logger.info("Loading dataset")
		input_path = Path(config["data_load"]["input_path"]) / config["data_load"]["input_name"]
		data = load_data(input_path)

		logger.info("Processing dataset")
		dataset = process_dataset(data, config['data_load']['target'])

		logger.info('Saving processed data')
		output_path = Path(config["data_load"]["output_path"]) / config["data_load"]["output_name"]
		save_data(dataset, output_path)

		logger.info("Data loading and processing completed successfully")
	except Exception as e:
		logger.error(f"An error occurred during data loading and processing: {e}")
		raise

if __name__ == "__main__":
	app()

