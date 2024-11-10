"""Script that serves for splitting the dataset"""

import typer
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import yaml
from pathlib import Path
from rich.console import Console

from miad.utils.logs import create_logger

app = typer.Typer(help="Data splitting application for ML pipeline")
console = Console()

def load_config(config_path: Path) -> Dict:
	"""Load configuration from YAML file.

	Args:
		config_path: Path to configuration file
	Returns:
		Dictionary containing configuration
	"""
	with open(config_path) as conf_file:
		return yaml.safe_load(conf_file)

def load_dataset(features_path: Path, logger) -> pd.DataFrame:
	"""Load dataset from specified path.

	Args:
		features_path: Path to the features file
		logger: Logger instance
	Returns:
		Loaded DataFrame
	"""
	logger.info('Loading features')
	with console.status("[bold green]Loading dataset..."):
		return pd.read_csv(features_path)

def perform_split(
		dataset: pd.DataFrame,
		val_size: float,
		test_size: float,
		random_state: int,
		logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split dataset into train, test, and validation sets.

	Args:
		dataset: Input DataFrame to split
		val_size: Size of validation set
		test_size: Size of test set
		random_state: Random state for reproducibility
		logger: Logger instance
	Returns:
		Tuple of (train, test, validation) DataFrames
	"""
	logger.info('Splitting features into train, test, and validation sets')

	with console.status("[bold green]Performing dataset split..."):
		# First split: separate validation set
		train_full, val = train_test_split(
				dataset,
				test_size=val_size,
				random_state=random_state,
				stratify=dataset.iloc[:, 0]
		)

		# Second split: separate train and test from remaining data
		train, test = train_test_split(
				train_full,
				test_size=test_size / (1 - val_size),
				random_state=random_state,
				stratify=train_full.iloc[:, 0]
		)

	return train, test, val

def save_splits(
		splits_dict: Dict[str, Tuple[pd.DataFrame, Path]],
		logger
) -> None:
	"""Save split datasets to CSV files.

	Args:
		splits_dict: Dictionary containing split names, DataFrames and paths
		logger: Logger instance
	"""
	logger.info('Saving train, test and validation sets')

	with console.status("[bold green]Saving split datasets..."):
		for name, (data, path) in splits_dict.items():
			path = Path(path)
			path.parent.mkdir(parents=True, exist_ok=True)
			data.to_csv(path, index=False)
			console.print(f"[green]✓[/green] Saved {name} set to {path}")
			logger.debug(f'Saved {name} set to {path}')

def process_data_split(config_path: Path) -> None:
	"""Main function to handle the data splitting process.

	Args:
		config_path: Path to configuration file
	"""
	# Load configuration
	config = load_config(config_path)

	# Setup logger
	logger = create_logger(
			'DATA_SPLIT',
			log_level=config['base']['log_level']
	)

	# Load dataset
	dataset = load_dataset(
			Path(config['data_load']['output_path']) / config['data_load']['output_name'],
			logger
	)

	# Perform split
	train, test, val = perform_split(
			dataset=dataset,
			val_size=config['data_split']['val_size'],
			test_size=config['data_split']['test_size'],
			random_state=config['base']['random_state'],
			logger=logger
	)

	# Prepare splits dictionary for saving
	splits = {
		'train': (train, Path(config['data_split']['output_path']) / config['data_split']['output_name_train']),
		'test': (test, Path(config['data_split']['output_path'])  / config['data_split']['output_name_test']),
		'val': (val, Path(config['data_split']['output_path'])  / config['data_split']['output_name_val'])
	}

	# Save splits
	save_splits(splits, logger)

@app.command()
def main(
		config: Path = typer.Argument(
				...,
				help="Path to configuration file",
				exists=True,
				file_okay=True,
				dir_okay=False,
				resolve_path=True
		)
):
	"""
	Split dataset into train, test, and validation sets based on configuration file.
	"""
	try:
		console.print("[bold blue]Starting data splitting process...[/bold blue]")
		process_data_split(config)
		console.print("[bold green]Data splitting completed successfully! ✨[/bold green]")
	except Exception as e:
		console.print(f"[bold red]Error: {str(e)}[/bold red]")
		raise typer.Exit(code=1)

if __name__ == "__main__":
	app()