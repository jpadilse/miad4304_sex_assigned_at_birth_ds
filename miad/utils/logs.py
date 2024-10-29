"""Provides functions to create loggers."""

import logging
from typing import Union
import sys
from functools import lru_cache

DEFAULT_LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"

@lru_cache(maxsize=1)
def create_console_handler(log_format: str = DEFAULT_LOG_FORMAT) -> logging.StreamHandler:
	"""
	Create a console handler for logging.

	Args:
		log_format (str): The format string for the log messages. Defaults to DEFAULT_LOG_FORMAT.

	Returns:
		logging.StreamHandler: A StreamHandler which logs to stdout.
	"""
	console_handler = logging.StreamHandler(sys.stdout)
	formatter = logging.Formatter(log_format)
	console_handler.setFormatter(formatter)
	return console_handler

def create_logger(
		name: str = __name__,
		log_level: Union[str, int] = logging.DEBUG,
		handler: logging.Handler = None,
		propagate: bool = False
) -> logging.Logger:
	"""
	Create a logger with the specified name and log level.

	Args:
		name (str): The name of the logger. Defaults to __name__.
		log_level (Union[str, int]): The logging level; can be a string name or integer value.
		handler (logging.Handler, optional): A custom handler to use. If None, a console handler is created.
		propagate (bool): Whether to propagate messages to parent loggers. Defaults to False.

	Returns:
		logging.Logger: A configured Logger instance.
	"""
	logger = logging.getLogger(name)
	logger.setLevel(log_level)

	# Remove existing handlers to prevent duplicate outputs
	logger.handlers.clear()

	if handler is None:
		handler = create_console_handler()

	logger.addHandler(handler)
	logger.propagate = propagate

	return logger