"""
Binary Classification Model Performance Visualization

This module provides a collection of visualization tools for evaluating binary
classification model performance. It includes implementations of various performance
curves and metrics commonly used in binary classification tasks.
"""
from functools import wraps
# Standard library imports
from typing import Union

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn metrics and utilities
from sklearn.metrics import (
	auc,
	roc_curve,
	brier_score_loss,
	confusion_matrix,
	precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import normalize

def validate_binary_inputs(func):
	"""Decorator for validating binary classification inputs."""
	@wraps(func)
	def wrapper(
			y_true: Union[list, np.ndarray],
			y_proba: Union[list, np.ndarray],
			*args,
			**kwargs
	):
		try:
			# Convert and validate inputs
			y_true = np.asarray(y_true, dtype=np.float64).flatten()
			y_proba = np.asarray(y_proba, dtype=np.float64).flatten()

			# Validate shapes
			if y_true.shape != y_proba.shape:
				raise ValueError(
						f"Shape mismatch: y_true {y_true.shape} != y_proba {y_proba.shape}"
				)

			# Validate binary values
			if not np.all(np.isin(y_true, [0, 1])):
				raise ValueError("y_true must contain only binary values (0 or 1)")

			# Validate probabilities
			if np.any((y_proba < 0) | (y_proba > 1)):
				raise ValueError("y_proba must contain probabilities between 0 and 1")

			return func(y_true, y_proba, *args, **kwargs)

		except Exception as e:
			plt.close()  # Clean up in case of error
			raise RuntimeError(f"Input validation failed: {str(e)}")

	return wrapper

@validate_binary_inputs
def generate_calibration_curve(
		y_true: Union[list, np.ndarray],
		y_proba: Union[list, np.ndarray],
		n_bins: int = 5,
		figsize: tuple = (5, 5),
		classifier_color: str = '#DA0081',
		perfect_line_color: str = '#200020'
) -> plt.Figure:
	"""
	Generate and plot a Calibration Curve with Brier Score.

	Args:
		y_true: Array-like object containing the actual binary labels.
		y_proba: Array-like object containing the predicted probabilities.
		n_bins: Number of bins to discretize the [0, 1] interval. Defaults to 5.
		figsize: Tuple specifying figure dimensions. Defaults to (5, 5).
		classifier_color: Color for the classifier curve. Defaults to '#DA0081'.
		perfect_line_color: Color for the perfect calibration line. Defaults to '#200020'.

	Returns:
		matplotlib.figure.Figure: The generated Calibration curve plot.

	Raises:
		ValueError: If input validation fails.
	"""

	try:
		# Calculate metrics
		prob_true, prob_prediction = calibration_curve(
				y_true,
				y_proba,
				n_bins=n_bins
		)
		brier_score = brier_score_loss(y_true, y_proba)

		# Create figure and axis
		fig, ax = plt.subplots(figsize=figsize)

		# Plot curves
		ax.plot(
				prob_prediction,
				prob_true,
				color=classifier_color,
				lw=2,
				label=f'Classifier (Brier Score = {brier_score:.2f})'
		)
		ax.plot(
				[0, 1],
				[0, 1],
				color=perfect_line_color,
				lw=2,
				linestyle='--',
				label='Perfect Calibration (Brier Score = 0.00)'
		)

		# Configure plot aesthetics
		ax.set_title('Calibration Curve')
		ax.set_xlabel('Mean Predicted Probability')
		ax.set_ylabel('Fraction of Positives')
		ax.set_xlim([-0.05, 1.05])
		ax.set_ylim([-0.05, 1.05])
		ax.grid(True, alpha=0.3)
		ax.legend(loc='best')

		plt.tight_layout()
		return fig

	except Exception as e:
		plt.close()
		raise RuntimeError(f"Error generating calibration curve: {str(e)}")


@validate_binary_inputs
def generate_roc_curve(
		y_true: Union[list, np.ndarray],
		y_proba: Union[list, np.ndarray],
		figsize: tuple = (5, 5),
		classifier_color: str = '#DA0081',
		baseline_color: str = '#200020'
) -> plt.Figure:
	"""
	Generate a Receiver Operating Characteristic (ROC) curve and plot it.

	Args:
		y_true: Array-like object containing the actual binary labels (0 or 1).
		y_proba: Array-like object containing the predicted probabilities.
		figsize: Tuple specifying figure dimensions. Defaults to (5, 5).
		classifier_color: Color for the ROC curve. Defaults to '#DA0081'.
		baseline_color: Color for the baseline curve. Defaults to '#200020'.

	Returns:
		matplotlib.figure.Figure: The generated ROC curve plot.

	Raises:
		ValueError: If input validation fails.
		RuntimeError: If curve generation or plotting fails.
	"""
	try:
		# Calculate the ROC curve and AUC
		fpr, tpr, _ = roc_curve(y_true, y_proba)
		roc_auc = auc(fpr, tpr)

		# Create figure and axis
		fig, ax = plt.subplots(figsize=figsize)

		# Plot ROC curve
		ax.plot(
				fpr,
				tpr,
				color=classifier_color,
				lw=2,
				label=f'Classifier (AUC = {roc_auc:.2f})'
		)

		# Plot baseline
		ax.plot(
				[0, 1],
				[0, 1],
				color=baseline_color,
				lw=2,
				linestyle='--',
				label='No Skill (AUC = 0.50)'
		)

		# Configure plot aesthetics
		ax.set_title('Receiver Operating Characteristic (ROC) Curve')
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_xlim([-0.05, 1.05])
		ax.set_ylim([-0.05, 1.05])
		ax.grid(True, alpha=0.3)
		ax.legend(loc='best')

		plt.tight_layout()
		return fig

	except Exception as e:
		plt.close()
		raise RuntimeError(f"Error generating ROC curve: {str(e)}")


@validate_binary_inputs
def generate_precision_recall_curve(
		y_true: Union[list, np.ndarray],
		y_proba: Union[list, np.ndarray],
		figsize: tuple = (5, 5),
		classifier_color: str = '#DA0081',
		baseline_color: str = '#200020'
) -> plt.Figure:
	"""
	Generate a Precision-Recall curve and plot it.

	Args:
		y_true: Array-like object containing the actual binary labels (0 or 1).
		y_proba: Array-like object containing the predicted probabilities.
		figsize: Tuple specifying figure dimensions. Defaults to (5, 5).
		classifier_color: Color for the PR curve. Defaults to '#DA0081'.
		baseline_color: Color for the baseline. Defaults to '#200020'.

	Returns:
		matplotlib.figure.Figure: The generated Precision-Recall curve plot.

	Raises:
		ValueError: If input validation fails.
		RuntimeError: If curve generation or plotting fails.
	"""
	try:
		# Calculate the Precision-Recall curve and AUC
		precision, recall, _ = precision_recall_curve(y_true, y_proba)
		pr_auc = auc(recall, precision)

		# Calculate no-skill baseline (proportion of positive class)
		no_skill = np.mean(y_true)

		# Create figure and axis
		fig, ax = plt.subplots(figsize=figsize)

		# Plot Precision-Recall curve
		ax.plot(
				recall,
				precision,
				color=classifier_color,
				lw=2,
				label=f'Classifier (PR AUC = {pr_auc:.2f})'
		)

		# Plot no-skill baseline
		ax.plot(
				[0, 1],
				[no_skill, no_skill],
				color=baseline_color,
				lw=2,
				linestyle='--',
				label=f'No Skill (PR AUC = {no_skill:.2f})'
		)

		# Configure plot aesthetics
		ax.set_title('Precision-Recall Curve')
		ax.set_xlabel('Recall')
		ax.set_ylabel('Precision')
		ax.set_xlim([-0.05, 1.05])
		ax.set_ylim([-0.05, 1.05])
		ax.grid(True, alpha=0.3)
		ax.legend(loc='best')

		plt.tight_layout()
		return fig

	except Exception as e:
		plt.close()
		raise RuntimeError(f"Error generating Precision-Recall curve: {str(e)}")


@validate_binary_inputs
def generate_ks_curve(
		y_true: Union[list, np.ndarray],
		y_proba: Union[list, np.ndarray],
		figsize: tuple = (5, 5),
		pos_class_color: str = '#5A9EFB',
		neg_class_color: str = '#DC410F',
		ks_line_color: str = '#200020'
) -> plt.Figure:
	"""
	Generate a Kolmogorov-Smirnov (KS) statistic plot.

	Args:
		y_true: Array-like object containing the actual binary labels (0 or 1).
		y_proba: Array-like object containing the predicted probabilities.
		figsize: Tuple specifying figure dimensions. Defaults to (5, 5).
		pos_class_color: Color for positive class curve. Defaults to '#5A9EFB'.
		neg_class_color: Color for negative class curve. Defaults to '#DC410F'.
		ks_line_color: Color for KS statistic line. Defaults to '#200020'.

	Returns:
		matplotlib.figure.Figure: The generated KS statistic plot.

	Raises:
		ValueError: If input validation fails.
		RuntimeError: If calculation or plotting fails.
	"""
	try:
		# Sort probabilities and corresponding true labels
		sorted_indices = np.argsort(y_proba)
		y_true_sorted = y_true[sorted_indices]
		y_proba_sorted = y_proba[sorted_indices]

		# Calculate cumulative distributions
		pos_sum = np.sum(y_true_sorted)
		neg_sum = len(y_true_sorted) - pos_sum

		if pos_sum == 0 or neg_sum == 0:
			raise ValueError("Both positive and negative classes must be present in y_true")

		cum_pos = np.cumsum(y_true_sorted) / pos_sum
		cum_neg = np.cumsum(1 - y_true_sorted) / neg_sum

		# Find the maximum difference (KS Statistic)
		ks_diff = np.abs(cum_pos - cum_neg)
		ks_statistic = np.max(ks_diff)
		ks_idx = np.argmax(ks_diff)
		ks_threshold = y_proba_sorted[ks_idx]

		# Create and configure plot
		fig, ax = plt.subplots(figsize=figsize)

		# Plot positive and negative class curves
		ax.plot(
				y_proba_sorted,
				cum_pos,
				color=pos_class_color,
				lw=2,
				label='Positive Class'
		)
		ax.plot(
				y_proba_sorted,
				cum_neg,
				color=neg_class_color,
				lw=2,
				label='Negative Class'
		)

		# Plot KS statistic line
		ax.plot(
				[ks_threshold, ks_threshold],
				[cum_pos[ks_idx], cum_neg[ks_idx]],
				color=ks_line_color,
				linestyle='--',
				lw=2,
				label=f'KS Statistic = {ks_statistic:.2f} at {ks_threshold:.2f}'
		)

		# Configure plot aesthetics
		ax.set_title('Kolmogorov-Smirnov (KS) Statistic Plot')
		ax.set_xlabel('Threshold')
		ax.set_ylabel('Percentage Below Threshold')
		ax.set_xlim([-0.05, 1.05])
		ax.set_ylim([-0.05, 1.05])
		ax.grid(True, alpha=0.3)
		ax.legend(loc='best')

		plt.tight_layout()
		return fig

	except Exception as e:
		plt.close()
		raise RuntimeError(f"Error generating KS curve: {str(e)}")

@validate_binary_inputs
def generate_gains_curve(
		y_true: Union[list, np.ndarray],
		y_proba: Union[list, np.ndarray],
		figsize: tuple = (5, 5),
		classifier_color: str = '#DA0081',
		baseline_color: str = '#200020',
		n_deciles: int = 10
) -> plt.Figure:
	"""
	Generate a Gains plot and plot it.

	Args:
		y_true: Array-like object containing the actual binary labels (0 or 1).
		y_proba: Array-like object containing the predicted probabilities.
		figsize: Tuple specifying figure dimensions. Defaults to (5, 5).
		classifier_color: Color for the classifier curve. Defaults to '#DA0081'.
		baseline_color: Color for the baseline curve. Defaults to '#200020'.
		n_deciles: Number of points to plot (deciles). Defaults to 10.

	Returns:
		matplotlib.figure.Figure: The generated Gains plot.

	Raises:
		ValueError: If input validation fails.
		RuntimeError: If calculation or plotting fails.
	"""
	try:
		# Sort the predicted probabilities and corresponding true values
		sorted_indices = np.argsort(y_proba)[::-1]
		y_true_sorted = y_true[sorted_indices]

		# Calculate the cumulative gains
		cumulative_true_positives = np.cumsum(y_true_sorted)
		total_true_positives = cumulative_true_positives[-1]

		if total_true_positives == 0:
			raise ValueError("No positive cases found in y_true")

		# Create arrays for deciles
		decile_indices = np.arange(1, n_deciles + 1) * (len(y_true) // n_deciles)
		cumulative_true_positive_rate = (
				cumulative_true_positives[decile_indices - 1] / total_true_positives
		)
		cumulative_population_rate = decile_indices / len(y_true)

		# Create and configure plot
		fig, ax = plt.subplots(figsize=figsize)

		# Plot classifier curve
		ax.plot(
				cumulative_population_rate,
				cumulative_true_positive_rate,
				color=classifier_color,
				lw=2,
				marker='o',
				label='Classifier'
		)

		# Plot baseline curve
		ax.plot(
				np.linspace(0.1, 1.0, n_deciles),
				np.linspace(0.1, 1.0, n_deciles),
				color=baseline_color,
				lw=2,
				linestyle='--',
				marker='o',
				label='No Skill'
		)

		# Configure plot aesthetics
		ax.set_title('Gains Curve')
		ax.set_xlabel('Percentage of Sample')
		ax.set_ylabel('Percentage of Positive Target')
		ax.set_xlim([0.05, 1.05])
		ax.set_ylim([0.05, 1.05])
		ax.grid(True, alpha=0.3)
		ax.legend(loc='best')

		plt.tight_layout()
		return fig

	except Exception as e:
		plt.close()
		raise RuntimeError(f"Error generating gains curve: {str(e)}")

@validate_binary_inputs
def generate_lift_curve(
		y_true: Union[list, np.ndarray],
		y_proba: Union[list, np.ndarray],
		figsize: tuple = (5, 5),
		classifier_color: str = '#DA0081',
		baseline_color: str = '#200020',
		n_deciles: int = 10,
		y_padding: float = 0.1
) -> plt.Figure:
	"""
	Generate a Lift curve and plot it.

	Args:
		y_true: Array-like object containing the actual binary labels (0 or 1).
		y_proba: Array-like object containing the predicted probabilities.
		figsize: Tuple specifying figure dimensions. Defaults to (5, 5).
		classifier_color: Color for the classifier curve. Defaults to '#DA0081'.
		baseline_color: Color for the baseline curve. Defaults to '#200020'.
		n_deciles: Number of points to plot (deciles). Defaults to 10.
		y_padding: Padding factor for y-axis upper limit. Defaults to 0.1.

	Returns:
		matplotlib.figure.Figure: The generated Lift curve.

	Raises:
		ValueError: If input validation fails.
		RuntimeError: If calculation or plotting fails.
	"""
	try:
		# Sort the predicted probabilities and corresponding true values
		sorted_indices = np.argsort(y_proba)[::-1]
		y_true_sorted = y_true[sorted_indices]

		# Calculate the cumulative gains
		cumulative_true_positives = np.cumsum(y_true_sorted)
		total_true_positives = cumulative_true_positives[-1]

		if total_true_positives == 0:
			raise ValueError("No positive cases found in y_true")

		# Create arrays for deciles
		decile_indices = np.arange(1, n_deciles + 1) * (len(y_true) // n_deciles)
		cumulative_true_positive_rate = (
				cumulative_true_positives[decile_indices - 1] / total_true_positives
		)
		cumulative_population_rate = decile_indices / len(y_true)

		# Calculate lift values
		lift_values = cumulative_true_positive_rate / cumulative_population_rate

		# Create and configure plot
		fig, ax = plt.subplots(figsize=figsize)

		# Plot classifier curve
		ax.plot(
				cumulative_population_rate,
				lift_values,
				color=classifier_color,
				lw=2,
				marker='o',
				label='Classifier'
		)

		# Plot baseline curve
		ax.plot(
				np.linspace(0.1, 1.0, n_deciles),
				np.ones(n_deciles),
				color=baseline_color,
				lw=2,
				linestyle='--',
				marker='o',
				label='No Skill'
		)

		# Configure plot aesthetics
		ax.set_title('Lift Curve')
		ax.set_xlabel('Percentage of Sample')
		ax.set_ylabel('Lift Value')
		ax.set_xlim([0.05, 1.05])
		ax.set_ylim([0.5, np.max(lift_values) * (1 + y_padding)])
		ax.grid(True, alpha=0.3)
		ax.legend(loc='best')

		plt.tight_layout()
		return fig

	except Exception as e:
		plt.close()
		raise RuntimeError(f"Error generating lift curve: {str(e)}")

@validate_binary_inputs
def generate_confusion_matrix(
		y_true: Union[list, np.ndarray],
		y_proba: Union[list, np.ndarray],
		threshold: float,
		figsize: tuple = (5, 5),
		cmap: str = 'RdPu',
		font_size: int = 10
) -> plt.Figure:
	"""
	Display a confusion matrix for a binary classifier.

	Args:
		y_true: Array-like object containing the actual binary labels (0 or 1).
		y_proba: Array-like object containing the predicted probabilities.
		threshold: Threshold for binary classification.
		figsize: Tuple specifying figure dimensions. Defaults to (5, 5).
		cmap: Colormap for the heatmap. Defaults to 'RdPu'.
		font_size: Font size for annotations. Defaults to 10.

	Returns:
		matplotlib.figure.Figure: The confusion matrix plot.

	Raises:
		ValueError: If input validation fails.
		RuntimeError: If matrix generation or plotting fails.
	"""
	try:
		# Compute the confusion matrix
		y_pred = (y_proba >= threshold).astype(int)
		cm = confusion_matrix(y_true, y_pred)

		# Define matrix elements
		group_names = [
			'True Negative',
			'False Positive',
			'False Negative',
			'True Positive'
		]
		group_counts = cm.flatten()

		# Compute percentages
		group_percentages = [
			f'{value:.2%}'
			for value in normalize(cm, axis=1, norm='l1').flatten()
		]

		# Create and format labels
		labels = [
			f'{name}\n{count:,}\n({percentage})'
			for name, count, percentage in zip(
					group_names,
					group_counts,
					group_percentages
			)
		]
		labels = np.asarray(labels).reshape(2, 2)

		# Create and configure plot
		fig, ax = plt.subplots(figsize=figsize)

		# Plot heatmap
		sns.heatmap(
				cm,
				annot=labels,
				fmt='',
				cmap=cmap,
				cbar=False,
				robust=True,
				linewidths=0.5,
				square=True,
				ax=ax,
				annot_kws={'size': font_size}
		)

		# Configure plot aesthetics
		ax.set_title(
				f'Confusion Matrix\n(Threshold: {threshold:.2f})',
				pad=10
		)
		ax.set_xlabel('Predicted Class')
		ax.set_ylabel('Actual Class')
		ax.set_yticklabels(
				ax.get_yticklabels(),
				rotation=0,
				va='center'
		)

		plt.tight_layout()
		return fig

	except Exception as e:
		plt.close()
		raise RuntimeError(f"Error generating confusion matrix: {str(e)}")
