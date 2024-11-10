"""
This module provides different plots to evaluate the performance of a binary
classification model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
	auc,
	brier_score_loss,
	confusion_matrix,
	precision_recall_curve,
	roc_curve
)
from sklearn.preprocessing import normalize


def generate_calibration_curve(y_true, y_proba, n_bins=5):
	"""
	Generate a Calibration Curve and plot it.

	Args:
		y_true (list, numpy array, or pandas Series): An array-like object containing the actual binary labels.
		y_proba (list, numpy array, or pandas Series): An array-like object containing the predicted probabilities.
		n_bins (int): Number of bins to discretize the [0, 1] interval into for the calibration curve.

	Returns:
		matplotlib.figure.Figure: The generated Calibration curve plot.
	"""
	# Calculate the calibration curve
	prob_true, prob_prediction = calibration_curve(y_true, y_proba, n_bins=n_bins)
	bl_statistic = brier_score_loss(y_true, y_proba)

	# Create the plot
	fig, ax = plt.subplots(figsize=(5, 5))
	fig.suptitle('Calibration Curve')
	ax.plot(
			prob_prediction,
			prob_true,
			color='#DA0081',
			lw=2,
			label=f'Classifier (Brier Score = {bl_statistic:.2f})'
	)
	ax.plot(
			[0, 1],
			[0, 1],
			color='#200020',
			lw=2,
			linestyle='--',
			label='Perfectly (Brier Score = 0.00)'
	)
	ax.set_xlabel('Mean Predicted Probability')
	ax.set_ylabel('Fraction of Positives')
	ax.set_xlim([-0.05, 1.05])
	ax.set_ylim([-0.05, 1.05])
	ax.legend(loc='best')
	plt.show()

	return fig


def generate_roc_curve(y_true, y_proba):
	"""
	Generate a Receiver Operating Characteristic (ROC) curve and plot it.

	Args:
		y_true (list, numpy array, or pandas Series): An array-like object containing the actual binary labels.
		y_proba (list, numpy array, or pandas Series): An array-like object containing the predicted binary labels.

	Returns:
		matplotlib.figure.Figure: The generated ROC curve plot.
	"""

	# Calculate the ROC curve and AUC
	fpr, tpr, _ = roc_curve(y_true, y_proba)
	roc_auc = auc(fpr, tpr)

	# Create the plot
	fig, ax = plt.subplots(figsize=(5, 5))
	fig.suptitle('Receiver Operating Characteristic (ROC) Curve')
	ax.plot(
			fpr,
			tpr,
			color='#DA0081',
			lw=2,
			label=f'Classifier (AUC = {roc_auc:.2f})'
	)
	ax.plot(
			[0, 1],
			[0, 1],
			color='#200020',
			lw=2,
			linestyle='--',
			label='No Skill (AUC = 0.50)'
	)
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_xlim([-0.05, 1.05])
	ax.set_ylim([-0.05, 1.05])
	ax.legend(loc='best')
	plt.show()

	return fig


def generate_precision_recall_curve(y_true, y_proba):
	"""
	Generate a Precision-Recall curve and plot it.

	Args:
		y_true (list, numpy array, or pandas Series): An array-like object containing the actual binary labels.
		y_proba (list, numpy array, or pandas Series): An array-like object containing the predicted binary labels.

	Returns:
		matplotlib.figure.Figure: The generated Precision-Recall curve plot.
	"""

	# Calculate the Precision-Recall curve and AUC
	precision, recall, _ = precision_recall_curve(y_true, y_proba)
	pr_auc = auc(recall, precision)

	no_skill = len(y_true[y_true == 1]) / len(y_true)

	# Create the plot
	fig, ax = plt.subplots(figsize=(5, 5))
	fig.suptitle('Precision-Recall Curve')
	ax.plot(
			recall,
			precision,
			color='#DA0081',
			lw=2,
			label=f'Classifier (PR AUC = {pr_auc:.2f})'
	)
	ax.plot(
			[0, 1],
			[no_skill, no_skill],
			color='#200020',
			lw=2,
			linestyle='--',
			label=f'No Skill (PR AUC = {no_skill:.2f})'
	)
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_xlim([-0.05, 1.05])
	ax.set_ylim([-0.05, 1.05])
	ax.legend(loc='best')
	plt.show()

	return fig


def generate_ks_plot(y_true, y_proba):
	"""
	Generate a Kolmogorov-Smirnov (KS) statistic plot.

	Args:
		y_true (list, numpy array, or pandas Series): An array-like object containing the actual binary labels.
		y_proba (list, numpy array, or pandas Series): An array-like object containing the predicted probabilities.

	Returns:
		matplotlib.figure.Figure: The generated KS statistic plot.
	"""
	# Ensure y_true and y_probas are numpy arrays
	y_true = np.array(y_true)
	y_probas = np.array(y_proba)

	# Sort the probabilities and corresponding true labels
	sorted_indices = np.argsort(y_probas)
	y_true_sorted = y_true[sorted_indices]
	y_probas_sorted = y_probas[sorted_indices]

	# Calculate cumulative distributions
	cum_pos = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
	cum_neg = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)

	# Find the maximum difference (KS Statistic)
	ks_statistic = np.max(np.abs(cum_pos - cum_neg))
	ks_statistic_index = np.argmax(np.abs(cum_pos - cum_neg))
	ks_x = y_probas_sorted[ks_statistic_index]

	# Create the plot
	fig, ax = plt.subplots(figsize=(5, 5))
	fig.suptitle('Kolmogorov-Smirnov (KS) Statistic Plot')
	ax.plot(
			y_probas_sorted,
			cum_pos,
			color='#5A9EFB',
			lw=2,
			label='Positive Class'
	)
	ax.plot(
			y_probas_sorted,
			cum_neg,
			color='#DC410F',
			lw=2,
			label='Negative Class'
	)
	ax.plot(
			[ks_x, ks_x],
			[cum_pos[ks_statistic_index], cum_neg[ks_statistic_index]],
			color='#200020',
			linestyle='--',
			lw=2,
			label=f'KS Statistic = {ks_statistic:.2f} at {ks_x:.2f}'
	)
	ax.set_xlabel('Threshold')
	ax.set_ylabel('Percentage Below Threshold')
	ax.set_xlim([-0.05, 1.05])
	ax.set_ylim([-0.05, 1.05])
	ax.legend(loc='best')
	plt.show()

	return fig


def generate_gains_plot(y_true, y_proba):
	"""
	Generate a Gains plot and plot it.

	Args:
		y_true (list, numpy array, or pandas Series): An array-like object containing the actual binary labels.
		y_proba (list, numpy array, or pandas Series): An array-like object containing the predicted probabilities.

	Returns:
		matplotlib.figure.Figure: The generated Gains plot.
	"""

	# Sort the predicted probabilities and corresponding true values
	sorted_indices = np.argsort(y_proba)[::-1]
	y_true_sorted = np.array(y_true)[sorted_indices]

	# Calculate the cumulative gains
	cumulative_true_positives = np.cumsum(y_true_sorted)
	total_true_positives = cumulative_true_positives[-1]

	# Create arrays for deciles
	decile_indices = np.arange(1, 10 + 1) * (len(y_true) // 10)
	cumulative_true_positive_rate_deciles = cumulative_true_positives[decile_indices - 1] / total_true_positives
	cumulative_population_rate_deciles = decile_indices / len(y_true)

	# Create the plot
	fig, ax = plt.subplots(figsize=(5, 5))
	fig.suptitle('Gain Curve')
	ax.plot(
			cumulative_population_rate_deciles,
			cumulative_true_positive_rate_deciles,
			color='#DA0081',
			lw=2,
			marker='o',
			label='Classifier'
	)
	ax.plot(
			np.arange(0.1, 1.1, 0.1),
			np.arange(0.1, 1.1, 0.1),
			color='#200020',
			lw=2,
			linestyle='--',
			marker='o',
			label='No Skill'
	)
	ax.set_xlabel('Percentage of Sample')
	ax.set_ylabel('Percentage of Positive Target')
	ax.set_xlim([0.05, 1.05])
	ax.set_ylim([0.05, 1.05])
	ax.legend(loc='best')
	plt.show()

	return fig


def generate_lift_curve(y_true, y_proba):
	"""
	Generate a Lift curve and plot it.

	Args:
		y_true (list, numpy array, or pandas Series): An array-like object containing the actual binary labels.
		y_proba (list, numpy array, or pandas Series): An array-like object containing the predicted probabilities.

	Returns:
		matplotlib.figure.Figure: The generated Lift curve.
	"""

	# Sort the predicted probabilities and corresponding true values
	sorted_indices = np.argsort(y_proba)[::-1]
	y_true_sorted = np.array(y_true)[sorted_indices]

	# Calculate the cumulative gains
	cumulative_true_positives = np.cumsum(y_true_sorted)
	total_true_positives = cumulative_true_positives[-1]

	# Create arrays for deciles
	decile_indices = np.arange(1, 10 + 1) * (len(y_true) // 10)
	cumulative_true_positive_rate_deciles = cumulative_true_positives[decile_indices - 1] / total_true_positives
	cumulative_population_rate_deciles = decile_indices / len(y_true)

	# Calculate lift values
	lift_values = cumulative_true_positive_rate_deciles / cumulative_population_rate_deciles

	# Create the plot
	fig, ax = plt.subplots(figsize=(5, 5))
	fig.suptitle('Lift Curve')
	ax.plot(
			cumulative_population_rate_deciles,
			lift_values,
			color='#DA0081',
			lw=2,
			marker='o',
			label='Classifier'
	)
	ax.plot(
			np.arange(0.1, 1.1, 0.1),
			np.ones(10),
			color='#200020',
			lw=2,
			linestyle='--',
			marker='o',
			label='No Skill'
	)
	ax.set_xlabel('Percentage of Sample')
	ax.set_ylabel('Lift Value')
	ax.set_xlim([0.05, 1.05])
	ax.set_ylim([0.5, np.max(lift_values) * 1.1])
	ax.legend(loc='best')
	plt.show()

	return fig


def generate_confusion_matrix(y_true, y_proba, threshold: float):
	"""
	Display a confusion matrix for a binary classifier.

	Args:
		y_true (list, numpy array, or pandas Series): An array-like object
			containing the actual binary labels.
		y_proba (list, numpy array, or pandas Series): An array-like object
			containing the predicted binary labels.
		threshold (float, optional): Threshold for binary classification.

	Returns:
		matplotlib.figure.Figure: The matplotlib figure object containing
			the confusion matrix plot.
	"""

	# Compute the confusion matrix
	cm = confusion_matrix(y_true, (y_proba >= threshold).astype(int))

	# Define the group names for the confusion matrix
	group_names = [
		'True Negative',
		'False Positive',
		'False Negative',
		'True Positive'
	]

	# Flatten the confusion matrix to get the group counts
	group_counts = cm.flatten()

	# Compute the group percentages for the confusion matrix
	group_percentages = [
		f'{value:.2%}'
		for value in normalize(cm, axis=1, norm='l1').flatten()
	]

	# Create the labels for the confusion matrix
	labels = [
		f'{v1}\n{v2:,}\n({v3})'
		for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
	]
	labels = np.asarray(labels).reshape(2, 2)

	# Create the figure and axes for the confusion matrix plot
	fig, axes = plt.subplots(figsize=(5, 5))
	fig.suptitle('Confusion Matrix')
	sns.heatmap(
			cm,
			annot=labels,
			fmt='',
			cmap='RdPu',
			cbar=False,
			robust=True,
			linewidths=0.5,
			square=True,
			ax=axes
	)
	axes.set_xlabel('Predicted Class')
	axes.set_ylabel('Actual Class')
	axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
	plt.tight_layout()
	plt.show()

	return fig
