"""Script that serves for custom tensorflow layers"""

import tensorflow as tf

from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class OneHotLayer(Layer):
	"""Custom layer for one-hot encoding with fixed vocabulary size."""

	def __init__(self, vocab_size: int, **kwargs):
		super().__init__(**kwargs)
		self.vocab_size = vocab_size

	def call(self, inputs: tf.Tensor) -> tf.Tensor:
		return tf.one_hot(inputs, self.vocab_size)

	def get_config(self) -> dict:
		config = super().get_config()
		config.update({"vocab_size": self.vocab_size})
		return config
