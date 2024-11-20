import pandas as pd
import streamlit as st
import tensorflow as tf
from typing import Optional, Tuple
from pathlib import Path

from loguru import logger

# Constants
MODEL_PATH = Path("./models/model.keras")
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 50

# Custom CSS
CUSTOM_CSS = """
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
    }
    .stProgress {
        height: 20px;
    }
    </style>
"""

@st.cache_resource
def load_model(model_path: Path) -> Optional[tf.keras.Model]:
	"""Load the Keras model with error handling."""
	try:
		model = tf.keras.models.load_model(str(model_path))
		logger.info("Model loaded successfully")
		return model
	except Exception as e:
		logger.error(f"Error loading model: {e}")
		return None

def predict_gender(name: str, model: tf.keras.Model) -> Optional[float]:
	"""Generate gender prediction and confidence from a name."""
	if not model or not name:
		return None

	try:
		input_tensor = tf.convert_to_tensor(pd.Series([name]))
		prediction = model.predict(input_tensor, verbose=0)
		return float(prediction[0][0])
	except Exception as e:
		logger.error(f"Prediction error: {e}")
		return None

def validate_name(name: str) -> Tuple[bool, str]:
	"""Validate the input name."""
	if not name.strip():
		return False, "Por favor, introduce un nombre."
	if len(name) < MIN_NAME_LENGTH:
		return False, f"El nombre debe tener al menos {MIN_NAME_LENGTH} caracteres."
	if len(name) > MAX_NAME_LENGTH:
		return False, f"El nombre no debe exceder {MAX_NAME_LENGTH} caracteres."
	if not name.replace(" ", "").isalpha():
		return False, "El nombre solo debe contener letras."
	return True, ""

def display_prediction_results(prediction: float):
	"""Display prediction results with styled components."""
	# Create columns for results
	col1, col2 = st.columns(2)

	# Calculate confidence
	is_male = prediction >= 0.5
	confidence = prediction * 100 if is_male else (1 - prediction) * 100
	gender_icon = "👨" if is_male else "👩"
	gender_text = "Hombre" if is_male else "Mujer"

	# Display results with styling
	with col1:
		st.markdown(f"### Género Predicho")
		st.markdown(f"### {gender_icon} {gender_text}")

	with col2:
		st.markdown("### Nivel de Confianza")
		st.markdown(f"### {confidence:.1f}%")

	# Confidence bar with color
	bar_color = "blue" if is_male else "pink"
	st.markdown(
			f"""
			<style>
			.stProgress > div > div > div > div {{
				background-color: {bar_color};
			}}
			</style>
			""",
			unsafe_allow_html=True
	)
	st.progress(prediction if is_male else 1 - prediction)

def show_model_explanation():
	"""Display the model explanation section."""
	with st.expander("📖 ¿Cómo funciona?"):
		st.write(
			"""
			Este modelo utiliza redes neuronales para analizar patrones en los caracteres del nombre:
			- Analiza la secuencia de letras y patrones comunes en nombres
			- Compara con patrones aprendidos de miles de nombres etiquetados
			- Genera una predicción basada en similitudes estadísticas
			- La confianza indica qué tan seguro está el modelo de su predicción
			"""
		)

def show_footer():
	"""Display the footer section."""
	st.markdown("---")
	st.markdown(
			"""
			<div style='text-align: center; color: #666;'>
			<p><em>⚠️ Nota: Este modelo utiliza aprendizaje automático y sus predicciones están basadas en datos históricos. 
			Los resultados son estimaciones estadísticas y no deben considerarse como definitivos.</em></p>
			<p>Versión 1.0 | Desarrollado️ usando Streamlit</p>
			</div>
			""",
			unsafe_allow_html=True
	)

def main():
	# Page configuration
	st.set_page_config(
			page_title="Predicción de Género por Nombre",
			page_icon="👤",
			layout="centered",
	)

	# Apply custom CSS
	st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

	# Header
	st.title("🔮 Predicción de Género por Nombre")
	st.markdown("### Descubre el género más probable asociado a un nombre")

	# Load model
	model = load_model(MODEL_PATH)

	if not model:
		st.error("Error: No se pudo cargar el modelo. Por favor, inténtalo más tarde.")
		return

	# Input section
	name_input = st.text_input(
			"Introduce un nombre:",
			placeholder="Ejemplo: María",
			help="Ingresa un nombre para predecir su género"
	)

	if name_input:
		# Validate input
		is_valid, error_message = validate_name(name_input)

		if not is_valid:
			st.error(error_message)
			return

		# Make prediction
		prediction = predict_gender(name_input, model)

		if prediction is not None:
			display_prediction_results(prediction)
			show_model_explanation()
		else:
			st.error("No se pudo realizar la predicción. Por favor, inténtalo de nuevo.")

	show_footer()

if __name__ == "__main__":
	main()