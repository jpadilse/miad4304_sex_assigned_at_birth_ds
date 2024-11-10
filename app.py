import pandas as pd
import streamlit as st
import tensorflow as tf

# Configurar la página
st.set_page_config(
		page_title="Predicción de Género por Nombre",
		page_icon="👤",
		layout="centered"
)

# Título principal
st.title("🔮 Predicción de Género por Nombre")
st.write("Introduce un nombre para predecir su género más probable")

# Añadir un campo de texto para el nombre
nombre_input = st.text_input("Introduce un nombre:", "")

# Cargar el modelo fuera de la condicional para evitar cargarlo múltiples veces
modelo_archivo = "./models/model.keras"

try:
	# Cargar el modelo
	modelo = tf.keras.models.load_model(modelo_archivo)

	if nombre_input:
		# Crear un DataFrame o tensor para la entrada
		datos_entrada = pd.Series([nombre_input])
		tensor_entrada = tf.convert_to_tensor(datos_entrada)

		# Realizar la predicción
		prediccion = modelo.predict(tensor_entrada)

		# Extraer el valor escalar del array de NumPy
		valor_prediccion = float(prediccion[0][0])

		# Crear columnas para la visualización
		col1, col2 = st.columns(2)

		# Mostrar la predicción con confianza
		with col1:
			st.markdown("### Género Predicho")
			if valor_prediccion >= 0.5:
				st.markdown(f"### 👨 Hombre")
				confianza = valor_prediccion * 100
			else:
				st.markdown(f"### 👩 Mujer")
				confianza = (1 - valor_prediccion) * 100

		with col2:
			st.markdown("### Confianza")
			st.markdown(f"### {confianza:.1f}%")

		# Añadir una barra de progreso de confianza (entre 0 y 1)
		st.progress(valor_prediccion if valor_prediccion >= 0.5 else 1 - valor_prediccion)

		# Añadir explicación
		st.markdown("---")
		st.markdown("### Cómo funciona")
		st.write("""
        Este modelo analiza los patrones en los caracteres del nombre para hacer su predicción.
        - Valores más cercanos a 1 indican nombres masculinos
        - Valores más cercanos a 0 indican nombres femeninos
        - La confianza muestra cuán seguro está el modelo acerca de su predicción
        """)

except Exception as e:
	st.error(f"Error al procesar los archivos: {str(e)}")

# Añadir pie de página con descargo de responsabilidad
st.markdown("---")
st.markdown("""
    *Nota: Este es un modelo de aprendizaje automático y las predicciones están basadas en los datos de entrenamiento. 
    Los resultados deben tomarse como predicciones estadísticas más que como respuestas definitivas.*
""")
