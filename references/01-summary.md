# Proyecto de Predicción de Sexo por Nombre en Neobanco

## 1. Problema y Contexto

El proyecto aborda la necesidad de identificar el sexo al nacer de los 
usuarios de un neobanco en Colombia a partir de sus nombres
registrados en la aplicación. Este es un desafío común en instituciones
financieras donde la información demográfica precisa es crucial para 
diversos análisis y servicios.

### 1.1 Situación Actual

**Necesidades del Neobanco:**
- Personalización de productos financieros
- Análisis de segmentación de mercado
- Cumplimiento regulatorio
- Estrategias de marketing dirigido
- Análisis de inclusión financiera por género

### 1.2 Desafíos Actuales
- Datos incompletos en el registro de usuarios
- Necesidad de automatización para grandes volúmenes
- Requisitos de protección de datos
- Variabilidad en la escritura de nombres
- Complejidad de nombres compuestos o poco comunes

## 2. Pregunta de Negocio y Alcance

### 2.1 Preguntas Principales
1. ¿Cómo predecir el género de manera automatizada y precisa?
2. ¿Qué nivel de confianza podemos obtener en las predicciones?
3. ¿Cómo integrar esta solución en los procesos existentes?

### 2.2 Alcance del Proyecto

#### Desarrollo Técnico
- Implementación de modelo ML como MVP
- Integración con tablero descriptivo
- Desarrollo de API para consumo interno
- Implementación de pipelines automatizados

#### Entregables

1. **Tablero de Predicción por Usuario**
   - Interfaz interactiva
   - Procesamiento por lotes
   - Métricas de confianza
   - Registro de auditoría

2. **Informe Audiovisual del Proyecto**
   - Metodología utilizada
   - Resultados y métricas
   - Recomendaciones
   - Limitaciones

### 2.3 Cronograma

**Octubre 2024**
- Recopilación y preparación de datos
- Desarrollo del modelo inicial
- Pruebas preliminares

**Noviembre 2024**
- Refinamiento del modelo
- Desarrollo de interfaz
- Documentación
- Entrega final

## 3. Conjuntos de Datos

Los datos a emplear tienen las siguientes características:

- Fuente: Registros de usuarios del neobanco
- Volumen: Se utilizará una muestra aleatoria de aproximadamente 100,000 registros
- Tipo de datos:
  * Identificación anonimizada
  * Nombres de usuarios
  * Sexo al nacer

- Formato: Texto almacenado en AWS (S3)
- Características del manejo de datos:
  * Serán anonimizados para proteger la identidad de los usuarios
  * Se aplicará normalización de nombres para estandarización
  * Se implementará un sistema de control de versiones

## 4. Exploración de los datos

- Estructura de los datos: El conjunto de datos contiene 113,738 filas y 4 columnas:
  - id: Identificador único (todos los registros son únicos y sin valores faltantes).
  - first_name: Primer nombre (sin valores faltantes).
  - second_name: Segundo nombre, con 60.014 registros faltantes.
  - sex: Codificado como 0 (mujeres) o 1 (hombres), con una media cercana a 0.41 (indicando que aproximadamente el 
    41% de los registros 
    están codificados como 1). No tiene valores faltantes.

- Distribución de los datos:
  - El primer y segundo nombre en su mayoría tienen entre cinco y ocho caracteres. No obstante, el primer nombre 
    tiene valores atípicos con hasta 35 caracteres
  - Los nombres más comunes son: Antonio, John, Roberto y Julian
  - No hay diferencias marcas en la longitud de los nombres entre hombres y mujeres

## 5. Tablero

Este panel incluye las siguientes características:

1. Un formulario para ingresar un primer nombre y obtener una predicción de género.
2. Un área de visualización para el resultado de la predicción, que muestra el género predicho y la probabilidad.
3. Un gráfico de línea simple que muestra la precisión del modelo a lo largo del tiempo.
4. Una tabla con ejemplos de predicciones para algunos nombres comunes.

El componente utiliza una llamada a una API para generar las predicciones usando un modelo de aprendizaje profundo. 

El panel es adaptable (responsive) y utiliza Tailwind CSS para el estilo. Aprovecha los componentes para un aspecto consistente y moderno.

## 6. Reporte de Trabajo en Equipo

- **Investigador Principal:** Juan Felipe Padilla Sepulveda
  - Encargado de crear el proyecto e inicializar Git y DVC. Asimismo, crear los primeros *commits* y *pull requests*.
- **Administradores de Datos:**
  - Mauricio Gonzalez Caro
    - Encargado de realizar el análisis exploratorio de la base original
  - Andres Quiñones Ortiz
    - Encargado de realizar el prototipo del tablero
- **Administrador Proyecto:** Eva Karina Díaz Gavalo
  - Encargada de redactar y consolidar el contexto, preguntas de negocio y alcances del proyecto