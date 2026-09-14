# GenerativeMultimodalAI

Repositorio que reúne el desarrollo práctico y experimental de los laboratorios del curso **CC5219 — Inteligencia Artificial Generativa y Multimodal**, impartido en el **Departamento de Ciencias de la Computación (DCC) de la Universidad de Chile**.

* **Docente**: Valentin Barriere:[(@valbarriere)](https://github.com/valbarriere)
* **Repositorio original del curso**: https://github.com/valbarriere/CC5219-IA-Generativa-MModal/tree/master

## Contexto y Objetivos del Curso

El curso ofrece una introducción conceptual y aplicada a los fundamentos de la inteligencia artificial generativa y el modelado multimodal moderno. A través de un enfoque que combina teoría y experimentación directa, se estudia cómo los modelos generativos actuales son capaces de crear, transformar, proyectar y comprender distintas modalidades de datos:

* **Modelos Generativos de Visión**: Redes Generativas Antagónicas (GANs), Autoencoders Variacionales (VAEs), Modelos de Difusión, inversión en el espacio latente y edición semántica de imágenes.
* **Procesamiento y Modelado de Audio**: Representaciones acústicas, arquitecturas basadas en Transformers para audio (HuBERT, DistilHuBERT) y aprendizaje autosupervisado sobre señales continuas.
* **Modelos de Lenguaje y Multimodalidad**: Integración de visión, lenguaje y audio mediante representaciones conjuntas y modelos multimodales avanzados.

Los laboratorios aquí desarrollados permiten experimentar con herramientas líderes del ecosistema (`PyTorch`, `Transformers`, `Hugging Face`, etc.) para adaptar, entrenar e interactuar con estos modelos generativos.

---

# Laboratorio 1: Modelos Generativos de Imágenes

Este laboratorio se enfoca en la exploración y experimentación con modelos generativos para imágenes basados en **StyleGAN** e inversión en el espacio latente mediante la arquitectura **ReStyle pSp** (*pixel2style2pixel*).

El objetivo central es la aplicación de la técnica de **Toonificación** (*Toonification*), transformando fotografías reales de rostros humanos al estilo de animación y caricatura 3D. El flujo de trabajo aborda:
1. **Alineación y preprocesamiento** facial de la imagen de entrada.
2. **Inversión iterativa** en el espacio latente $W^+$ de StyleGAN mediante el codificador ReStyle pSp con red troncal piramidal (*Feature Pyramid Network*).
3. **Inferencia y síntesis** del rostro toonificado a través de múltiples iteraciones de refinamiento residual.
4. **Manipulación semántica en el espacio latente**: edición controlada de atributos faciales (edad y género) desplazando el vector latente obtenido a lo largo de direcciones precalculadas antes de decodificarlo con el generador.

### Resultados obtenidos en el Laboratorio 1

#### 1. Inferencia Iterativa de Toonificación (`toonify_results_10.jpg`)
![Resultados Toonify 10 Iteraciones](src/figures/toonify_results_10.jpg)

* **¿Cómo se obtuvo?**: Se obtuvo mediante la ejecución del codificador **ReStyle pSp** (`restyle_psp_toonify.pt`) configurado para realizar 10 iteraciones de refinamiento (`n_iters_per_batch = 10`). En cada paso, el modelo predice correcciones residuales sobre el código latente en el espacio extendido $W^+$, mejorando iterativamente la calidad visual y fidelidad del estilo. La imagen resultante muestra horizontalmente la evolución del rostro a través de los 10 pasos de inferencia hasta converger en la versión toonificada final acoplada con la imagen de entrada.

#### 2. Edición Semántica en el Espacio Latente (`toonify_results_alterated.jpg`)
![Resultados Toonify Alterados](src/figures/toonify_results_alterated.jpg)

* **¿Cómo se obtuvo?**: Se obtuvo a partir del vector latente final generado tras las 10 iteraciones del proceso de inversión. Se aplicó una manipulación vectorial en el espacio latente sumando desplazamientos controlados por un factor $\alpha$:

  $\mathbf{w}_{\text{editado}} = \mathbf{w}_{\text{final}} + \alpha \cdot \mathbf{v}_{\text{dirección}}$
  
  donde $\mathbf{v}_{\text{dirección}}$ corresponde a vectores de dirección semántica precalculados para StyleGAN2 (`direction_age` y `direction_gender`). Luego, los vectores resultantes fueron decodificados por la red generadora para sintetizar las nuevas imágenes. La figura compara la imagen original frente a las variaciones generadas con alteración de edad (`Age`) y alteración de género (`Gender`).

---

# Laboratorio 2: Modelos Generativos de Audio

Este laboratorio aborda el estudio, análisis y aplicación práctica de arquitecturas basadas en **Transformers para el procesamiento y modelado de señales de audio**.

El contenido se estructura en dos componentes principales:
1. **Fundamentos y Arquitecturas de Audio Transformers**:
   - Comprensión de la representación de datos de audio desde señales crudas (*raw waveforms*) hasta características de espectrogramas y parches tiempo-frecuencia.
   - Estudio detallado del funcionamiento de **HuBERT** (*Hidden-Unit BERT*), explorando el descubrimiento de unidades acústicas discretas (*pseudo-labels*) mediante clustering k-means y el entrenamiento autosupervisado con predicción enmascarada (*Masked Prediction*).
2. **Sesión Práctica de Fine-Tuning para Clasificación Musical**:
   - Adaptación y *fine-tuning* del modelo preentrenado **DistilHuBERT** sobre el conjunto de datos **GTZAN** para la tarea de clasificación de géneros musicales (*Music Genre Classification*).
   - Pipeline completo de ingeniería de datos usando `AutoFeatureExtractor`, configuración de hiperparámetros de entrenamiento con Hugging Face `Trainer`.
   - Evaluación del rendimiento, análisis de errores e interpretabilidad mediante matrices de confusión, visualización de representaciones latentes e inferencia con archivos de audio propios.
