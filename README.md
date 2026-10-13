# Image Processing Lab

### Sistema de Procesamiento Digital de Imágenes en Tiempo Real

Aplicación desarrollada en Python que captura imágenes desde una cámara web y aplica técnicas de procesamiento digital de imágenes para analizar las condiciones de iluminación, el contraste y el ruido, así como realizar una inspección básica de objetos mediante operaciones sobre píxeles.

El sistema utiliza OpenCV y NumPy para implementar filtros espaciales, analizar histogramas, segmentar imágenes y visualizar los resultados mediante una interfaz gráfica con indicadores de estado.

## Funcionalidades

- **Captura en tiempo real:** adquisición de imágenes mediante una cámara web.
- **Análisis de iluminación:** clasificación de las condiciones de iluminación en insuficiente, adecuada, sobreexpuesta o con contraluz.
- **Análisis de histogramas:** estudio de la distribución de los niveles de intensidad de los píxeles.
- **Evaluación del contraste:** cálculo de estadísticas para estimar el contraste de la imagen.
- **Estimación de ruido:** análisis de las diferencias entre la imagen original y su versión suavizada.
- **Filtros espaciales:** implementación manual de filtros de suavizado de 3 × 3 y 5 × 5.
- **Operaciones sobre píxeles:** transformación negativa de imágenes en escala de grises.
- **Umbralización:** segmentación de imágenes mediante el método de Otsu.
- **Procesamiento de vecindades:** aplicación de un filtro de promedio de 3 × 3 sobre la máscara de segmentación.
- **Conteo de píxeles:** cálculo del área y del porcentaje de imagen ocupado por el objeto segmentado.
- **Calibración por referencia:** comparación con un fondo y un porcentaje de área esperado.
- **Visualización de resultados:** presentación de histogramas, máscaras, estadísticas e indicadores tipo semáforo.

## Tecnologías utilizadas

- **Python:** implementación de la lógica y los algoritmos de procesamiento.
- **OpenCV:** captura de video, transformación de imágenes y visualización.
- **NumPy:** operaciones matriciales y cálculos numéricos.

## Requisitos

- Python 3.
- Una cámara web disponible.
- Las dependencias del proyecto instaladas.

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd image-processing-lab
```

Reemplaza `URL_DEL_REPOSITORIO` por la dirección real de tu repositorio.

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

En Windows, activa el entorno virtual:

```powershell
.venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
python main.py
```

La aplicación abrirá la cámara disponible y mostrará los resultados del procesamiento en tiempo real.

## Controles de la aplicación

| Tecla | Función |
|---|---|
| `q` o `Esc` | Cerrar la aplicación. |
| `n` | Alternar entre la imagen original y la simulación de ruido de sal y pimienta. |
| `b` | Registrar el fondo y el objeto de referencia. Una tercera pulsación reinicia la calibración. |
| `r` | Borrar la referencia y volver al modo de umbralización de Otsu. |
| `v` | Verificar los filtros manuales y comparar sus resultados con OpenCV. |
| `s` | Guardar una captura de la interfaz. |

## Flujo de procesamiento

1. Captura de imágenes mediante la cámara.
2. Redimensionamiento y conversión a escala de grises.
3. Cálculo del histograma y análisis de iluminación.
4. Aplicación de un filtro lineal de suavizado de 5 × 5.
5. Estimación del ruido mediante diferencias de intensidad.
6. Segmentación por comparación con un fondo o mediante umbralización de Otsu.
7. Aplicación de un filtro de promedio de 3 × 3 sobre la máscara.
8. Conteo de píxeles y cálculo del área relativa del objeto.
9. Evaluación de las condiciones de la imagen y del objeto.
10. Presentación de los resultados y estadísticas en la interfaz.

## Estructura del proyecto

```text
image-processing-lab/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore

```

## Alcance y limitaciones

Este proyecto es una implementación académica de técnicas clásicas de procesamiento digital de imágenes. Los resultados dependen de los umbrales configurados, las condiciones de iluminación, el fondo y la posición del objeto.

La inspección utiliza segmentación, comparación con referencias y análisis del área ocupada por los píxeles. No implementa un modelo de aprendizaje automático para reconocer categorías generales de objetos.

## Objetivo académico

Aplicar conceptos de procesamiento digital de imágenes mediante la implementación de operaciones sobre píxeles, histogramas, filtros espaciales, umbralización y análisis estadístico, integrándolos en una aplicación que procesa imágenes capturadas en tiempo real.

## Autor

Desarrollado por Santiago Sinza, como proyecto académico enfocado en el procesamiento digital de imágenes mediante Python, OpenCV y NumPy.