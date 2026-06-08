# Sistema Modular para la Lectura Automatizada de Sensores Colorimétricos en Logística

Este repositorio contiene la arquitectura de software desarrollada para el Trabajo Fin de Grado (TFG) en Ingeniería de Tecnologías de Telecomunicación. 

El proyecto resuelve el problema de la trazabilidad subjetiva en cadenas logísticas mediante la digitalización de etiquetas colorimétricas (temperatura reversible/irreversible, humedad y tiempo). Se ha diseñado un *pipeline* integral que abarca desde la captura de telemetría física hasta el despliegue de modelos de Machine Learning sobre anclajes fiduciales dinámicos (Códigos QR).

## Dataset Masivo y Especificaciones de Hardware
Para mantener la agilidad del repositorio, los directorios locales contienen únicamente un *Minimal Reproducible Example (MRE)*. 
El conjunto de datos fotográficos longitudinal (> X.XXX capturas por experimento) y las hojas de características técnicas (Datasheets) de los sensores comerciales evaluados se encuentran alojados en un repositorio externo:
* **[🔗 Enlace al Dataset Completo y Datasheets (Google Drive)](https://drive.google.com/drive/folders/10mqV171t6ejhxQpkpCbrcj8Qoxw4qmoo?usp=sharing)**

## Arquitectura del Repositorio

El sistema está desacoplado en cuatro grandes bloques funcionales:

### 1. Fases de Extracción ETL y Procesado de Señal (MATLAB & Python)
Pipelines independientes para calibración radiométrica, sincronización temporal con la telemetría de la cámara climática y mitigación de histéresis térmica/higrométrica.
* `\Experimento_1`: Sensores de Temperatura Irreversible.
* `\Experimento_2`: Sensores de Temperatura Reversible (Histéresis).
* `\Experimento_3`: Sensores de Humedad Relativa (Gradiente Vertical).
* `\Experimento_4`: Ensayo Longitudinal en entorno real (Sensor TimeStrip).

### 2. Motor Central de Inteligencia Artificial
* `\Modelado_Predictivo`: Orquestador centralizado para la validación cruzada (*Hold-out*) y entrenamiento de algoritmos (*Random Forest Regressor* y *Regresión Polinómica*). Contiene los artefactos `.pkl` listos para producción.

### 3. Entorno de Producción e Inferencia
* `\Lector_Plantillas`: Core del sistema final. Algoritmo dinámico que decodifica IDs en códigos QR, carga topologías espaciales predefinidas, compensa distorsiones de perspectiva en fotografías no controladas y ejecuta predicciones simultáneas multi-sensor.
* `\Generador_QR`: Herramienta industrial para la creación de anclajes fiduciales con corrección de errores de alto nivel (Reed-Solomon).

### 4. Herramientas Auxiliares de Diagnóstico
* `\Utils`: Scripts de automatización para la telemetría visual (*Time-Lapse* a 1 Hz) y herramientas de compilación de vídeo dinámico para auditorías rápidas de integridad mecánica en laboratorio.

## Flujo de Ejecución (Reproducibilidad)
1. **Extracción Visión Artificial:** Ejecutar `color_analysis_expX.py` dentro del experimento deseado. Se generarán los datos brutos en `\Datos`.
2. **Fusión Matemática:** Ejecutar `procesado_expX.m` en MATLAB para sincronizar con el hardware de la cámara climática y obtener los CSV limpios.
3. **Compilación ML:** Desde `\Modelado_Predictivo`, emplear `sensor_training.py` para re-entrenar y actualizar los modelos globales en `\Modelos_Exportados`.
4. **Inferencia:** El usuario final solo necesita interactuar con `lector_plantillas.py` pasando una captura en campo.

## Tecnologías y Librerías
* **Visión por Computador:** `OpenCV`, `Numpy`.
* **Procesado Digital de Señal:** `MATLAB` (Interpolación, Filtering, Signal Fusion).
* **Machine Learning:** `Scikit-Learn`, `Pandas` (Random Forest, Pipelines, StandardScaler).
* **Fiducial Tracking:** `Segno` (QR Generation), Algoritmos propios de anclaje por baricentro y morfología.

---
*Desarrollado por Pedro Gabriel Fernández Cañete* | *Universidad de Granada (UGR)*
