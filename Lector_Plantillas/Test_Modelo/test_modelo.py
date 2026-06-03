"""
Módulo de Validación Estática y Prueba Unitaria (Mock Test)
-----------------------------------------------------------
Este script actúa como un entorno de prueba aislado para auditar la precisión
de los modelos predictivos pre-entrenados (.pkl). Al utilizar coordenadas
absolutas estáticas sobre una imagen de control conocida, permite validar
la integridad de las funciones de calibración radiométrica y el motor de
inferencia de Machine Learning, aislando posibles variables de error derivadas
del sistema dinámico de visión artificial (lector QR).

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import numpy as np
import joblib
import os

# ==========================================
# 1. CONFIGURACIÓN DE COORDENADAS ORIGINALES (STATIC MOCK)
# ==========================================
# Imagen de control con condiciones lumínicas conocidas
IMG_PATH = '2000_exp2.jpg'
# Dimensión física de la ventana de lectura (Cuadrado de 20x20 píxeles)
SIZE_PX = 20

# Coordenadas de calibración radiométrica (Fondo del ensayo)
W_REF_TL = (647, 215)
B_REF_TL = (647, 335)

# Array estático de coordenadas topológicas (Sensor de Temperatura Reversible)
VENTANAS_TL = [
    (444, 474),  # V_1
    (444, 426),  # V_2
    (445, 385),  # V_3
    (445, 346),  # V_4
    (446, 307),  # V_5
    (447, 265),  # V_6
    (447, 225),  # V_7
    (448, 185),  # V_8
    (448, 145),  # V_9
    (449, 112)  # V_10
]


# ==========================================
# 2. MOTOR DE EXTRACCIÓN Y CALIBRACIÓN
# ==========================================
def obtener_media_rgb_roi(img, tl_x, tl_y, size):
    """
    Extrae la media de los canales RGB de una Región de Interés (ROI) estática.

    Args:
        img (numpy.ndarray): Matriz de la imagen en espacio RGB.
        tl_x, tl_y (int): Coordenada X e Y de la esquina superior izquierda (Top-Left).
        size (int): Lado del cuadrado de extracción en píxeles.

    Returns:
        numpy.ndarray: Vector [R, G, B] con la media de intensidad.
    """
    roi = img[tl_y: tl_y + size, tl_x: tl_x + size]
    return np.array([np.mean(roi[:, :, 0]), np.mean(roi[:, :, 1]), np.mean(roi[:, :, 2])])


def calibrar_y_extraer_rojo(val_crudo, w_ref, b_ref):
    """
    Aplica la ecuación de calibración radiométrica y normaliza la característica objetivo.
    (Nota: El Experimento 2 utiliza exclusivamente el canal Rojo para la inferencia).
    """
    if w_ref - b_ref == 0:
        return 0.0
    calibrado = (255.0 / (w_ref - b_ref)) * (val_crudo - b_ref)
    calibrado_clip = np.clip(calibrado, 0, 255)
    return calibrado_clip / 255.0


# ==========================================
# 3. EJECUCIÓN DIRECTA DEL MOCK TEST
# ==========================================
print(">> INICIANDO TEST DE PRECISIÓN (EXP 2: REVERSIBLE)...\n")

try:
    # 1. Ingesta del Modelo Serializado y la Imagen de Control
    # Subimos 2 niveles de carpetas para llegar a Modelado_Predictivo
    ruta_modelo = os.path.join("..", "..", "Modelado_Predictivo", "Modelos_Exportados", "modelo_random_forest_exp2.pkl")
    modelo = joblib.load(ruta_modelo)

    img_bgr = cv2.imread(IMG_PATH)

    if img_bgr is None:
        raise ValueError(f"No se encontró el artefacto de imagen en: {IMG_PATH}")

    # Transformación del espacio de color BGR (OpenCV) a RGB estándar
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Lectura del entorno lumínico (Referencias Absolutas)
    w_rgb = obtener_media_rgb_roi(img_rgb, W_REF_TL[0], W_REF_TL[1], SIZE_PX)
    b_rgb = obtener_media_rgb_roi(img_rgb, B_REF_TL[0], B_REF_TL[1], SIZE_PX)

    # 3. Construcción iterativa del Vector de Características
    vector_features = []
    for (tl_x, tl_y) in VENTANAS_TL:
        rgb_crudo = obtener_media_rgb_roi(img_rgb, tl_x, tl_y, SIZE_PX)

        # Preprocesamiento: Calibración y normalización del canal de interés
        feature_rojo = calibrar_y_extraer_rojo(rgb_crudo[0], w_rgb[0], b_rgb[0])
        vector_features.append(feature_rojo)

    # 4. Inferencia del Modelo (Predicción Estricta)
    features_array = np.array([vector_features])

    print(f"[DEBUG] Vector Final Extraído: {np.round(features_array[0], 3)}")

    prediccion = modelo.predict(features_array)[0]
    print(f"\n>> [RESULTADO FINAL] Temperatura Predicha: {prediccion:.2f} °C\n")

except Exception as e:
    print(f">> ERROR DE EJECUCIÓN LÓGICA: {e}")