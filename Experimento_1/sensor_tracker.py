"""
Módulo de Seguimiento Fiduciario (Fiducial Tracking)
----------------------------------------------------
Este script implementa un motor de anclaje visual dinámico. Detecta marcadores
físicos de referencia (etiquetas rojas rectangulares) mediante umbralización en
el espacio de color HSV y filtrado de contornos geométricos.
Su objetivo es compensar las vibraciones mecánicas o desplazamientos de la cámara
durante ensayos longitudinales (Time-Lapse), garantizando que las regiones de
extracción colorimétrica se mantengan espacialmente coherentes.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import numpy as np


def _get_all_sensor_rectangles(img):
    """
    Función interna (Core). Escanea la imagen, aísla el espectro de color rojo,
    reduce el ruido morfológico y devuelve las coordenadas espaciales de los
    marcadores detectados, ordenados posicionalmente de izquierda a derecha.

    Args:
        img (numpy.ndarray): Fotograma en formato BGR.

    Returns:
        list: Lista de tuplas (x, y, w, h) con los bounding boxes detectados.
    """
    # 1. Transformación al Espacio de Color Cilíndrico (HSV)
    # Permite aislar el cromatismo (Hue) independientemente de la iluminación (Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Segmentación Cromática Dual (Espectro Rojo)
    # En OpenCV, el color rojo cruza el límite del cilindro HSV (0 y 180).
    # Se requieren dos máscaras para capturar ambos extremos del espectro.
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 3. Filtrado Morfológico (Cierre)
    # Rellena pequeños huecos negros dentro del área roja (ruido o reflejos)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Extracción Topológica (Contornos)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        area = w * h

        # 5. Filtrado Geométrico
        # Discrimina el ruido basándose en la física conocida de la etiqueta:
        # Debe tener un área significativa (>2000px) y ser rectangular vertical (ratio > 3)
        if area > 2000 and aspect_ratio > 3:
            rectangles.append((x, y, w, h))

    if not rectangles:
        return None

    # 6. Ordenación Espacial Absoluta (Eje X)
    # Garantiza que el índice 0 siempre sea el sensor izquierdo, sin importar
    # en qué orden encontró OpenCV los contornos.
    rectangles.sort(key=lambda r: r[0])
    return rectangles


def get_left_sensor_anchor(img):
    """
    Interfaz de extracción. Devuelve la coordenada origen (Top-Left) del
    marcador fiduciario correspondiente al sensor instalado a la izquierda.
    """
    rectangles = _get_all_sensor_rectangles(img)
    if rectangles:
        x_base, y_base, _, _ = rectangles[0]  # Selección por índice inferior
        return x_base, y_base
    return None


def get_right_sensor_anchor(img):
    """
    Interfaz de extracción. Devuelve la coordenada origen (Top-Left) del
    marcador fiduciario correspondiente al sensor instalado a la derecha.
    """
    rectangles = _get_all_sensor_rectangles(img)
    if rectangles and len(rectangles) >= 2:
        x_base, y_base, _, _ = rectangles[-1]  # Selección por índice superior (cola)
        return x_base, y_base
    return None