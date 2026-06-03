"""
Módulo de Seguimiento Fiduciario Central (Fiducial Tracking - Exp 2)
--------------------------------------------------------------------
Este script implementa un motor de anclaje visual dinámico específico para la
topología del Experimento 2 (Sensores de Temperatura Reversibles). Detecta marcadores
físicos de referencia (cuadrados negros) mediante segmentación cromática en el
espacio HSV y filtrado morfológico-geométrico.

A diferencia del seguimiento lateral (Eje X), este algoritmo minimiza el error de
paralaje y distorsión de lente seleccionando automáticamente el marcador fiduciario
más próximo al eje central vertical (Eje Y) de la captura fotográfica.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import numpy as np


def get_center_square_anchor(img):
    """
    Localiza el marcador fiduciario cuadrangular negro más estable de la imagen.

    Args:
        img (numpy.ndarray): Fotograma original en espacio de color BGR.

    Returns:
        tuple: Coordenadas (x_base, y_base) de la esquina superior izquierda (Top-Left)
               del cuadrado seleccionado, o None si hay oclusión total.
    """
    # 1. Análisis del plano espacial de la imagen
    # Se extrae la altura total para calcular el eje de simetría horizontal (Y-center)
    alto_imagen, ancho_imagen = img.shape[:2]
    centro_y_imagen = alto_imagen / 2.0

    # Transformación cilíndrica para aislar la luminosidad del croma
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Segmentación Cromática (Búsqueda de absorción total de luz / Negro)
    # Rango amplio en Hue y Saturation, pero estricto en Value (0-60) para capturar oscuridad
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 3. Filtrado Morfológico (Operador de Cierre)
    # Elimina ruido de sal y pimienta (píxeles blancos aislados dentro de los cuadrados negros)
    # provocados por texturas de la impresión o reflejos especulares leves.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Extracción Topológica
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h

        # 5. Filtrado Geométrico Estricto
        # Condición 1: Filtrado de ruido de fondo (Área mínima > 1000 px)
        # Condición 2: El marcador debe ser un cuadrado perfecto (Tolerancia de ratio 0.8 - 1.2)
        if area > 1000 and 0.8 < aspect_ratio < 1.2:
            # Cálculo del baricentro vertical del candidato
            centro_y_cuadrado = y + (h / 2.0)

            # Cálculo del error espacial absoluto respecto al centro óptico de la cámara
            distancia_al_centro = abs(centro_y_cuadrado - centro_y_imagen)

            # Almacenamiento de metadatos del candidato: (Error, X, Y, W, H)
            squares.append((distancia_al_centro, x, y, w, h))

    # Control de excepciones por oclusión severa o fallo de iluminación
    if not squares:
        return None

    # 6. Selección de Óptimo Global
    # Se ordena la matriz de candidatos en función de su distancia al centro (índice 0).
    # Esto garantiza seleccionar el marcador que sufre menor distorsión de perspectiva.
    squares.sort(key=lambda s: s[0])

    # El candidato óptimo es el primer elemento [0] de la lista ordenada
    _, x_base, y_base, _, _ = squares[0]

    return x_base, y_base