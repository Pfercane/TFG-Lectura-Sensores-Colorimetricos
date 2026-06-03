"""
Módulo de Seguimiento Fiduciario Lateral (Fiducial Tracking - Exp 3)
--------------------------------------------------------------------
Este script implementa el motor de anclaje visual dinámico diseñado
específicamente para la topología del Experimento 3 (Sensores de Humedad).

En lugar de utilizar marcadores aislados, este algoritmo segmenta el bloque
vertical de referencias radiométricas (masas negras) y extrae su vértice
superior derecho. Esta decisión geométrica minimiza el vector de distancia
(brazo de palanca) hacia las ventanas de lectura del sensor, reduciendo 
significativamente el error de traslación frente a leves rotaciones de la cámara.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import numpy as np


def get_ref_rectangle_anchor(img):
    """
    Detecta el bloque de referencias negras de la izquierda y calcula
    la coordenada espacial (X, Y) de su esquina SUPERIOR DERECHA visible.

    Args:
        img (numpy.ndarray): Fotograma original en espacio de color BGR.

    Returns:
        tuple: Coordenadas (anchor_x_right, anchor_y_top) del vértice óptimo,
               o None si existe oclusión total.
    """
    # 1. Transformación al Espacio de Color Cilíndrico (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Segmentación Cromática (Búsqueda de absorción total de luz / Negro)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 3. Filtrado Morfológico (Operador de Cierre)
    # Suaviza los bordes del bloque negro y rellena posibles reflejos
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Extracción Topológica de Contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_rects = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Prevención de divisiones por cero en contornos anómalos
        aspect_ratio = h / float(w) if w > 0 else 0

        # 5. Filtrado Geométrico Estricto
        # Condición 1: Eliminar ruido de fondo (área > 1000)
        # Condición 2: Buscar topología VERTICAL (aspect_ratio > 1.2),
        # lo que descarta cuadrados (Exp2) o líneas horizontales.
        if area > 1000 and aspect_ratio > 1.2:
            candidate_rects.append((x, y, w, h))

    # Control de excepciones (Oclusión o desenfoque severo)
    if not candidate_rects:
        return None

    # 6. Ordenación Espacial Absoluta (Eje Y)
    # Ordenamos la matriz de candidatos de ARRIBA a ABAJO.
    candidate_rects.sort(key=lambda r: r[1])

    # Se selecciona de forma determinista el rectángulo superior (índice 0)
    best_x, best_y, best_w, best_h = candidate_rects[0]

    # 7. Cálculo del Vértice Óptimo (Esquina Superior Derecha)
    # Al sumar la anchura (best_w) a la coordenada origen en X (best_x),
    # desplazamos el ancla al borde más cercano a los sensores de humedad.
    anchor_x_right = best_x + best_w
    anchor_y_top = best_y

    return anchor_x_right, anchor_y_top