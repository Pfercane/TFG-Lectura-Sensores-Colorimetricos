"""
Módulo de Análisis Colorimétrico para Sensores TimeStrip (Experimento 4)
------------------------------------------------------------------------
Este script automatiza la extracción de datos colorimétricos (RGB) a partir
de fotografías tomadas en entornos no controlados. Emplea técnicas de Visión
por Computador (OpenCV) y transformaciones afines para localizar regiones
de interés (ROIs) basándose en anclajes geométricos relativos.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import numpy as np
import os
import glob
import re
import csv

# ==========================================
# 1. CONFIGURACIÓN GEOMÉTRICA (CALIBRACIÓN)
# ==========================================
# Medida física real de la caja ancla en centímetros
ANCHO_CAJA_CM = 2.5

# Coordenadas en píxeles de la imagen de referencia para establecer el origen
CAJA_TL = (712, 387)
CAJA_TR = (1175, 390)

VENTANA_TL = (896, 565)
VENTANA_ANCHO = 75
VENTANA_ALTO = 180

W_REF_TL = (1315, 720)

# Coordenadas de los 4 parches negros redundantes para mitigar brillos especulares
B_REFS_TL = [
    (1315, 520),
    (1330, 955),
    (2135, 922),
    (2140, 510)
]
REF_LADO = 100

# ==========================================
# 2. RUTAS DE DIRECTORIOS
# ==========================================
CARPETA_IMAGENES = "Imagenes"
CARPETA_DATOS = "Datos"

# Creación automática del directorio de resultados si no existe
os.makedirs(CARPETA_DATOS, exist_ok=True)

# Archivo de volcado de la telemetría RGB
ARCHIVO_CSV_SALIDA = os.path.join(CARPETA_DATOS, 'datos_color_4.csv')
CARPETA_DEBUG = os.path.join(CARPETA_IMAGENES, 'debug_vision')


# ==========================================
# 3. MOTOR MATEMÁTICO Y GEOMÉTRICO
# ==========================================
def preparar_configuracion_interna():
    """
    Convierte las coordenadas absolutas en píxeles a offsets relativos en centímetros
    tomando como origen (0,0) la esquina superior izquierda de la caja principal.

    Returns:
        dict: Diccionario anidado con los desplazamientos (off_x, off_y) y dimensiones
              en centímetros para cada región de interés (Sensores y Referencias).
    """
    ancho_caja_px_paint = np.linalg.norm(np.array(CAJA_TR) - np.array(CAJA_TL))
    px_por_cm_paint = ancho_caja_px_paint / ANCHO_CAJA_CM

    origen_x, origen_y = CAJA_TL[0], CAJA_TL[1]

    def calcular_centro_relativo_cm(tl_px, ancho_px, alto_px):
        centro_x_px = tl_px[0] + (ancho_px / 2.0)
        centro_y_px = tl_px[1] + (alto_px / 2.0)
        offset_x_cm = (centro_x_px - origen_x) / px_por_cm_paint
        offset_y_cm = (centro_y_px - origen_y) / px_por_cm_paint
        ancho_cm = ancho_px / px_por_cm_paint
        alto_cm = alto_px / px_por_cm_paint
        return offset_x_cm, offset_y_cm, ancho_cm, alto_cm

    v_off_x, v_off_y, v_w_cm, v_h_cm = calcular_centro_relativo_cm(VENTANA_TL, VENTANA_ANCHO, VENTANA_ALTO)
    w_off_x, w_off_y, ref_w_cm, _ = calcular_centro_relativo_cm(W_REF_TL, REF_LADO, REF_LADO)

    b_refs_config = []
    for b_tl in B_REFS_TL:
        b_off_x, b_off_y, _, _ = calcular_centro_relativo_cm(b_tl, REF_LADO, REF_LADO)
        b_refs_config.append({"off_x": b_off_x, "off_y": b_off_y, "w": ref_w_cm, "h": ref_w_cm})

    return {
        "Ventana": {"off_x": v_off_x, "off_y": v_off_y, "w": v_w_cm, "h": v_h_cm},
        "W_Ref": {"off_x": w_off_x, "off_y": w_off_y, "w": ref_w_cm, "h": ref_w_cm},
        "B_Refs": b_refs_config
    }


def ordenar_puntos(pts):
    """
    Ordena los vértices de un cuadrilátero de forma consistente.

    Args:
        pts (numpy.ndarray): Matriz 4x2 con las coordenadas de los vértices.

    Returns:
        numpy.ndarray: Vértices ordenados [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extraer_region(img, img_debug, tl_caja, tr_caja, bl_caja, config_region, nombre_etiqueta, tipo="sensor"):
    """
    Calcula la posición exacta de una ROI mediante proyección vectorial y extrae su color medio.

    Args:
        img (numpy.ndarray): Imagen original en formato BGR.
        img_debug (numpy.ndarray): Imagen de lienzo para dibujar las detecciones.
        tl_caja, tr_caja, bl_caja (numpy.ndarray): Vértices de la caja ancla detectada.
        config_region (dict): Configuración relativa de la región a extraer.
        nombre_etiqueta (str): Etiqueta visual para el modo debug.
        tipo (str): Define la lógica de extracción ("sensor" o "referencia").

    Returns:
        list: Valores medios [R, G, B] extraídos de la región, o None si hay error de límites.
    """
    # 1. Cálculo del factor de escala dinámico para la imagen actual
    ancho_px = np.linalg.norm(tr_caja - tl_caja)
    pixeles_por_cm = ancho_px / ANCHO_CAJA_CM

    # 2. Obtención de los vectores direccionales (mitiga rotaciones y perspectiva)
    dir_x = (tr_caja - tl_caja) / ancho_px
    dir_y = (bl_caja - tl_caja) / np.linalg.norm(bl_caja - tl_caja)

    vec_x_1cm = dir_x * pixeles_por_cm
    vec_y_1cm = dir_y * pixeles_por_cm

    # 3. Proyección del centro teórico de la ROI
    pos_centro = tl_caja + (config_region["off_x"] * vec_x_1cm) + (config_region["off_y"] * vec_y_1cm)

    # --- LÓGICA DE EXTRACCIÓN: SENSORES (MÁSCARA ROTADA) ---
    if tipo == "sensor":
        mitad_w = (config_region["w"] * pixeles_por_cm) / 2.0
        mitad_h = (config_region["h"] * pixeles_por_cm) / 2.0

        p1 = pos_centro - (dir_x * mitad_w) - (dir_y * mitad_h)
        p2 = pos_centro + (dir_x * mitad_w) - (dir_y * mitad_h)
        p3 = pos_centro + (dir_x * mitad_w) + (dir_y * mitad_h)
        p4 = pos_centro - (dir_x * mitad_w) + (dir_y * mitad_h)

        pts = np.array([p1, p2, p3, p4], np.int32)

        color_debug = (0, 255, 0)
        cv2.polylines(img_debug, [pts], True, color_debug, 2)
        cv2.putText(img_debug, nombre_etiqueta, (pts[0][0], pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_debug,
                    2)

        # Se aplica una máscara poligonal para promediar solo el área interior rotada
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mean_color = cv2.mean(img, mask=mask)
        return [mean_color[2], mean_color[1], mean_color[0]]

    # --- LÓGICA DE EXTRACCIÓN: REFERENCIAS (PROYECCIÓN DIRECTA) ---
    else:
        cx, cy = int(pos_centro[0]), int(pos_centro[1])

        # Margen de seguridad: reducción del área de lectura al 70% para evitar bordes
        mitad_w = int((config_region["w"] * pixeles_por_cm) * 0.3)
        mitad_h = int((config_region["h"] * pixeles_por_cm) * 0.3)

        if cy - mitad_h < 0 or cy + mitad_h > img.shape[0] or cx - mitad_w < 0 or cx + mitad_w > img.shape[1]:
            return None

        pt1 = (cx - mitad_w, cy - mitad_h)
        pt2 = (cx + mitad_w, cy + mitad_h)

        color_debug = (255, 0, 0)
        cv2.rectangle(img_debug, pt1, pt2, color_debug, 2)
        cv2.putText(img_debug, nombre_etiqueta, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_debug, 2)

        roi = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        return [np.mean(roi[:, :, 2]), np.mean(roi[:, :, 1]), np.mean(roi[:, :, 0])]


# ==========================================
# 4. PIPELINE PRINCIPAL DE EJECUCIÓN
# ==========================================
def procesar_imagenes():
    """
    Función orquestadora. Itera sobre el dataset de imágenes, aplica binarización
    adaptativa para detectar los anclajes, extrae los valores RGB y consolida los
    datos en un archivo CSV.
    """
    config = preparar_configuracion_interna()
    archivos = sorted(glob.glob(os.path.join(CARPETA_IMAGENES, '*.jpg')))

    if not archivos:
        print("No se encontraron imágenes.")
        return

    os.makedirs(CARPETA_DEBUG, exist_ok=True)

    with open(ARCHIVO_CSV_SALIDA, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fecha', 'Sensor', 'R', 'G', 'B'])

        for ruta_img in archivos:
            nombre_archivo = os.path.basename(ruta_img)

            # Extracción de metadatos desde el nombre del archivo (YYYYMMDD_GRUPO)
            match = re.search(r'(\d{8})_(\d)', nombre_archivo)
            if not match: continue

            fecha = match.group(1)
            grupo = int(match.group(2))
            offset_sensor_id = 0 if grupo == 1 else 3

            img = cv2.imread(ruta_img)
            if img is None: continue

            img_debug = img.copy()

            # 1. Preprocesamiento y Segmentación (Umbralización adaptativa)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 2. Filtrado Geométrico de Cajas de Anclaje
            rectangulos_candidatos = []
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                area = w * h
                if area > 10000:  # Filtro de ruido por área mínima
                    if 1.2 < max(w, h) / min(w, h) < 4.0:  # Filtro por relación de aspecto
                        box = np.intp(cv2.boxPoints(rect))
                        rectangulos_candidatos.append((box, area))

            # Se retienen estrictamente los 3 rectángulos de mayor área y se ordenan en el eje X
            rectangulos_candidatos = sorted(rectangulos_candidatos, key=lambda x: x[1], reverse=True)[:3]
            cajas_finales = [item[0] for item in rectangulos_candidatos]
            cajas_finales = sorted(cajas_finales, key=lambda b: b[0][0])

            if len(cajas_finales) != 3:
                print(f"AVISO: {nombre_archivo} no tiene 3 cajas válidas (detectadas {len(cajas_finales)}). Saltando.")
                continue

            print(f"Procesando {nombre_archivo}...")

            # 3. Establecimiento del Sistema de Referencia Local (Basado en la Caja Izquierda)
            pts_caja1 = ordenar_puntos(cajas_finales[0])
            tl1, tr1, br1, bl1 = pts_caja1[0], pts_caja1[1], pts_caja1[2], pts_caja1[3]

            cv2.polylines(img_debug, [np.array([tl1, tr1, br1, bl1], np.int32)], True, (0, 255, 255), 2)

            # 4. Extracción Colorimétrica Estricta mediante Vectorización
            # 4.1. Referencia Blanca
            rgb_w = extraer_region(img, img_debug, tl1, tr1, bl1, config["W_Ref"], "W_Ref", tipo="referencia")

            # 4.2. Referencias Negras (Redundancia para eliminación de brillos especulares)
            b_results = []
            for idx, b_cfg in enumerate(config["B_Refs"]):
                rgb_b_temp = extraer_region(img, img_debug, tl1, tr1, bl1, b_cfg, f"B{idx + 1}", tipo="referencia")
                if rgb_b_temp:
                    b_results.append(rgb_b_temp)

            if rgb_w and b_results:
                # Criterio de Selección: El píxel más oscuro (menor suma RGB) es el menos afectado por el flash
                rgb_b_final = min(b_results, key=lambda x: sum(x))
                writer.writerow([fecha, 'W_Ref', rgb_w[0], rgb_w[1], rgb_w[2]])
                writer.writerow([fecha, 'B_Ref', rgb_b_final[0], rgb_b_final[1], rgb_b_final[2]])
            else:
                print(f"  -> Error extrayendo Referencias en {nombre_archivo}")

            # 4.3. Sensores Múltiples (Anclaje iterativo)
            for i, box in enumerate(cajas_finales):
                num_sensor = str(offset_sensor_id + i + 1)
                pts = ordenar_puntos(box)
                tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

                if i > 0:
                    cv2.polylines(img_debug, [np.array([tl, tr, br, bl], np.int32)], True, (0, 255, 255), 2)

                rgb_sensor = extraer_region(img, img_debug, tl, tr, bl, config["Ventana"], f"Sen {num_sensor}",
                                            tipo="sensor")

                if rgb_sensor:
                    writer.writerow([fecha, num_sensor, rgb_sensor[0], rgb_sensor[1], rgb_sensor[2]])
                else:
                    print(f"  -> Error extrayendo Sensor {num_sensor} en {nombre_archivo}")

            # 5. Generación de Artefactos de Validación (Debug)
            ruta_debug = os.path.join(CARPETA_DEBUG, f"debug_{nombre_archivo}")
            cv2.imwrite(ruta_debug, img_debug)

    print(f"\nTerminado. Resultados en: {ARCHIVO_CSV_SALIDA}")
    print(f"Imágenes de depuración guardadas en: {CARPETA_DEBUG}")


if __name__ == "__main__":
    procesar_imagenes()