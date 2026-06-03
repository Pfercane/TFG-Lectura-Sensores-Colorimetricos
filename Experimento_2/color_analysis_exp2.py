"""
Módulo de Extracción Colorimétrica Dinámica (Experimento 2)
-----------------------------------------------------------
Este script unificado procesa simultáneamente múltiples etiquetas de
temperatura reversible (izquierda y derecha). Utiliza un algoritmo de
seguimiento basado en el baricentro de la imagen para anclar las
Regiones de Interés (ROIs) y compensar los micromovimientos de la cámara,
garantizando una extracción RGB espacialmente coherente durante todo el ensayo.

El sistema emplea entrada/salida (I/O) de pasada única, analizando la
imagen una sola vez para extraer la telemetría de todos los sensores
presentes y serializándola concurrentemente en ficheros independientes.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import os
import csv
import re
from sensor_tracker_2 import get_center_square_anchor

# ==========================================
# 1. CONFIGURACIÓN TOPOLÓGICA DE LOS SENSORES
# ==========================================
# Rutas relativas para garantizar la portabilidad del repositorio (GitHub)
IMAGE_FOLDER = "Imagenes"
DATOS_FOLDER = "Datos"

# Creación automática del directorio de volcado si no existe
os.makedirs(DATOS_FOLDER, exist_ok=True)

ROI_SIZE = 20  # Dimensión de las ventanas de extracción (20x20 píxeles)

# Diccionario maestro de topología. En este experimento, ambos sensores
# pivotan sobre la misma referencia fiducial central para evitar paralaje.
SENSORES_CONFIG = {
    "Izquierdo": {
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_2_left.csv"),
        "tracker_func": get_center_square_anchor,
        "coords": {
            'W_Ref': (647, 215), 'B_Ref': (647, 335),
            'Temp_05': (444, 474), 'Temp_10': (444, 426), 'Temp_15': (445, 385),
            'Temp_20': (445, 346), 'Temp_25': (446, 307), 'Temp_30': (447, 265),
            'Temp_35': (447, 225), 'Temp_40': (448, 185), 'Temp_45': (448, 145),
            'Temp_50': (449, 112)
        }
    },
    "Derecho": {
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_2_right.csv"),
        "tracker_func": get_center_square_anchor,
        "coords": {
            'W_Ref': (647, 215), 'B_Ref': (647, 335),
            'Temp_05': (838, 472), 'Temp_10': (838, 427), 'Temp_15': (838, 387),
            'Temp_20': (839, 347), 'Temp_25': (839, 308), 'Temp_30': (840, 266),
            'Temp_35': (840, 225), 'Temp_40': (840, 185), 'Temp_45': (841, 146),
            'Temp_50': (841, 114)
        }
    }
}


# ==========================================
# 2. MOTOR DE EXTRACCIÓN Y PIPELINE PRINCIPAL
# ==========================================
def procesar_experimento():
    """
    Orquesta el flujo de trabajo de visión artificial. Itera sobre el dataset
    fotográfico, calcula los offsets de calibración inicial (auto-calibración)
    y proyecta las mallas de lectura de forma dinámica.
    """
    # Obtención de la lista de imágenes con ordenación natural (numérica)
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))],
                         key=lambda x: int(re.search(r'\d+', x).group()))

    if not image_files:
        print(">> ERROR: No se encontraron imágenes en el directorio especificado.")
        return

    # Estructuras de estado para gestionar el tracking y la escritura concurrente
    archivos_csv = {}
    escritores_csv = {}
    offsets_dinamicos = {"Izquierdo": {}, "Derecho": {}}
    ultimos_anchors = {"Izquierdo": None, "Derecho": None}

    try:
        # 2.1. Apertura concurrente de archivos de salida
        for nombre_sensor, config in SENSORES_CONFIG.items():
            f = open(config["csv_file"], mode='w', newline='')
            writer = csv.writer(f)
            writer.writerow(['Image', 'Window', 'R', 'G', 'B'])
            archivos_csv[nombre_sensor] = f
            escritores_csv[nombre_sensor] = writer

        # 2.2. Bucle principal de procesado por fotograma
        for index, image_name in enumerate(image_files):
            image_path = os.path.join(IMAGE_FOLDER, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"  -> AVISO: Error de lectura en imagen {image_name}")
                continue

            # Procesamiento iterativo de la malla izquierda y derecha
            for nombre_sensor, config in SENSORES_CONFIG.items():

                # A. Tracking: Localización espacial del marcador en el frame actual
                current_anchor = config["tracker_func"](image)

                # Fallback: Tolerancia a fallos por oclusión o desenfoque
                if current_anchor is None:
                    if ultimos_anchors[nombre_sensor] is not None:
                        current_anchor = ultimos_anchors[nombre_sensor]
                    else:
                        print(f"  -> AVISO: {image_name} sin anclaje para sensor {nombre_sensor}.")
                        continue

                ultimos_anchors[nombre_sensor] = current_anchor
                anchor_x, anchor_y = current_anchor

                # B. Auto-Calibración Inicial (Fotograma 0)
                # Ancla el sistema de coordenadas relativo entre el fiducial y las ROIs
                if index == 0:
                    for window_name, (orig_x, orig_y) in config["coords"].items():
                        offset_x = orig_x - anchor_x
                        offset_y = orig_y - anchor_y
                        offsets_dinamicos[nombre_sensor][window_name] = (offset_x, offset_y)
                    print(f">> Auto-calibración completada para sensor {nombre_sensor}.")

                # C. Extracción Radiométrica Dinámica
                for window_name, (off_x, off_y) in offsets_dinamicos[nombre_sensor].items():
                    # Proyección geométrica aplicando los offsets al anclaje actualizado
                    new_x = anchor_x + off_x
                    new_y = anchor_y + off_y

                    # Segmentación de la Región de Interés (ROI)
                    window_roi = image[new_y:new_y + ROI_SIZE, new_x:new_x + ROI_SIZE]

                    # Extracción y promedio del espacio de color RGB
                    avg_color = cv2.mean(window_roi)[:3]  # OpenCV utiliza formato BGR
                    r_avg, g_avg, b_avg = avg_color[2], avg_color[1], avg_color[0]

                    # Serialización asíncrona del dato
                    escritores_csv[nombre_sensor].writerow([image_name, window_name, r_avg, g_avg, b_avg])

            if index % 100 == 0:
                print(f">> Procesada imagen {index}: {image_name}")

        print("\n>> EXTRACCIÓN DINÁMICA MULTI-SENSOR FINALIZADA CON ÉXITO.")

    finally:
        # 2.3. Cierre seguro de flujos de datos (Memory cleanup)
        for f in archivos_csv.values():
            f.close()


if __name__ == "__main__":
    procesar_experimento()