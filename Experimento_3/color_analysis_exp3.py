"""
Módulo de Extracción Colorimétrica Dinámica Multi-Capa (Experimento 3)
----------------------------------------------------------------------
Este script unificado procesa simultáneamente un array vertical de sensores
de humedad relativa ubicados en tres cotas distintas (Inferior, Media, Superior)
dentro de la cámara climática.

Emplea un anclaje fiducial asimétrico (esquina superior derecha del bloque de
referencias) para minimizar el error de paralaje. Además, la arquitectura
soporta dimensiones de extracción (ROI) heterogéneas entre capas (40x40px
vs 20x20px), adaptándose dinámicamente a la dispersión focal de la cámara.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import os
import csv
import re
from sensor_tracker_3 import get_ref_rectangle_anchor

# ==========================================
# 1. CONFIGURACIÓN TOPOLÓGICA DE LOS SENSORES
# ==========================================
# Rutas relativas para portabilidad del proyecto (GitHub)
IMAGE_FOLDER = "Imagenes"
DATOS_FOLDER = "Datos"

# Creación automática del directorio de resultados
os.makedirs(DATOS_FOLDER, exist_ok=True)

# Diccionario maestro de topología.
# Define el comportamiento dinámico de cada cota espacial (Abajo, Medio, Arriba),
# especificando su archivo de salida, la dimensión de su ROI y sus coordenadas.
SENSORES_CONFIG = {
    "Abajo": {
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_3_bot.csv"),
        "tracker_func": get_ref_rectangle_anchor,
        "roi_size": 40,
        "coords": {
            'W_Ref': (170, 355), 'B_Ref': (170, 260),
            'H_20': (362, 172), 'H_40': (487, 173), 'H_60': (613, 173), 'H_80': (732, 173)
        }
    },
    "Medio": {
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_3_mid.csv"),
        "tracker_func": get_ref_rectangle_anchor,
        "roi_size": 40,
        "coords": {
            'W_Ref': (170, 355), 'B_Ref': (170, 260),
            'H_20': (365, 353), 'H_40': (492, 352), 'H_60': (615, 352), 'H_80': (736, 351)
        }
    },
    "Arriba": {
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_3_up.csv"),
        "tracker_func": get_ref_rectangle_anchor,
        "roi_size": 40,
        "coords": {
            'W_Ref': (170, 355), 'B_Ref': (170, 260),
            'H_20': (364, 520), 'H_40': (492, 518), 'H_60': (613, 517), 'H_80': (734, 515)
        }
    }
}


# ==========================================
# 2. MOTOR DE EXTRACCIÓN Y PIPELINE PRINCIPAL
# ==========================================
def procesar_experimento():
    """
    Orquesta el flujo de trabajo de visión artificial. Mantiene estados paralelos
    para cada cota espacial (Abajo, Medio, Arriba) extrayendo las lecturas de
    humedad simultáneamente mediante Single-pass I/O.
    """
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))],
                         key=lambda x: int(re.search(r'\d+', x).group()))

    if not image_files:
        print(">> ERROR: No se encontraron imágenes en el directorio especificado.")
        return

    archivos_csv = {}
    escritores_csv = {}
    offsets_dinamicos = {k: {} for k in SENSORES_CONFIG.keys()}
    ultimos_anchors = {k: None for k in SENSORES_CONFIG.keys()}

    try:
        # 2.1. Apertura concurrente de flujos de datos
        for nombre_sensor, config in SENSORES_CONFIG.items():
            f = open(config["csv_file"], mode='w', newline='')
            writer = csv.writer(f)
            writer.writerow(['Image', 'Window', 'R', 'G', 'B'])
            archivos_csv[nombre_sensor] = f
            escritores_csv[nombre_sensor] = writer

        # 2.2. Bucle principal iterativo sobre el dataset temporal
        for index, image_name in enumerate(image_files):
            image_path = os.path.join(IMAGE_FOLDER, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"  -> AVISO: Error de lectura en imagen {image_name}")
                continue

            # Procesamiento paralelo de las 3 capas
            for nombre_sensor, config in SENSORES_CONFIG.items():

                # A. Tracking: Anclaje de la esquina superior derecha del bloque de referencias
                current_anchor = config["tracker_func"](image)

                if current_anchor is None:
                    if ultimos_anchors[nombre_sensor] is not None:
                        current_anchor = ultimos_anchors[nombre_sensor]
                    else:
                        print(f"  -> AVISO: {image_name} sin anclaje para sensor cota {nombre_sensor}.")
                        continue

                ultimos_anchors[nombre_sensor] = current_anchor
                anchor_x, anchor_y = current_anchor

                # B. Auto-Calibración (Fotograma Base)
                if index == 0:
                    for window_name, (orig_x, orig_y) in config["coords"].items():
                        offset_x = orig_x - anchor_x
                        offset_y = orig_y - anchor_y
                        offsets_dinamicos[nombre_sensor][window_name] = (offset_x, offset_y)
                    print(f">> Auto-calibración completada para sensor cota {nombre_sensor}.")

                # C. Extracción Dinámica adaptativa (tamaño de ROI heterogéneo)
                current_roi_size = config["roi_size"]

                for window_name, (off_x, off_y) in offsets_dinamicos[nombre_sensor].items():
                    new_x = int(anchor_x + off_x)
                    new_y = int(anchor_y + off_y)

                    # Segmentación de la ROI según el parámetro de la cota actual
                    window_roi = image[new_y:new_y + current_roi_size, new_x:new_x + current_roi_size]

                    # Verificación de bordes para evitar excepciones de matriz vacía
                    if window_roi.size == 0:
                        continue

                    # Promediado de canales RGB
                    avg_color = cv2.mean(window_roi)[:3]
                    r_avg, g_avg, b_avg = avg_color[2], avg_color[1], avg_color[0]

                    # Serialización
                    escritores_csv[nombre_sensor].writerow([image_name, window_name, r_avg, g_avg, b_avg])

            if index % 100 == 0:
                print(f">> Procesada imagen {index}: {image_name}")

        print("\n>> EXTRACCIÓN DINÁMICA MULTI-CAPA FINALIZADA CON ÉXITO.")

    finally:
        for f in archivos_csv.values():
            f.close()


if __name__ == "__main__":
    procesar_experimento()