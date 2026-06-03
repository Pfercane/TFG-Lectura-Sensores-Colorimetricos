"""
Módulo de Extracción Colorimétrica Dinámica (Experimento 1)
-----------------------------------------------------------
Este script unificado procesa simultáneamente múltiples sensores (izquierdo y derecho)
presentes en una misma secuencia de imágenes. Utiliza algoritmos de tracking visual
para calcular anclajes dinámicos, compensando el micromovimiento mecánico de la cámara
climática, y extrae los valores RGB de cada Región de Interés (ROI).

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import os
import csv
import re
from sensor_tracker import get_left_sensor_anchor, get_right_sensor_anchor

# ==========================================
# 1. CONFIGURACIÓN TOPOLÓGICA DE LOS SENSORES
# ==========================================
# Rutas relativas: Asumen que el script se ejecuta desde la carpeta Experimento_1
IMAGE_FOLDER = "Imagenes"
DATOS_FOLDER = "Datos"

# Nos aseguramos de que la carpeta Datos exista (si no, la crea automáticamente)
os.makedirs(DATOS_FOLDER, exist_ok=True)

ROI_SIZE = 20  # Dimensión de las ventanas de extracción (20x20 píxeles)

SENSORES_CONFIG = {
    "Izquierdo": {
        # Usamos os.path.join para unir "Datos" y el nombre del archivo de forma segura
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_1_left.csv"),
        "tracker_func": get_left_sensor_anchor,
        "coords": {
            'W_Ref': (541, 200), 'B_Ref': (541, 305),
            'Temp_40': (334, 492), 'Temp_44': (334, 450), 'Temp_46': (334, 402),
            'Temp_49': (335, 353), 'Temp_54': (335, 306), 'Temp_60': (336, 258),
            'Temp_62': (336, 210), 'Temp_65': (336, 163), 'Temp_71': (336, 115)
        }
    },
    "Derecho": {
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_1_right.csv"),
        "tracker_func": get_right_sensor_anchor,
        "coords": {
            'W_Ref': (541, 200), 'B_Ref': (541, 305),
            'Temp_40': (751, 494), 'Temp_44': (751, 451), 'Temp_46': (751, 404),
            'Temp_49': (751, 355), 'Temp_54': (751, 308), 'Temp_60': (751, 260),
            'Temp_62': (751, 212), 'Temp_65': (751, 165), 'Temp_71': (751, 118)
        }
    }
}


# ==========================================
# 2. MOTOR DE EXTRACCIÓN Y PIPELINE PRINCIPAL
# ==========================================
def procesar_experimento():
    """
    Orquesta el flujo de trabajo de visión artificial iterando sobre el dataset
    de imágenes. Mantiene el estado de los anclajes dinámicos y gestiona la
    escritura concurrente de los resultados en múltiples archivos CSV.
    """
    # Obtención de la lista de imágenes con ordenación natural (numérica)
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))],
                         key=lambda x: int(re.search(r'\d+', x).group()))

    if not image_files:
        print(">> ERROR: No se encontraron imágenes en el directorio especificado.")
        return

    # Estructuras de estado para gestionar ambos sensores simultáneamente
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

        # 2.2. Bucle principal de procesado por fotograma (Single-pass I/O)
        for index, image_name in enumerate(image_files):
            image_path = os.path.join(IMAGE_FOLDER, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"  -> AVISO: Error de lectura en imagen {image_name}")
                continue

            # Procesamos iterativamente cada sensor definido en la configuración
            for nombre_sensor, config in SENSORES_CONFIG.items():

                # A. Tracking: Localización espacial del fiducial marker en el frame actual
                current_anchor = config["tracker_func"](image)

                # Fallback: Tolerancia a fallos de oclusión usando el último frame válido
                if current_anchor is None:
                    if ultimos_anchors[nombre_sensor] is not None:
                        current_anchor = ultimos_anchors[nombre_sensor]
                    else:
                        print(f"  -> AVISO: {image_name} sin anclaje para sensor {nombre_sensor}.")
                        continue

                ultimos_anchors[nombre_sensor] = current_anchor
                anchor_x, anchor_y = current_anchor

                # B. Auto-Calibración Inicial (Frame 0)
                # Establece las distancias relativas entre el ancla detectada y las ventanas
                if index == 0:
                    for window_name, (orig_x, orig_y) in config["coords"].items():
                        offset_x = orig_x - anchor_x
                        offset_y = orig_y - anchor_y
                        offsets_dinamicos[nombre_sensor][window_name] = (offset_x, offset_y)
                    print(f">> Auto-calibración completada para sensor {nombre_sensor}.")

                # C. Extracción Radiométrica Dinámica
                for window_name, (off_x, off_y) in offsets_dinamicos[nombre_sensor].items():
                    # Proyección geométrica aplicando los offsets al ancla actual
                    new_x = anchor_x + off_x
                    new_y = anchor_y + off_y

                    # Segmentación de la ROI (20x20)
                    window_roi = image[new_y:new_y + ROI_SIZE, new_x:new_x + ROI_SIZE]

                    # Extracción y promedio del espacio de color RGB
                    avg_color = cv2.mean(window_roi)[:3]  # Retorna (B, G, R)
                    r_avg, g_avg, b_avg = avg_color[2], avg_color[1], avg_color[0]

                    # Serialización del dato
                    escritores_csv[nombre_sensor].writerow([image_name, window_name, r_avg, g_avg, b_avg])

            if index % 100 == 0:
                print(f">> Procesada imagen {index}: {image_name}")

        print("\n>> EXTRACCIÓN DINÁMICA MULTI-SENSOR FINALIZADA CON ÉXITO.")

    finally:
        # 2.3. Cierre seguro de flujos de datos
        for f in archivos_csv.values():
            f.close()


if __name__ == "__main__":
    procesar_experimento()