"""
Módulo de Extracción Colorimétrica Estática Multi-Capa (Experimento 3)
----------------------------------------------------------------------
Este script unificado procesa simultáneamente un array vertical de sensores
de humedad relativa ubicados en tres cotas distintas (Inferior, Media, Superior).

Tras auditar la física del ensayo, se determinó que la condensación extrema
(>80% HR) invalidaba el uso de anclajes fiduciales dinámicos por reflexión
especular. Por tanto, el sistema emplea una arquitectura de extracción
estática por coordenadas absolutas, garantizando inmunidad total frente
al ruido óptico del vapor de agua.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import os
import csv
import re

# ==========================================
# 1. CONFIGURACIÓN TOPOLÓGICA DE LOS SENSORES
# ==========================================
IMAGE_FOLDER = "Imagenes"
DATOS_FOLDER = "Datos"

os.makedirs(DATOS_FOLDER, exist_ok=True)

# Coordenadas estáticas absolutas ajustadas manualmente.
SENSORES_CONFIG = {
    "Abajo": {
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_3_bot.csv"),
        "roi_size": 40,
        "coords": {
            'W_Ref': (248, 347), 'B_Ref': (140, 260),
            'H_20': (362, 172), 'H_40': (487, 173), 'H_60': (613, 173), 'H_80': (732, 173)
        }
    },
    "Medio": {
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_3_mid.csv"),
        "roi_size": 40,
        "coords": {
            'W_Ref': (248, 347), 'B_Ref': (140, 260), # ¡Ajustado a los buenos!
            'H_20': (365, 353), 'H_40': (492, 352), 'H_60': (615, 352), 'H_80': (736, 351)
        }
    },
    "Arriba": {
        "csv_file": os.path.join(DATOS_FOLDER, "datos_color_3_up.csv"),
        "roi_size": 20,
        "coords": {
            'W_Ref': (248, 347), 'B_Ref': (140, 260), # ¡Ajustado a los buenos!
            'H_20': (364, 520), 'H_40': (492, 518), 'H_60': (613, 517), 'H_80': (734, 515)
        }
    }
}

# ==========================================
# 2. MOTOR DE EXTRACCIÓN (ESTÁTICA)
# ==========================================
def procesar_experimento():
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))],
                         key=lambda x: int(re.search(r'\d+', x).group()))

    if not image_files:
        print(">> ERROR: No se encontraron imágenes en el directorio especificado.")
        return

    archivos_csv = {}
    escritores_csv = {}

    try:
        # Apertura de los 3 CSV a la vez
        for nombre_sensor, config in SENSORES_CONFIG.items():
            f = open(config["csv_file"], mode='w', newline='')
            writer = csv.writer(f)
            writer.writerow(['Image', 'Window', 'R', 'G', 'B'])
            archivos_csv[nombre_sensor] = f
            escritores_csv[nombre_sensor] = writer

        # Bucle ultrarrápido sin tracker
        for index, image_name in enumerate(image_files):
            image_path = os.path.join(IMAGE_FOLDER, image_name)
            image = cv2.imread(image_path)

            if image is None: continue

            for nombre_sensor, config in SENSORES_CONFIG.items():
                current_roi_size = config["roi_size"]

                for window_name, (tl_x, tl_y) in config["coords"].items():
                    # Recorte directo usando la coordenada absoluta
                    window_roi = image[tl_y:tl_y + current_roi_size, tl_x:tl_x + current_roi_size]

                    if window_roi.size == 0: continue

                    # Media RGB
                    avg_color = cv2.mean(window_roi)[:3]
                    r_avg, g_avg, b_avg = avg_color[2], avg_color[1], avg_color[0]

                    escritores_csv[nombre_sensor].writerow([image_name, window_name, r_avg, g_avg, b_avg])

            if index % 100 == 0:
                print(f">> Procesada imagen {index}: {image_name}")

        print("\n>> EXTRACCIÓN ESTÁTICA MULTI-CAPA FINALIZADA CON ÉXITO.")

    finally:
        for f in archivos_csv.values():
            f.close()

if __name__ == "__main__":
    procesar_experimento()