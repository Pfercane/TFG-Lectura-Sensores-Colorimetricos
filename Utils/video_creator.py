"""
Módulo de Diagnóstico Visual Rápido (Generador de Time-Lapse)
-------------------------------------------------------------
Esta herramienta auxiliar permite transformar secuencias masivas de imágenes
extraídas de los ensayos en vídeo comprimido (MP4). Facilita la auditoría
visual rápida de las pruebas de laboratorio, permitiendo al investigador
verificar la integridad mecánica del ensayo (movimientos, fallos de luz o
anomalías físicas) sin necesidad de procesar miles de archivos individuales.

El script incorpora decimación espacial (frame skipping) y cálculo dinámico
de fotogramas por segundo (FPS) basado en una duración objetivo constante.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import os

# ==========================================
# 1. CONFIGURACIÓN DE HERRAMIENTA
# ==========================================
folder_name = "D:/TFG/test_TFG_2025-11-17_13-17-34"
output_video = "D:/TFG/test_TFG_2025-11-17_13-17-34.mp4"

# Segmentación del dataset (Rango de fotogramas a compilar)
start_index = 0
end_index = 2849

# Parámetro de escalado de tiempo (El algoritmo ajustará los FPS para cumplir esta duración)
desired_duration_seconds = 20

# ==========================================
# 2. VALIDACIÓN E INGESTA DEL DATASET
# ==========================================
if not os.path.exists(folder_name):
    print(f">> ERROR: El directorio de origen '{folder_name}' no existe.")
else:
    # Extracción y ordenación natural (numérica) de las imágenes JPEG
    image_files = sorted(
        [f for f in os.listdir(folder_name) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0])
    )

    # Aplicación de máscara de rango (Filtro espacial)
    image_files = [f for f in image_files if start_index <= int(f.split(".")[0]) <= end_index]

    # Decimación temporal (Frame Skipping). Un paso de [::1] procesa todo.
    # Un paso de [::2] descarta la mitad de las imágenes (aumentando la velocidad aparente).
    image_files = image_files[::1]

    if not image_files:
        print(f">> AVISO: No se encontraron fotogramas válidos en el rango [{start_index} - {end_index}].")
    else:
        # ==========================================
        # 3. CÁLCULO DINÁMICO DE RENDERIZADO
        # ==========================================
        num_frames = len(image_files)
        # Adaptación del framerate para forzar la duración objetivo del vídeo
        fps = num_frames / desired_duration_seconds

        # Extracción de resolución geométrica desde el primer fotograma
        first_image_path = os.path.join(folder_name, image_files[0])
        first_image = cv2.imread(first_image_path)
        height, width, _ = first_image.shape

        print(f">> Compilando {num_frames} fotogramas a {fps:.2f} FPS...")

        # ==========================================
        # 4. COMPRESIÓN Y CODIFICACIÓN (VIDEO WRITER)
        # ==========================================
        # Se emplea el códec 'mp4v' para garantizar compatibilidad universal
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for idx, image_name in enumerate(image_files):
            image_path = os.path.join(folder_name, image_name)
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"  -> AVISO: Error de lectura en {image_name}. Saltando trama.")
                continue

            video_writer.write(frame)

            if idx % 500 == 0:
                print(f"  -> Renderizando trama {idx}/{num_frames}...")

        # ==========================================
        # 5. FINALIZACIÓN Y CIERRE DE FLUJOS
        # ==========================================
        video_writer.release()
        print(f"\n>> ÉXITO: Diagnóstico visual generado en '{output_video}'.")
        print(f">> Duración teórica: {desired_duration_seconds} segundos.")