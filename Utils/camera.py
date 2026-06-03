"""
Módulo de Adquisición de Datos y Telemetría Visual (Time-Lapse)
---------------------------------------------------------------
Este script implementa el motor de captura de imágenes ininterrumpida para los
ensayos longitudinales realizados en la cámara climática. Gestiona la conexión
con el hardware de visión (cámara USB), automatiza la creación de directorios
etiquetados por fecha y hora, y ejecuta un bucle de muestreo a frecuencia
constante (1 Hz) para generar el dataset fotográfico en bruto.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import os
from datetime import datetime
import time

# ==========================================
# 1. INICIALIZACIÓN DEL ENTORNO DE CAPTURA
# ==========================================
# Generación de una marca de tiempo absoluta para trazabilidad del experimento
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Definición y creación automática del directorio de volcado (Dataset)
folder_name = f"./test_TFG_{current_datetime}"
os.makedirs(folder_name, exist_ok=True)
print(f">> Sesión de captura iniciada. Directorio de trabajo: {folder_name}")

# ==========================================
# 2. CONEXIÓN E INTERFAZ DE HARDWARE
# ==========================================
# Instanciación del objeto VideoCapture (Índice 0 = Cámara USB predeterminada)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print(">> ERROR CRÍTICO: No se pudo establecer conexión con la cámara.")
else:
    counter = 0  # Índice secuencial para nomenclatura de fotogramas
    max_frames = 5000  # Límite de seguridad para evitar desbordamiento de disco

    print(">> Grabación en curso... Pulse 'q' en la terminal para abortar.")

    # ==========================================
    # 3. BUCLE DE MUESTREO (SAMPLING LOOP)
    # ==========================================
    while counter < max_frames:
        # Petición de lectura de un fotograma en el búfer de hardware
        ret, frame = camera.read()

        if ret:
            # Serialización en disco del fotograma capturado
            filename = os.path.join(folder_name, f"{counter}.jpg")
            cv2.imwrite(filename, frame)
        else:
            print(f">> AVISO: Pérdida de trama en la iteración {counter}.")

        # Mantenimiento de la frecuencia de muestreo (1 fotograma / segundo)
        time.sleep(1)
        counter += 1

        # Interrupción manual de seguridad por teclado
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(">> Interrupción manual detectada.")
            break

# ==========================================
# 4. CIERRE Y LIBERACIÓN DE RECURSOS
# ==========================================
print(f">> Adquisición finalizada. Total fotogramas: {counter}")
camera.release()
cv2.destroyAllWindows()
