"""
Módulo Generador de Topología Espacial para Plantillas Sensoriales
------------------------------------------------------------------
Este script actúa como una herramienta de desarrollo (DevTool). Ingiere
coordenadas absolutas en píxeles (obtenidas mediante inspección visual, ej. MS Paint)
y aplica una transformación geométrica para convertirlas en un sistema de coordenadas
relativas basadas en centímetros. Toma como origen universal (0,0) el centro
del código QR de anclaje.

El output es un diccionario serializado listo para ser inyectado en la
configuración del lector dinámico de plantillas.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

# ==========================================
# 1. ENTRADA DE DATOS (COORDENADAS ABSOLUTAS EN PÍXELES)
# ==========================================

# --- ANCLAJE PRINCIPAL (FIDUCIAL MARKER) ---
# Esquinas del contorno del código QR. Actúa como regla de calibración física (3x3 cm)
QR_TL = (708, 412)
QR_BR = (1063, 768)

# --- CALIBRACIÓN RADIOMÉTRICA (1x1 cm) ---
REF_NEGRA_TL = (827, 240)
REF_BLANCA_TL = (680, 240)

# --- SENSOR 1: TEMPERATURA IRREVERSIBLE ---
S1_CAJA_TL = (414, 198)
S1_VENTANA_W = 20
S1_VENTANA_H = 20
S1_VENTANAS_TL = [
    (521, 675), (520, 634), (520, 587), (521, 539),
    (521, 493), (522, 446), (522, 399), (522, 352)
]

# --- SENSOR 2: TEMPERATURA REVERSIBLE ---
S2_CAJA_TL = (1123, 198)
S2_VENTANA_W = 20
S2_VENTANA_H = 20
S2_VENTANAS_TL = [
    (1227, 661), (1227, 615), (1227, 575), (1228, 533), (1228, 493),
    (1230, 451), (1230, 409), (1230, 368), (1232, 329), (1232, 296)
]

# --- SENSOR 3: HUMEDAD RELATIVA ---
S3_CAJA_TL = (508, 868)
S3_VENTANA_W = 30
S3_VENTANA_H = 30
S3_VENTANAS_TL = [
    (545, 896), (670, 897), (796, 897), (915, 897)
]

# --- SENSOR 4: TIEMPO (TIMESTRIP) ---
S4_CAJA_TL = (1418, 263)
S4_VENTANA_W = 50
S4_VENTANA_H = 120
S4_VENTANAS_TL = [
    (1510, 360)
]


# ==========================================
# 2. MOTOR DE TRANSFORMACIÓN ESPACIAL
# ==========================================
def calcular_configuracion():
    """
    Calcula el factor de escala píxel/cm y traslada todas las coordenadas
    absolutas a un espacio relativo referenciado al centro geométrico del QR.
    Imprime por consola la estructura de datos resultante.
    """
    # 1. Establecimiento de la escala física
    ancho_qr_px = QR_BR[0] - QR_TL[0]
    px_por_cm = ancho_qr_px / 3.0

    # 2. Definición del Origen (0,0) del sistema
    centro_qr_x = (QR_TL[0] + QR_BR[0]) / 2.0
    centro_qr_y = (QR_TL[1] + QR_BR[1]) / 2.0

    # 3. Transformación de referencias radiométricas
    ref_n_cx_cm = ((REF_NEGRA_TL[0] + (1.0 * px_por_cm) / 2.0) - centro_qr_x) / px_por_cm
    ref_n_cy_cm = ((REF_NEGRA_TL[1] + (1.0 * px_por_cm) / 2.0) - centro_qr_y) / px_por_cm

    ref_b_cx_cm = ((REF_BLANCA_TL[0] + (1.0 * px_por_cm) / 2.0) - centro_qr_x) / px_por_cm
    ref_b_cy_cm = ((REF_BLANCA_TL[1] + (1.0 * px_por_cm) / 2.0) - centro_qr_y) / px_por_cm

    print(">> Copiando configuracion al portapapeles... (Copia el texto de abajo y pégalo en el script principal)\n")
    print("        config_plantilla = {")
    print('            "Referencias": {')
    print(f'                "W_Ref": {{"x": {ref_b_cx_cm:.3f}, "y": {ref_b_cy_cm:.3f}}},')
    print(f'                "B_Ref": {{"x": {ref_n_cx_cm:.3f}, "y": {ref_n_cy_cm:.3f}}}')
    print('            },')
    print('            "Sensores": {')

    def procesar_sensor(nombre, caja_tl, ventanas_tl, size_w_px, size_h_px, is_last=False):
        """
        Función auxiliar para procesar iterativamente los arrays de sensores,
        transformando las cajas contenedoras y las sub-ventanas de lectura.
        """
        caja_tl_x_cm = (caja_tl[0] - centro_qr_x) / px_por_cm
        caja_tl_y_cm = (caja_tl[1] - centro_qr_y) / px_por_cm
        size_w_cm = size_w_px / px_por_cm
        size_h_cm = size_h_px / px_por_cm

        print(f'                "{nombre}": {{')
        print(f'                    "caja_tl_x": {caja_tl_x_cm:.3f},')
        print(f'                    "caja_tl_y": {caja_tl_y_cm:.3f},')
        print(f'                    "size_w": {size_w_cm:.3f},')
        print(f'                    "size_h": {size_h_cm:.3f},')
        print('                    "ventanas": [')

        for i, v_tl in enumerate(ventanas_tl):
            cx_px = v_tl[0] + (size_w_px / 2.0)
            cy_px = v_tl[1] + (size_h_px / 2.0)

            # Offset interno de cada ventana respecto al marco de su propio sensor
            offset_x_cm = (cx_px - caja_tl[0]) / px_por_cm
            offset_y_cm = (cy_px - caja_tl[1]) / px_por_cm
            coma = "," if i < len(ventanas_tl) - 1 else ""
            print(
                f'                        {{"nombre": "V_{i + 1}", "offset_x": {offset_x_cm:.3f}, "offset_y": {offset_y_cm:.3f}}}{coma}')

        print('                    ]')
        coma_externa = "," if not is_last else ""
        print(f'                }}{coma_externa}')

    # 4. Invocación dinámica de procesado por tipo de sensor
    procesar_sensor("Exp1_Irr", S1_CAJA_TL, S1_VENTANAS_TL, S1_VENTANA_W, S1_VENTANA_H)
    procesar_sensor("Exp2_Rev", S2_CAJA_TL, S2_VENTANAS_TL, S2_VENTANA_W, S2_VENTANA_H)
    procesar_sensor("Exp3_Hum", S3_CAJA_TL, S3_VENTANAS_TL, S3_VENTANA_W, S3_VENTANA_H)
    procesar_sensor("Exp4_Time", S4_CAJA_TL, S4_VENTANAS_TL, S4_VENTANA_W, S4_VENTANA_H, is_last=True)

    print('            }')
    print('        }')
    print(f"\n>> INFO: Resolución calculada de {px_por_cm:.1f} píxeles por centímetro.")


if __name__ == "__main__":
    if QR_BR != (1063, 768) or S4_CAJA_TL != (0, 0):
        calcular_configuracion()
    else:
        print("Rellena las coordenadas del bloque 1 antes de ejecutar el script.")