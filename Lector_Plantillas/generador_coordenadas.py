"""
Módulo Generador de Topología Espacial para Plantillas Sensoriales
------------------------------------------------------------------
Este script actúa como una herramienta de desarrollo (DevTool). Ingiere
coordenadas absolutas en píxeles (obtenidas mediante inspección visual, ej. MS Paint)
y aplica una transformación geométrica para convertirlas en un sistema de coordenadas
relativas basadas en centímetros. Toma como origen universal (0,0) el centro
del código QR de anclaje.

El output es un diccionario serializado y optimizado, listo para ser inyectado en la
configuración del lector dinámico de plantillas.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

# ==========================================
# 1. ENTRADA DE DATOS (COORDENADAS ABSOLUTAS EN PÍXELES Y TAMAÑOS REALES)
# ==========================================

# --- ANCLAJE PRINCIPAL (FIDUCIAL MARKER) ---
# Esquinas del contorno del código QR. Actúa como regla de calibración física (3x3 cm)
QR_TL = (645, 405)
QR_BR = (1129, 887)

# --- CALIBRACIÓN RADIOMÉTRICA (1x1 cm) ---
REF_NEGRA_TL = (805, 175)
REF_BLANCA_TL = (605, 175)

# --- SENSOR 1: TEMPERATURA IRREVERSIBLE ---
S1_CAJA_TL = (243, 113)
S1_CAJA_W_REAL = 2.0
S1_CAJA_H_REAL = 5.3
S1_VENTANA_W = 15
S1_VENTANA_H = 15
S1_VENTANAS_TL = [
    (392, 773), (392, 720), (392, 658), (392, 595),
    (392, 535), (391, 473), (391, 409), (391, 347)
]

# --- SENSOR 2: TEMPERATURA REVERSIBLE ---
S2_CAJA_TL = (1206, 114)
S2_CAJA_W_REAL = 2.0
S2_CAJA_H_REAL = 5.3
S2_VENTANA_W = 15
S2_VENTANA_H = 15
S2_VENTANAS_TL = [
    (1365, 737), (1364, 675), (1364, 623), (1364, 570), (1364, 514),
    (1364, 463), (1364, 406), (1364, 352), (1364, 299), (1364, 258)
]

# --- SENSOR 3: HUMEDAD RELATIVA ---
S3_CAJA_TL = (381, 1018)
S3_CAJA_W_REAL = 6.6
S3_CAJA_H_REAL = 0.7
S3_VENTANA_W = 20
S3_VENTANA_H = 20
S3_VENTANAS_TL = [
    (447, 1065), (611, 1066), (774, 1067), (930, 1065)
]

# --- SENSOR 4: TIEMPO (TIMESTRIP) ---
S4_CAJA_TL = (1608, 201)
S4_CAJA_W_REAL = 2.1
S4_CAJA_H_REAL = 4.2
S4_VENTANA_W = 50
S4_VENTANA_H = 160
S4_VENTANAS_TL = [
    (1746, 354)
]


# ==========================================
# 2. MOTOR DE TRANSFORMACIÓN ESPACIAL
# ==========================================
def calcular_configuracion():
    """
    Calcula el factor de escala píxel/cm y traslada todas las coordenadas
    absolutas a un espacio relativo referenciado al centro geométrico del QR.
    Imprime por consola la estructura de datos resultante en formato compacto.
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

    def procesar_sensor(nombre, caja_tl, ventanas_tl, size_w_px, size_h_px, w_real, h_real, is_last=False):
        caja_tl_x_cm = (caja_tl[0] - centro_qr_x) / px_por_cm
        caja_tl_y_cm = (caja_tl[1] - centro_qr_y) / px_por_cm
        size_w_cm = size_w_px / px_por_cm
        size_h_cm = size_h_px / px_por_cm

        print(f'                "{nombre}": {{')
        print(f'                    "caja_tl_x": {caja_tl_x_cm:.3f}, "caja_tl_y": {caja_tl_y_cm:.3f}, "caja_w_real": {w_real:.1f}, "caja_h_real": {h_real:.1f}, "size_w": {size_w_cm:.3f}, "size_h": {size_h_cm:.3f},')
        print('                    "ventanas": [{"nombre": f"V_{i+1}", "offset_x": x, "offset_y": y} for i, (x, y) in enumerate([')

        # Agrupamos las coordenadas en bloques para que quede elegante en el código
        offsets = []
        for v_tl in ventanas_tl:
            cx_px = v_tl[0] + (size_w_px / 2.0)
            cy_px = v_tl[1] + (size_h_px / 2.0)
            offset_x_cm = (cx_px - caja_tl[0]) / px_por_cm
            offset_y_cm = (cy_px - caja_tl[1]) / px_por_cm
            offsets.append(f"({offset_x_cm:.3f}, {offset_y_cm:.3f})")

        # Imprimimos de 4 en 4 para mantener la legibilidad
        for i in range(0, len(offsets), 4):
            chunk = ", ".join(offsets[i:i+4])
            coma_final = "," if i + 4 < len(offsets) else ""
            print(f'                        {chunk}{coma_final}')

        print('                    ])]')
        coma_externa = "," if not is_last else ""
        print(f'                }}{coma_externa}')

    # 4. Invocación dinámica de procesado por tipo de sensor
    procesar_sensor("Exp1_Irr", S1_CAJA_TL, S1_VENTANAS_TL, S1_VENTANA_W, S1_VENTANA_H, S1_CAJA_W_REAL, S1_CAJA_H_REAL)
    procesar_sensor("Exp2_Rev", S2_CAJA_TL, S2_VENTANAS_TL, S2_VENTANA_W, S2_VENTANA_H, S2_CAJA_W_REAL, S2_CAJA_H_REAL)
    procesar_sensor("Exp3_Hum", S3_CAJA_TL, S3_VENTANAS_TL, S3_VENTANA_W, S3_VENTANA_H, S3_CAJA_W_REAL, S3_CAJA_H_REAL)
    procesar_sensor("Exp4_Time", S4_CAJA_TL, S4_VENTANAS_TL, S4_VENTANA_W, S4_VENTANA_H, S4_CAJA_W_REAL, S4_CAJA_H_REAL, is_last=True)

    print('            }')
    print('        }')
    print(f"\n>> INFO: Resolución calculada de {px_por_cm:.1f} píxeles por centímetro.")


if __name__ == "__main__":
    calcular_configuracion()