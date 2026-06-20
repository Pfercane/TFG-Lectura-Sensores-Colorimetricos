"""
Módulo Lector de Plantillas Sensoriales y Predicción Logística
--------------------------------------------------------------
Este script representa el núcleo operativo del sistema. Implementa una
Alineación Jerárquica con Fiduciarios Locales ("Coarse-to-Fine").
Detecta el QR central para identificar la topología y estimar las regiones
de búsqueda (Anclaje Global). Posteriormente, segmenta los recuadros
delimitadores locales de cada sensor para anular la distorsión radial de
las ópticas móviles en la periferia de la imagen (Anclaje Local).

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import numpy as np
import joblib
import os

# ==========================================
# 1. FUNCIONES AUXILIARES Y GEOMETRÍA
# ==========================================
def ordenar_puntos_cuadrilatero(pts):
    """ Ordena 4 puntos en: Top-Left, Top-Right, Bottom-Right, Bottom-Left """
    suma = pts.sum(axis=1)
    resta = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(suma)],   # TL
        pts[np.argmin(resta)],  # TR
        pts[np.argmax(suma)],   # BR
        pts[np.argmax(resta)]   # BL
    ], dtype=np.float32)

def obtener_media_rgb_cruda(img_rgb, centro_x, centro_y, size_w_cm, size_h_cm, pixeles_por_cm):
    """ Extrae la media RGB de una región de interés (ROI) controlando los bordes. """
    mitad_w = max(1, int(size_w_cm * pixeles_por_cm) // 2)
    mitad_h = max(1, int(size_h_cm * pixeles_por_cm) // 2)

    y_inicio = max(0, centro_y - mitad_h)
    y_fin = min(img_rgb.shape[0], centro_y + mitad_h)
    x_inicio = max(0, centro_x - mitad_w)
    x_fin = min(img_rgb.shape[1], centro_x + mitad_w)

    roi = img_rgb[y_inicio:y_fin, x_inicio:x_fin]
    if roi.size == 0: return np.array([0.0, 0.0, 0.0])

    return np.array([np.mean(roi[:, :, 0]), np.mean(roi[:, :, 1]), np.mean(roi[:, :, 2])])

def calibrar_canal(val_crudo, w_ref, b_ref):
    if w_ref - b_ref == 0: return 0.0
    return np.clip((255.0 / (w_ref - b_ref)) * (val_crudo - b_ref), 0, 255)

def extraer_target_feature(rgb_calibrado, tipo_sensor):
    if tipo_sensor == "Exp1_Irr":
        return (0.21 * rgb_calibrado[0] + 0.72 * rgb_calibrado[1] + 0.07 * rgb_calibrado[2]) / 255.0
    return rgb_calibrado[0] / 255.0

def proyectar_punto_homografia(x_cm, y_cm, matriz_h):
    """ Convierte una coordenada física (cm) a un píxel deformado por la perspectiva. """
    pt_pixel = cv2.perspectiveTransform(np.array([[[float(x_cm), float(y_cm)]]], dtype=np.float32), matriz_h)
    return pt_pixel[0][0]

def dibujar_roi_homografia(img, center_x_cm, center_y_cm, w_cm, h_cm, matriz_h, color):
    """ Dibuja un rectángulo que respeta la inclinación y perspectiva exacta. """
    hw, hh = w_cm / 2.0, h_cm / 2.0
    pts_fisicos = np.array([
        [center_x_cm - hw, center_y_cm - hh], [center_x_cm + hw, center_y_cm - hh],
        [center_x_cm + hw, center_y_cm + hh], [center_x_cm - hw, center_y_cm + hh]
    ], dtype=np.float32)
    pts_pixel = cv2.perspectiveTransform(np.array([pts_fisicos]), matriz_h)
    cv2.polylines(img, [np.int32(pts_pixel)], isClosed=True, color=color, thickness=2)


# ==========================================
# 2. CARGA DE MODELOS (.PKL)
# ==========================================
ruta_modelos = os.path.join("..", "Modelado_Predictivo", "Modelos_Exportados")
modelos = {}
try:
    modelos['Exp1'] = joblib.load(os.path.join(ruta_modelos, 'modelo_random_forest_exp1.pkl'))
    modelos['Exp2'] = joblib.load(os.path.join(ruta_modelos, 'modelo_random_forest_exp2.pkl'))
    modelos['Exp3'] = joblib.load(os.path.join(ruta_modelos, 'modelo_random_forest_exp3.pkl'))
    modelos['Exp4'] = joblib.load(os.path.join(ruta_modelos, 'modelo_polinomico_exp4.pkl'))
    print(">> Modelos predictivos vinculados con éxito.\n")
except FileNotFoundError:
    print(">> AVISO: Faltan modelos predictivos. Modo solo extracción activado.\n")


# ==========================================
# 3. MOTOR PRINCIPAL DE EXTRACCIÓN
# ==========================================
ruta_imagen = os.path.join("Plantillas", "30C_40RH.jpeg")

if not os.path.exists(ruta_imagen):
    print(f">> ERROR: No se encontró la imagen en {ruta_imagen}")
    exit()

img_bgr = cv2.imread(ruta_imagen)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_debug = img_bgr.copy()

detector = cv2.QRCodeDetector()
datos_qr, bbox, _ = detector.detectAndDecode(img_bgr)

if bbox is None or not datos_qr:
    print(">> ERROR: No se ha detectado un anclaje QR válido en la imagen.")
    exit()

plantilla_id = datos_qr[:2]

# 3.1 Procesamiento del QR Central (Anclaje Global)
pts_qr = ordenar_puntos_cuadrilatero(bbox[0])
centro_qr = np.mean(pts_qr, axis=0)

cv2.circle(img_debug, (int(centro_qr[0]), int(centro_qr[1])), 5, (0, 0, 255), -1)
cv2.polylines(img_debug, [np.int32(pts_qr)], True, (0, 255, 255), 3)

escala_global_px_cm = np.linalg.norm(pts_qr[1] - pts_qr[0]) / 2.4  # tr - tl
config_plantilla = {}

# 3.2 Carga dinámica de la Topología Fiduciaria
if plantilla_id == "01":
    print(f">> PROCESANDO PLANTILLA 01 (Paquete ID: {datos_qr[2:]})")
    config_plantilla = {
        "Referencias": {
            "W_Ref": {"x": -1.237, "y": -2.458}, "B_Ref": {"x": 0.006, "y": -2.458}
        },
        "Sensores": {
            "Exp1_Irr": {
                "caja_tl_x": -3.985, "caja_tl_y": -3.313, "caja_w_real": 2.0, "caja_h_real": 5.3, "size_w": 0.169, "size_h": 0.169,
                "ventanas": [{"nombre": f"V_{i+1}", "offset_x": x, "offset_y": y} for i, (x, y) in enumerate([
                    (0.989, 4.115), (0.980, 3.769), (0.980, 3.372), (0.989, 2.966),
                    (0.989, 2.577), (0.997, 2.180), (0.997, 1.783), (0.997, 1.386)
                ])]
            },
            "Exp2_Rev": {
                "caja_tl_x": 2.007, "caja_tl_y": -3.313, "caja_w_real": 2.0, "caja_h_real": 5.3, "size_w": 0.169, "size_h": 0.169,
                "ventanas": [{"nombre": f"V_{i+1}", "offset_x": x, "offset_y": y} for i, (x, y) in enumerate([
                    (0.963, 3.997), (0.963, 3.608), (0.963, 3.270), (0.972, 2.915), (0.972, 2.577),
                    (0.989, 2.223), (0.989, 1.868), (0.989, 1.521), (1.006, 1.192), (1.006, 0.913)
                ])]
            },
            "Exp3_Hum": {
                "caja_tl_x": -3.190, "caja_tl_y": 2.349, "caja_w_real": 6.6, "caja_h_real": 0.7, "size_w": 0.254, "size_h": 0.254,
                "ventanas": [{"nombre": f"V_{i+1}", "offset_x": x, "offset_y": y} for i, (x, y) in enumerate([
                    (0.439, 0.363), (1.496, 0.372), (2.561, 0.372), (3.566, 0.372)
                ])]
            }
        }
    }

elif plantilla_id == "02":
    print(f">> PROCESANDO PLANTILLA 02 (Paquete ID: {datos_qr[2:]})")
    config_plantilla = {
        "Referencias": {
            "W_Ref": {"x": -1.248, "y": -2.419},
            "B_Ref": {"x": -0.008, "y": -2.419}
        },
        "Sensores": {
            "Exp1_Irr": {
                "caja_tl_x": -3.992, "caja_tl_y": -3.304, "caja_w_real": 2.0, "caja_h_real": 5.3, "size_w": 0.093,
                "size_h": 0.093,
                "ventanas": [{"nombre": f"V_{i + 1}", "offset_x": x, "offset_y": y} for i, (x, y) in enumerate([
                    (0.970, 4.137), (0.970, 3.809), (0.970, 3.425), (0.970, 3.034),
                    (0.970, 2.662), (0.964, 2.278), (0.964, 1.881), (0.964, 1.497)
                ])]
            },
            "Exp2_Rev": {
                "caja_tl_x": 1.977, "caja_tl_y": -3.298, "caja_w_real": 2.0, "caja_h_real": 5.3, "size_w": 0.093,
                "size_h": 0.093,
                "ventanas": [{"nombre": f"V_{i + 1}", "offset_x": x, "offset_y": y} for i, (x, y) in enumerate([
                    (1.032, 3.908), (1.026, 3.524), (1.026, 3.201), (1.026, 2.873),
                    (1.026, 2.526), (1.026, 2.210), (1.026, 1.856), (1.026, 1.522),
                    (1.026, 1.193), (1.026, 0.939)
                ])]
            },
            "Exp3_Hum": {
                "caja_tl_x": -3.136, "caja_tl_y": 2.306, "caja_w_real": 6.6, "caja_h_real": 0.7, "size_w": 0.124,
                "size_h": 0.124,
                "ventanas": [{"nombre": f"V_{i + 1}", "offset_x": x, "offset_y": y} for i, (x, y) in enumerate([
                    (0.471, 0.353), (1.488, 0.360), (2.498, 0.366), (3.465, 0.353)
                ])]
            },
            "Exp4_Time": {
                "caja_tl_x": 4.469, "caja_tl_y": -2.758, "caja_w_real": 2.1, "caja_h_real": 4.2, "size_w": 0.310,
                "size_h": 0.992,
                "ventanas": [{"nombre": f"V_{i + 1}", "offset_x": x, "offset_y": y} for i, (x, y) in enumerate([
                    (1.010, 1.444)
                ])]
            }
        }
    }

if config_plantilla:
    # 3.3 Generación de la Matriz Homográfica Global (Fallback)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    M_homografia_global = None

    # Intenta encontrar la caja externa del QR (3x3)
    for c in contours:
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), _ = rect
        if w > 0 and h > 0 and np.linalg.norm(np.array([cx, cy]) - centro_qr) < (max(w, h) * 0.2):
            if min(w, h) / max(w, h) > 0.85 and max(w, h) > (np.linalg.norm(pts_qr[1] - pts_qr[0]) * 1.05):
                pts_caja = ordenar_puntos_cuadrilatero(cv2.boxPoints(rect))
                src_pts = np.array([[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]], dtype=np.float32)
                M_homografia_global = cv2.getPerspectiveTransform(src_pts, pts_caja)
                break

    # Fallback al QR interno (2.4x2.4) si no encuentra la caja
    if M_homografia_global is None:
        src_pts = np.array([[-1.2, -1.2], [1.2, -1.2], [1.2, 1.2], [-1.2, 1.2]], dtype=np.float32)
        M_homografia_global = cv2.getPerspectiveTransform(src_pts, pts_qr)

    # 3.4 Extracción de Referencias Globales (Luz)
    rgb_refs = {}
    for nombre, params in config_plantilla["Referencias"].items():
        pos_pixel = proyectar_punto_homografia(params["x"], params["y"], M_homografia_global)
        rgb_refs[nombre] = obtener_media_rgb_cruda(img_rgb, int(pos_pixel[0]), int(pos_pixel[1]), 0.75, 0.75, escala_global_px_cm)
        dibujar_roi_homografia(img_debug, params["x"], params["y"], 0.75, 0.75, M_homografia_global, (0, 255, 0))
    w_rgb, b_rgb = rgb_refs.get("W_Ref", [0,0,0]), rgb_refs.get("B_Ref", [0,0,0])

    # 3.5 Inferencia por Anclaje Local
    vectores_finales = {}
    for tipo_sensor, config in config_plantilla["Sensores"].items():
        vector_sensor = []

        centro_sensor_x = config["caja_tl_x"] + (config["caja_w_real"] / 2.0)
        centro_sensor_y = config["caja_tl_y"] + (config["caja_h_real"] / 2.0)
        centro_estimado_px = proyectar_punto_homografia(centro_sensor_x, centro_sensor_y, M_homografia_global)

        caja_local_encontrada = False
        M_homografia_local = M_homografia_global
        escala_local_px_cm = escala_global_px_cm
        ratio_esperado = max(config["caja_w_real"], config["caja_h_real"]) / min(config["caja_w_real"], config["caja_h_real"])

        mejor_candidato = None
        menor_diferencia_ratio = float('inf')

        # Búsqueda de la caja local (Segmentación)
        for c in contours:
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), _ = rect
            if w == 0 or h == 0: continue

            if np.linalg.norm(np.array([cx, cy]) - centro_estimado_px) < (max(w, h) * 0.8):
                error_ratio = abs((max(w, h) / min(w, h)) - ratio_esperado) / ratio_esperado
                if error_ratio < 0.30 and error_ratio < menor_diferencia_ratio:
                    menor_diferencia_ratio = error_ratio
                    mejor_candidato = rect

        # Anclaje Local (Cálculo de Matriz Fina)
        if mejor_candidato is not None:
            pts_local = ordenar_puntos_cuadrilatero(cv2.boxPoints(mejor_candidato))

            src_w = max(config["caja_w_real"], config["caja_h_real"]) if config["caja_w_real"] > config["caja_h_real"] else min(config["caja_w_real"], config["caja_h_real"])
            src_h = min(config["caja_w_real"], config["caja_h_real"]) if config["caja_w_real"] > config["caja_h_real"] else max(config["caja_w_real"], config["caja_h_real"])

            src_pts_local = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)

            try:
                M_homografia_local = cv2.getPerspectiveTransform(src_pts_local, pts_local)
                escala_local_px_cm = np.linalg.norm(pts_local[1] - pts_local[0]) / src_w # tr - tl
                caja_local_encontrada = True
                cv2.polylines(img_debug, [np.int32(pts_local)], True, (255, 0, 255), 3)
            except: pass

        if not caja_local_encontrada:
            print(f">> [AVISO] Recuadro local no hallado para {tipo_sensor}. Fallback a perspectiva global.")

        # Extracción Colorimétrica Final
        for win in config["ventanas"]:
            if caja_local_encontrada:
                x_local, y_local = win["offset_x"], win["offset_y"]
                pos_pixel = proyectar_punto_homografia(x_local, y_local, M_homografia_local)
                dibujar_roi_homografia(img_debug, x_local, y_local, config["size_w"], config["size_h"], M_homografia_local, (255, 0, 0))
            else:
                x_global, y_global = config["caja_tl_x"] + win["offset_x"], config["caja_tl_y"] + win["offset_y"]
                pos_pixel = proyectar_punto_homografia(x_global, y_global, M_homografia_global)
                dibujar_roi_homografia(img_debug, x_global, y_global, config["size_w"], config["size_h"], M_homografia_global, (255, 0, 0))

            rgb_crudo = obtener_media_rgb_cruda(img_rgb, int(pos_pixel[0]), int(pos_pixel[1]), config["size_w"], config["size_h"], escala_local_px_cm)
            r_cal = calibrar_canal(rgb_crudo[0], w_rgb[0], b_rgb[0])
            g_cal = calibrar_canal(rgb_crudo[1], w_rgb[1], b_rgb[1])
            b_cal = calibrar_canal(rgb_crudo[2], w_rgb[2], b_rgb[2])

            vector_sensor.append(extraer_target_feature(np.array([r_cal, g_cal, b_cal]), tipo_sensor))

        vectores_finales[tipo_sensor] = vector_sensor

    # ==========================================
    # 4. PREDICCIÓN Y SALIDA DE DATOS
    # ==========================================
    cv2.imwrite('debug_vision.jpg', img_debug)
    print("\n>> INFO: Artefacto de depuración generado ('debug_vision.jpg').")
    print("\n>> RESULTADOS DE LA PREDICCIÓN LOGÍSTICA:")

    if "Exp1_Irr" in vectores_finales and len(vectores_finales["Exp1_Irr"]) > 0 and modelos.get('Exp1'):
        print(f"   [Sensor 1] Temp Irreversible: {modelos['Exp1'].predict([vectores_finales['Exp1_Irr']])[0]:.1f} °C")
    if "Exp2_Rev" in vectores_finales and len(vectores_finales["Exp2_Rev"]) > 0 and modelos.get('Exp2'):
        print(f"   [Sensor 2] Temp Reversible:   {modelos['Exp2'].predict([vectores_finales['Exp2_Rev']])[0]:.1f} °C")
    if "Exp3_Hum" in vectores_finales and len(vectores_finales["Exp3_Hum"]) > 0 and modelos.get('Exp3'):
        print(f"   [Sensor 3] Humedad Relativa:  {modelos['Exp3'].predict([vectores_finales['Exp3_Hum']])[0]:.1f} %")
    if "Exp4_Time" in vectores_finales and len(vectores_finales["Exp4_Time"]) > 0 and modelos.get('Exp4'):
        print(f"   [Sensor 4] Días en Servicio:  {modelos['Exp4'].predict([vectores_finales['Exp4_Time']])[0]:.1f} Días")
