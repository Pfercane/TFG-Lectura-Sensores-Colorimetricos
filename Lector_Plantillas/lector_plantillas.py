"""
Módulo Lector de Plantillas Sensoriales y Predicción Logística
--------------------------------------------------------------
Este script representa el núcleo operativo del sistema. Detecta de forma
autónoma códigos QR en imágenes no controladas, decodifica su ID para cargar
dinámicamente la topología del sensor (plantilla), compensa la perspectiva y
escala de la fotografía mediante álgebra vectorial, extrae las características
colorimétricas, y emplea modelos de Machine Learning pre-entrenados para
predecir variables de entorno (Temperatura, Humedad, Tiempo).

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import cv2
import numpy as np
import joblib
import os


# ==========================================
# 1. FUNCIONES DE PROCESAMIENTO Y CALIBRACIÓN
# ==========================================
def obtener_media_rgb_cruda(img_rgb, centro_x, centro_y, size_w_cm, size_h_cm, pixeles_por_cm):
    """
    Extrae la media de los canales RGB de una región de interés (ROI) específica.

    Args:
        img_rgb (numpy.ndarray): Imagen en espacio de color RGB.
        centro_x, centro_y (int): Coordenadas del centro geométrico de la ROI en píxeles.
        size_w_cm, size_h_cm (float): Dimensiones físicas de la ventana de lectura en cm.
        pixeles_por_cm (float): Factor de escala dinámico de la imagen actual.

    Returns:
        numpy.ndarray: Vector [R, G, B] con los valores promediados.
    """
    mitad_w = max(1, int(size_w_cm * pixeles_por_cm) // 2)
    mitad_h = max(1, int(size_h_cm * pixeles_por_cm) // 2)
    roi = img_rgb[centro_y - mitad_h: centro_y + mitad_h, centro_x - mitad_w: centro_x + mitad_w]

    media_r = np.mean(roi[:, :, 0])
    media_g = np.mean(roi[:, :, 1])
    media_b = np.mean(roi[:, :, 2])
    return np.array([media_r, media_g, media_b])


def calibrar_canal(val_crudo, w_ref, b_ref):
    """
    Aplica calibración radiométrica a un canal de color para compensar la iluminancia.
    Escala el valor bruto basándose en las referencias de blanco y negro absolutas del entorno.
    """
    if w_ref - b_ref == 0: return 0.0
    calibrado = (255.0 / (w_ref - b_ref)) * (val_crudo - b_ref)
    return np.clip(calibrado, 0, 255)


def extraer_target_feature(rgb_calibrado, tipo_sensor):
    """
    Aplica la transformación colorimétrica específica requerida por la física de cada sensor.
    """
    if tipo_sensor == "Exp1_Irr":
        # Luminosidad ponderada (Munsell) para sensores de temperatura irreversible
        r, g, b = rgb_calibrado[0], rgb_calibrado[1], rgb_calibrado[2]
        return (0.21 * r + 0.72 * g + 0.07 * b) / 255.0
    else:
        # Canal rojo normalizado para sensores TimeStrip y resto de sensores
        return rgb_calibrado[0] / 255.0


# ==========================================
# 2. CARGA DE MODELOS PREDICTIVOS (.PKL)
# ==========================================
# Ruta relativa hacia el repositorio centralizado de Inteligencia Artificial
ruta_modelos = os.path.join("..", "Modelado_Predictivo", "Modelos_Exportados")

try:
    modelo_exp1 = joblib.load(os.path.join(ruta_modelos, 'modelo_random_forest_exp1.pkl'))
    modelo_exp2 = joblib.load(os.path.join(ruta_modelos, 'modelo_random_forest_exp2.pkl'))
    modelo_exp3 = joblib.load(os.path.join(ruta_modelos, 'modelo_random_forest_exp3.pkl'))
    modelo_exp4 = joblib.load(os.path.join(ruta_modelos, 'modelo_polinomico_exp4.pkl'))
    print(">> Modelos predictivos vinculados con éxito.\n")
except FileNotFoundError as e:
    print(f">> AVISO: Faltan modelos en el directorio central: {e}\nModo de solo extracción activado.\n")
    modelo_exp1, modelo_exp2, modelo_exp3, modelo_exp4 = None, None, None, None

# ==========================================
# 3. LECTURA DE QR Y ANCLAJE VECTORIAL
# ==========================================
ruta_imagen = os.path.join("Plantillas", "foto_movil.jpg")
img_bgr = cv2.imread(ruta_imagen)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_debug = img_bgr.copy()

detector = cv2.QRCodeDetector()
datos_qr, bbox, _ = detector.detectAndDecode(img_bgr)

if bbox is not None and datos_qr:
    # Extracción de la cabecera (Payload) para identificar la estructura física
    plantilla_id = datos_qr[:2]

    # Cálculo del centro de gravedad del marcador fiducial (QR)
    tl, tr, br, bl = np.array(bbox[0][0]), np.array(bbox[0][1]), np.array(bbox[0][2]), np.array(bbox[0][3])
    centro_qr = (tl + tr + br + bl) / 4.0
    cv2.circle(img_debug, (int(centro_qr[0]), int(centro_qr[1])), 5, (0, 0, 255), -1)

    # Detección del recuadro interior del QR para calcular la escala física (píxeles/cm)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pixeles_por_cm = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if x < centro_qr[0] < x + w and y < centro_qr[1] < y + h:
            relacion_aspecto = float(w) / h
            if 0.95 <= relacion_aspecto <= 1.05 and w > (tr[0] - tl[0]):
                pixeles_por_cm = w / 3.0
                cv2.rectangle(img_debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
                break

    # Fallback de seguridad si falla la detección del recuadro interior
    if pixeles_por_cm is None:
        pixeles_por_cm = np.linalg.norm(tr - tl) / 2.4

    # Generación de vectores base ortonormales para compensar inclinación de cámara
    vector_x_1cm = (((tr - tl) + (br - bl)) / 2.0) / (np.linalg.norm((tr - tl) + (br - bl)) / (2.0 * pixeles_por_cm))
    vector_y_1cm = (((bl - tl) + (br - tr)) / 2.0) / (np.linalg.norm((bl - tl) + (br - tr)) / (2.0 * pixeles_por_cm))

    # ==========================================
    # 4. EXTRACCIÓN DINÁMICA JERÁRQUICA
    # ==========================================
    config_plantilla = {}

    # El sistema decide en tiempo de ejecución qué topología cargar según el QR
    if plantilla_id == "01":
        print(f">> PROCESANDO PLANTILLA 01 (Paquete ID: {datos_qr[2:]})")
        config_plantilla = {
            "Referencias": {
                "W_Ref": {"x": -1.237, "y": -2.458},
                "B_Ref": {"x": 0.006, "y": -2.458}
            },
            "Sensores": {
                "Exp1_Irr": {
                    "caja_tl_x": -3.985,
                    "caja_tl_y": -3.313,
                    "size_w": 0.169,
                    "size_h": 0.169,
                    "ventanas": [
                        {"nombre": "V_1", "offset_x": 0.989, "offset_y": 4.115},
                        {"nombre": "V_2", "offset_x": 0.980, "offset_y": 3.769},
                        {"nombre": "V_3", "offset_x": 0.980, "offset_y": 3.372},
                        {"nombre": "V_4", "offset_x": 0.989, "offset_y": 2.966},
                        {"nombre": "V_5", "offset_x": 0.989, "offset_y": 2.577},
                        {"nombre": "V_6", "offset_x": 0.997, "offset_y": 2.180},
                        {"nombre": "V_7", "offset_x": 0.997, "offset_y": 1.783},
                        {"nombre": "V_8", "offset_x": 0.997, "offset_y": 1.386}
                    ]
                },
                "Exp2_Rev": {
                    "caja_tl_x": 2.007,
                    "caja_tl_y": -3.313,
                    "size_w": 0.169,
                    "size_h": 0.169,
                    "ventanas": [
                        {"nombre": "V_1", "offset_x": 0.963, "offset_y": 3.997},
                        {"nombre": "V_2", "offset_x": 0.963, "offset_y": 3.608},
                        {"nombre": "V_3", "offset_x": 0.963, "offset_y": 3.270},
                        {"nombre": "V_4", "offset_x": 0.972, "offset_y": 2.915},
                        {"nombre": "V_5", "offset_x": 0.972, "offset_y": 2.577},
                        {"nombre": "V_6", "offset_x": 0.989, "offset_y": 2.223},
                        {"nombre": "V_7", "offset_x": 0.989, "offset_y": 1.868},
                        {"nombre": "V_8", "offset_x": 0.989, "offset_y": 1.521},
                        {"nombre": "V_9", "offset_x": 1.006, "offset_y": 1.192},
                        {"nombre": "V_10", "offset_x": 1.006, "offset_y": 0.913}
                    ]
                },
                "Exp3_Hum": {
                    "caja_tl_x": -3.190,
                    "caja_tl_y": 2.349,
                    "size_w": 0.254,
                    "size_h": 0.254,
                    "ventanas": [
                        {"nombre": "V_1", "offset_x": 0.439, "offset_y": 0.363},
                        {"nombre": "V_2", "offset_x": 1.496, "offset_y": 0.372},
                        {"nombre": "V_3", "offset_x": 2.561, "offset_y": 0.372},
                        {"nombre": "V_4", "offset_x": 3.566, "offset_y": 0.372}
                    ]
                }
            }
        }

    elif plantilla_id == "02":
        print(f">> PROCESANDO PLANTILLA 02 (Paquete ID: {datos_qr[2:]})")
        config_plantilla = {
            "Referencias": {
                "W_Ref": {"x": -1.237, "y": -2.458},
                "B_Ref": {"x": 0.006, "y": -2.458}
            },
            "Sensores": {
                "Exp1_Irr": {
                    "caja_tl_x": -3.985,
                    "caja_tl_y": -3.313,
                    "size_w": 0.169,
                    "size_h": 0.169,
                    "ventanas": [
                        {"nombre": "V_1", "offset_x": 0.989, "offset_y": 4.115},
                        {"nombre": "V_2", "offset_x": 0.980, "offset_y": 3.769},
                        {"nombre": "V_3", "offset_x": 0.980, "offset_y": 3.372},
                        {"nombre": "V_4", "offset_x": 0.989, "offset_y": 2.966},
                        {"nombre": "V_5", "offset_x": 0.989, "offset_y": 2.577},
                        {"nombre": "V_6", "offset_x": 0.997, "offset_y": 2.180},
                        {"nombre": "V_7", "offset_x": 0.997, "offset_y": 1.783},
                        {"nombre": "V_8", "offset_x": 0.997, "offset_y": 1.386}
                    ]
                },
                "Exp2_Rev": {
                    "caja_tl_x": 2.007,
                    "caja_tl_y": -3.313,
                    "size_w": 0.169,
                    "size_h": 0.169,
                    "ventanas": [
                        {"nombre": "V_1", "offset_x": 0.963, "offset_y": 3.997},
                        {"nombre": "V_2", "offset_x": 0.963, "offset_y": 3.608},
                        {"nombre": "V_3", "offset_x": 0.963, "offset_y": 3.270},
                        {"nombre": "V_4", "offset_x": 0.972, "offset_y": 2.915},
                        {"nombre": "V_5", "offset_x": 0.972, "offset_y": 2.577},
                        {"nombre": "V_6", "offset_x": 0.989, "offset_y": 2.223},
                        {"nombre": "V_7", "offset_x": 0.989, "offset_y": 1.868},
                        {"nombre": "V_8", "offset_x": 0.989, "offset_y": 1.521},
                        {"nombre": "V_9", "offset_x": 1.006, "offset_y": 1.192},
                        {"nombre": "V_10", "offset_x": 1.006, "offset_y": 0.913}
                    ]
                },
                "Exp3_Hum": {
                    "caja_tl_x": -3.190,
                    "caja_tl_y": 2.349,
                    "size_w": 0.254,
                    "size_h": 0.254,
                    "ventanas": [
                        {"nombre": "V_1", "offset_x": 0.439, "offset_y": 0.363},
                        {"nombre": "V_2", "offset_x": 1.496, "offset_y": 0.372},
                        {"nombre": "V_3", "offset_x": 2.561, "offset_y": 0.372},
                        {"nombre": "V_4", "offset_x": 3.566, "offset_y": 0.372}
                    ]
                },
                "Exp4_Time": {
                    "caja_tl_x": 4.500,
                    "caja_tl_y": -2.763,
                    "size_w": 0.423,
                    "size_h": 1.014,
                    "ventanas": [
                        {"nombre": "V_1", "offset_x": 0.989, "offset_y": 1.327}
                    ]
                }
            }
        }

    else:
        print(f">> ERROR: ID de Plantilla '{plantilla_id}' desconocido.")
        config_plantilla = None

    if config_plantilla:
        # 4.1 Extraer Patrones de Referencia Globales (Iluminancia)
        rgb_refs = {}
        for nombre, params in config_plantilla["Referencias"].items():
            pos = centro_qr + (params["x"] * vector_x_1cm) + (params["y"] * vector_y_1cm)
            size_ref_cm = 0.75

            rgb_refs[nombre] = obtener_media_rgb_cruda(img_rgb, int(pos[0]), int(pos[1]), size_ref_cm, size_ref_cm,
                                                       pixeles_por_cm)

            size_pix_ref = int(size_ref_cm * pixeles_por_cm)
            mitad_ref = max(1, size_pix_ref // 2)
            pt1 = (int(pos[0]) - mitad_ref, int(pos[1]) - mitad_ref)
            pt2 = (int(pos[0]) + mitad_ref, int(pos[1]) + mitad_ref)
            cv2.rectangle(img_debug, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(img_debug, nombre, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        w_rgb, b_rgb = rgb_refs["W_Ref"], rgb_refs["B_Ref"]

        # 4.2 Iteración sobre Sensores y Ventanas de Lectura
        vectores_finales = {}

        for tipo_sensor, config in config_plantilla["Sensores"].items():
            vector_sensor = []
            caja_tl_x, caja_tl_y = config["caja_tl_x"], config["caja_tl_y"]
            size_w_cm, size_h_cm = config["size_w"], config["size_h"]

            for win in config["ventanas"]:
                # Proyección espacial desde el centro del QR hasta la ventana específica
                x_cm_total = caja_tl_x + win["offset_x"]
                y_cm_total = caja_tl_y + win["offset_y"]
                pos = centro_qr + (x_cm_total * vector_x_1cm) + (y_cm_total * vector_y_1cm)

                rgb_crudo = obtener_media_rgb_cruda(img_rgb, int(pos[0]), int(pos[1]), size_w_cm, size_h_cm,
                                                    pixeles_por_cm)

                mitad_w = max(1, int(size_w_cm * pixeles_por_cm) // 2)
                mitad_h = max(1, int(size_h_cm * pixeles_por_cm) // 2)
                pt1 = (int(pos[0]) - mitad_w, int(pos[1]) - mitad_h)
                pt2 = (int(pos[0]) + mitad_w, int(pos[1]) + mitad_h)
                cv2.rectangle(img_debug, pt1, pt2, (255, 0, 0), 2)

                # Pipeline de tratamiento de señal: Raw -> Calibrado -> Feature Extracted
                r_cal = calibrar_canal(rgb_crudo[0], w_rgb[0], b_rgb[0])
                g_cal = calibrar_canal(rgb_crudo[1], w_rgb[1], b_rgb[1])
                b_cal = calibrar_canal(rgb_crudo[2], w_rgb[2], b_rgb[2])

                valor_final = extraer_target_feature(np.array([r_cal, g_cal, b_cal]), tipo_sensor)
                vector_sensor.append(valor_final)

            # Empaquetado del vector de características para Machine Learning
            vectores_finales[tipo_sensor] = vector_sensor

        # ==========================================
        # 5. INFERENCIA Y RESULTADOS DE PRODUCCIÓN
        # ==========================================
        cv2.imwrite('debug_vision.jpg', img_debug)
        print("\n>> INFO: Artefacto de depuración generado ('debug_vision.jpg').")
        print("\n>> RESULTADOS DE LA PREDICCIÓN LOGÍSTICA:")

        if "Exp1_Irr" in vectores_finales and len(vectores_finales["Exp1_Irr"]) > 0:
            features_exp1 = np.array([vectores_finales["Exp1_Irr"]])
            if modelo_exp1:
                pred_1 = modelo_exp1.predict(features_exp1)[0]
                print(f"   [Sensor 1] Temp Irreversible: {pred_1:.1f} °C")

        if "Exp2_Rev" in vectores_finales and len(vectores_finales["Exp2_Rev"]) > 0:
            features_exp2 = np.array([vectores_finales["Exp2_Rev"]])
            if modelo_exp2:
                pred_2 = modelo_exp2.predict(features_exp2)[0]
                print(f"   [Sensor 2] Temp Reversible:   {pred_2:.1f} °C")

        if "Exp3_Hum" in vectores_finales and len(vectores_finales["Exp3_Hum"]) > 0:
            features_exp3 = np.array([vectores_finales["Exp3_Hum"]])
            if modelo_exp3:
                pred_3 = modelo_exp3.predict(features_exp3)[0]
                print(f"   [Sensor 3] Humedad Relativa:  {pred_3:.1f} %")

        if "Exp4_Time" in vectores_finales and len(vectores_finales["Exp4_Time"]) > 0:
            features_exp4 = np.array([vectores_finales["Exp4_Time"]])
            if modelo_exp4:
                pred_4 = modelo_exp4.predict(features_exp4)[0]
                print(f"   [Sensor 4] Días en Servicio:  {pred_4:.1f} Días")

else:
    print("Error: No se ha detectado un anclaje QR válido en la imagen proporcionada.")
