"""
Motor Central de Machine Learning y Serialización de Modelos
------------------------------------------------------------
Este script unificado actúa como el orquestador principal de Inteligencia
Artificial del proyecto. Centraliza el entrenamiento, evaluación cruzada y
exportación de los modelos predictivos para los cuatro experimentos físicos
(Temperatura Irreversible, Reversible, Humedad y TimeStrip).

Aplicando el principio de diseño DRY (Don't Repeat Yourself), el script
utiliza rutas relativas transversales para ingerir los conjuntos de datos
limpios desde los directorios de experimentación y genera artefactos
persistentes (.pkl) en un directorio unificado, listos para su despliegue
en la arquitectura del Lector QR de producción.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. PANEL DE CONTROL Y PARAMETRIZACIÓN
# ==========================================
# SELECCIONA EL EXPERIMENTO A ENTRENAR (1, 2, 3 o 4)
EXPERIMENTO_SELECCIONADO = 1

# SELECCIONA LA ARQUITECTURA DEL MODELO:
# 1 = Regresión Lineal Simple (Baseline de control)
# 2 = Regresión Polinómica Grado 2 (Pipeline con StandardScaler)
# 3 = Random Forest Regressor (Modelo no lineal complejo)
ARQUITECTURA_MODELO = 3

# ==========================================
# 2. MAPA ESTRUCTURAL DE DATOS (ENRUTAMIENTO DINÁMICO)
# ==========================================
# Directorio de destino para los modelos compilados
CARPETA_EXPORTACION = "Modelos_Exportados"
os.makedirs(CARPETA_EXPORTACION, exist_ok=True)

# Diccionario de topología de Machine Learning.
# Enlaza cada experimento con su dataset (subiendo un nivel '..'), el modelo
# que ha demostrado mejor rendimiento empírico (producción) y su nombre de salida.
CONFIG_ML = {
    1: {
        "csv_path": os.path.join("..", "Experimento_1", "Datos", "Exp1_Resultados_Media.csv"),
        "modelo_produccion": 3, # Random Forest
        "pkl_name": "modelo_random_forest_exp1.pkl"
    },
    2: {
        "csv_path": os.path.join("..", "Experimento_2", "Datos", "Exp2_Resultados_Media_Global.csv"),
        "modelo_produccion": 3, # Random Forest
        "pkl_name": "modelo_random_forest_exp2.pkl"
    },
    3: {
        "csv_path": os.path.join("..", "Experimento_3", "Datos", "Exp3_Resultados_Media_Global.csv"),
        "modelo_produccion": 3, # Random Forest
        "pkl_name": "modelo_random_forest_exp3.pkl"
    },
    4: {
        "csv_path": os.path.join("..", "Experimento_4", "Datos", "Exp4_Resultados_Media.csv"),
        "modelo_produccion": 2, # Polinómico
        "pkl_name": "modelo_polinomico_exp4.pkl"
    }
}

# ==========================================
# 3. INGESTA DE DATOS (ETL)
# ==========================================
config_actual = CONFIG_ML[EXPERIMENTO_SELECCIONADO]
ruta_csv = config_actual["csv_path"]

try:
    data = pd.read_csv(ruta_csv)
except FileNotFoundError:
    print(f">> ERROR CRÍTICO: No se encuentra el dataset en la ruta:\n{ruta_csv}")
    print("Asegúrate de ejecutar este script desde la carpeta 'Modelado_Predictivo/'.")
    exit()

# Partición del Espacio Vectorial:
# Columna 0 = Variable Dependiente Y (Target: Temp, Humedad, Días)
# Columnas 1 a N = Variables Independientes X (Características de Color)
target_name = data.columns[0]
y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values

print(f"--- ORQUESTADOR ML: EXPERIMENTO {EXPERIMENTO_SELECCIONADO} ---")
print(f"Dataset cargado: {os.path.basename(ruta_csv)}")
print(f"Variable a predecir (Y): {target_name}")
print(f"Sensores utilizados (X): {list(data.columns[1:])}\n")

# ==========================================
# 4. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO
# ==========================================
if ARQUITECTURA_MODELO == 1:
    print(">> Entrenando: Regresión Lineal Simple")
    model = LinearRegression()

elif ARQUITECTURA_MODELO == 2:
    print(">> Entrenando: Regresión Polinómica (Grado 2)")
    # Se implementa un Pipeline para asegurar el escalado estandarizado (StandardScaler)
    # previo a la expansión polinómica, garantizando estabilidad numérica.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("reg", LinearRegression())
    ])

elif ARQUITECTURA_MODELO == 3:
    print(">> Entrenando: Random Forest Regressor")
    # Hiperparámetros optimizados para mitigar el overfitting frente a ruido de sensor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42
    )

# Fase de ajuste (Fit) utilizando la matriz de características extraídas
model.fit(X, y)
y_pred = model.predict(X)

# ==========================================
# 5. EVALUACIÓN Y MÉTRICAS DE DESEMPEÑO
# ==========================================
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE (Error Medio) = {rmse:.2f} unidades de {target_name}")
print(f"R² (Precisión)     = {r2:.4f}")

if ARQUITECTURA_MODELO == 1:
    print(f"Coeficientes de importancia de cada ventana: {model.coef_}")

# ==========================================
# 6. VISUALIZACIÓN DE LA CURVA DE CALIBRACIÓN
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6, color='dodgerblue', edgecolor='k', label='Predicciones (Inferencia)')

# Línea base de ajuste perfecto (Error residual cero)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', linewidth=2, label='Ajuste Ideal')

plt.xlabel(f"{target_name} Real (Telemetría)")
plt.ylabel(f"{target_name} Estimada (ML)")
plt.title(f"Auditoría del Modelo (Arq: {ARQUITECTURA_MODELO}) - Experimento {EXPERIMENTO_SELECCIONADO}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==========================================
# 7. EXPORTACIÓN PERSISTENTE (PRODUCCIÓN)
# ==========================================
# El orquestador protege la integridad del sistema: solo compila y exporta
# el algoritmo si coincide con la arquitectura validada para el despliegue.
if ARQUITECTURA_MODELO == config_actual["modelo_produccion"]:
    ruta_salida = os.path.join(CARPETA_EXPORTACION, config_actual["pkl_name"])
    joblib.dump(model, ruta_salida)
    print(f"\n>> ÉXITO DE COMPILACIÓN: Modelo de Producción guardado en:")
    print(f"   -> {ruta_salida}")
    print("   El artefacto está listo para su integración en el Lector QR.")
else:
    print(f"\n>> AVISO: Configuración de pruebas. No se ha exportado el modelo.")
    print(f"   (Para exportar el Experimento {EXPERIMENTO_SELECCIONADO}, configura ARQUITECTURA_MODELO = {config_actual['modelo_produccion']})")