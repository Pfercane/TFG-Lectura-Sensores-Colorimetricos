"""
Módulo de Validación Cruzada y Evaluación de Generalización (Hold-out Validation)
---------------------------------------------------------------------------------
Este script implementa una separación estructural de los datos en conjuntos de
Entrenamiento (Train) y Prueba (Test). Su finalidad es auditar las arquitecturas
predictivas midiendo su rendimiento empírico ante datos "invisibles", permitiendo
detectar y mitigar el fenómeno de sobreajuste (overfitting).

A través de un enrutamiento dinámico, el script es capaz de auditar cualquiera
de los cuatro experimentos físicos del proyecto utilizando hiperparámetros de
particionado específicos para cada topología de sensor.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PANEL DE CONTROL Y PARAMETRIZACIÓN
# ==========================================
# SELECCIONA EL EXPERIMENTO A AUDITAR (1, 2, 3 o 4)
EXPERIMENTO_SELECCIONADO = 4

# SELECCIONA LA ARQUITECTURA DEL MODELO A EVALUAR:
# 1 = Regresión Lineal Simple (Baseline)
# 2 = Regresión Polinómica Grado 2
# 3 = Random Forest Regressor
ARQUITECTURA_MODELO = 2

# ==========================================
# 2. MAPA ESTRUCTURAL E HIPERPARÁMETROS (ENRUTAMIENTO)
# ==========================================
# Diccionario maestro que enlaza cada experimento con su dataset relativo y
# define los hiperparámetros óptimos de particionado detectados empíricamente.
CONFIG_SPLIT = {
    1: {
        "csv_path": os.path.join("..", "Experimento_1", "Datos", "Exp1_Resultados_Media.csv"),
        "test_size": 0.2,
        "random_state": 42
    },
    2: {
        "csv_path": os.path.join("..", "Experimento_2", "Datos", "Exp2_Resultados_Media_Global.csv"),
        "test_size": 0.2,
        "random_state": 42
    },
    3: {
        "csv_path": os.path.join("..", "Experimento_3", "Datos", "Exp3_Resultados_Media_Global.csv"),
        "test_size": 0.2,
        "random_state": 42
    },
    4: {
        "csv_path": os.path.join("..", "Experimento_4", "Datos", "Exp4_Resultados_Media.csv"),
        "test_size": 0.1,  # Ajuste fino a 90/10 por volumen de muestras
        "random_state": 23
    }
}

# ==========================================
# 3. INGESTA Y PARTICIÓN DE DATOS (DATA SPLIT)
# ==========================================
config_actual = CONFIG_SPLIT[EXPERIMENTO_SELECCIONADO]
ruta_csv = config_actual["csv_path"]

try:
    data = pd.read_csv(ruta_csv)
except FileNotFoundError:
    print(f">> ERROR CRÍTICO: No se encuentra el dataset en la ruta:\n{ruta_csv}")
    print("Asegúrate de ejecutar este script desde la carpeta 'Modelado_Predictivo/'.")
    exit()

target_name = data.columns[0]
y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values

# Partición de Hold-out (Train/Test)
# Se emplea el random_state configurado para garantizar reproducibilidad científica.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config_actual["test_size"],
    random_state=config_actual["random_state"]
)

print(f"--- AUDITORÍA DE GENERALIZACIÓN: EXPERIMENTO {EXPERIMENTO_SELECCIONADO} ---")
print(f"Filas destinadas al entrenamiento (Train): {len(X_train)}")
print(f"Filas ocultas para evaluación ciega (Test): {len(X_test)}\n")

# ==========================================
# 4. CONSTRUCCIÓN DE PIPELINES Y ENTRENAMIENTO AISLADO
# ==========================================
if ARQUITECTURA_MODELO == 1:
    print(">> Arquitectura: Regresión Lineal Simple")
    model = LinearRegression()

elif ARQUITECTURA_MODELO == 2:
    print(">> Arquitectura: Regresión Polinómica (Grado 2)")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("reg", LinearRegression())
    ])

elif ARQUITECTURA_MODELO == 3:
    print(">> Arquitectura: Random Forest Regressor")
    model = RandomForestRegressor(n_estimators=100, random_state=42)

# IMPORTANTE: El modelo se ajusta EXCLUSIVAMENTE con la porción de entrenamiento
model.fit(X_train, y_train)

# ==========================================
# 5. AUDITORÍA CONTRA DATOS INVISIBLES
# ==========================================
# Predicción sobre el conjunto de Test (datos que el modelo jamás ha visto)
y_pred = model.predict(X_test)

# Cálculo del error real de generalización
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RESULTADOS REALES CON DATOS INVISIBLES (CONJUNTO DE TEST):")
print(f"RMSE (Error Medio) = {rmse:.2f} unidades de {target_name}")
print(f"R² (Precisión)     = {r2:.4f}")

# ==========================================
# 6. VISUALIZACIÓN DE GENERALIZACIÓN
# ==========================================
plt.figure(figsize=(8, 6))

# Se grafica únicamente la evaluación ciega
plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue', edgecolor='k', label='Predicciones del Modelo (Test)')

# Línea de ajuste ideal delimitada por los valores mínimos y máximos de la muestra de prueba
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2, label='Ajuste Ideal')

plt.xlabel(f"{target_name} Real (Oculto)")
plt.ylabel(f"{target_name} Estimada (Predicción)")
plt.title(f"Evaluación de Generalización (Arq {ARQUITECTURA_MODELO}) - Experimento {EXPERIMENTO_SELECCIONADO}")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()