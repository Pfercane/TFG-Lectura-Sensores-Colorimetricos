"""
Módulo de Generación de Anclajes Físicos y Trazabilidad (Códigos QR)
--------------------------------------------------------------------
Este script genera códigos QR industriales diseñados para actuar como
sistemas de anclaje (fiducial markers) y portadores de metadatos.
Codifica la plantilla estructural del sensor y el identificador único del
paquete, permitiendo que el sistema de visión artificial posterior sea
completamente autónomo y modular.

Autor: Pedro Gabriel Fernández Cañete
Institución: Universidad de Granada (UGR)
"""

import segno

# ==========================================
# 1. CONFIGURACIÓN DE TRAZABILIDAD
# ==========================================
# 'plantilla' define la geometría a cargar en el lector (01 = 3 sensores, 02 = 4 sensores)
plantilla = "02"
# 'numero_envio' actúa como Primary Key en una base de datos logística
numero_envio = "000001"

# Concatenación para formar la cadena alfanumérica de 8 dígitos de carga útil (Payload)
codigo_final = plantilla + numero_envio

print(f"--- Generador de QR Industrial ---")
print(f"Plantilla objetivo: {plantilla}")
print(f"ID de envío:        {numero_envio}")
print(f"Código a codificar: {codigo_final}\n")

# ==========================================
# 2. MOTOR DE GENERACIÓN Y CORRECCIÓN DE ERRORES
# ==========================================
# Se fuerza un nivel de corrección de errores 'High' (h). Esto añade redundancia
# matemática (algoritmo Reed-Solomon) permitiendo que el lector recupere el 100%
# de los datos aunque hasta un 30% del QR físico esté destruido, sucio o manchado.
qr = segno.make(codigo_final, micro=False, error='h')

# Volcado de metadatos de validación por consola
print(f"Versión del QR: {qr.version}")
print(f"Nivel de corrección: {qr.error}")
print(f"Tamaño de la matriz: {qr.symbol_size()} módulos\n")

# ==========================================
# 3. EXPORTACIÓN Y SERIALIZACIÓN DE ARTEFACTOS
# ==========================================
# Generación de nombres de archivo dinámicos para trazabilidad de versiones
filename_color = f'QR_T{plantilla}_ID{numero_envio}_color.png'
filename_black = f'QR_T{plantilla}_ID{numero_envio}_black.png'

# Renderizado vectorial/matricial escalado (scale=10) para evitar pérdida de
# resolución en la impresión física.
qr.save(filename_color, scale=10, dark='#006EA6')
qr.save(filename_black, scale=10)

print(f">> ÉXITO: Códigos QR generados y guardados como:")
print(f"   - {filename_color}")
print(f"   - {filename_black}")