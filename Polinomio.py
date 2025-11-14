import pandas as pd
import numpy as np

# === 1) Ruta del archivo === C:\Users\rsmed\PycharmProjects\pythonProject
file_path = r"C:\Users\rsmed\PycharmProjects\pythonProject\metrics.csv"

# === 2) Cargar datos ===
df = pd.read_csv(file_path)

# Columnas que usaremos (asegúrate que existan)
assert "GooseTemp" in df.columns, "Falta columna GooseTemp"
assert "GooseCapHex" in df.columns, "Falta columna GooseCapHex"
assert "GooseTPM" in df.columns, "Falta columna GooseTPM"

# --- convertir a numpy/valores ---
T = df["GooseTemp"].astype(float).to_numpy()
C_hex_list = df["GooseCapHex"].astype(str).to_list()
TPM_medido = df["GooseTPM"].astype(float).to_numpy()

# === 3) Convertir hex->decimal (array numpy) ===
# Maneja cadenas con/ sin prefijo "0x"
def hex_to_int_safe(h):
    hstr = str(h).strip()
    if hstr.lower().startswith("0x"):
        hstr = hstr[2:]
    return int(hstr, 16)

C = np.array([hex_to_int_safe(x) for x in C_hex_list], dtype=float)

# === 4) Escalas (usar exactamente los min/max del dataset) ===
T_min, T_max = float(np.min(T)), float(np.max(T))
C_min, C_max = float(np.min(C)), float(np.max(C))
# evitar división por cero (no pasa aquí pero por seguridad)
if T_max == T_min:
    raise ValueError("T_max == T_min, datos de temperatura inválidos")
if C_max == C_min:
    raise ValueError("C_max == C_min, datos de capacitancia inválidos")

T_scaled = (T - T_min) / (T_max - T_min)
C_scaled = (C - C_min) / (C_max - C_min)

# === 5) Theta (квадратный многочлен) ===
theta1 = -76.16391930260264
theta2 = -1.4556770920304842
theta3 = 6.917343053617422
theta4 = 168.92273573166048
theta5 = 95.86741996935297
theta6 = -85.54864868707303

# === 6) Calcular TPM_surf para todo el dataset (modelo cuadrático) ===
# TPM_surf = theta1 + theta2*T + theta3*C + theta4*T^2 + theta5*C^2 + theta6*T*C
TPM_surf = (
    theta1
    + theta2 * T_scaled
    + theta3 * C_scaled
    + theta4 * (T_scaled ** 2)
    + theta5 * (C_scaled ** 2)
    + theta6 * (T_scaled * C_scaled)
)

# === 7) Métricas y comprobaciones ===
diff = TPM_medido - TPM_surf
err_rel = diff / TPM_medido * 100

print(f"Filas usadas: {len(T)}")
print("\n--- Escala de variables (dataset) ---")
print(f"T: Min={T_min:.3f}  Max={T_max:.3f}")
print(f"C_dec: Min={int(C_min)}  Max={int(C_max)}")
print(f"T_scaled: Min={T_scaled.min():.5f}  Max={T_scaled.max():.5f}")
print(f"C_scaled: Min={C_scaled.min():.5f}  Max={C_scaled.max():.5f}")

print("\n--- Coeficientes theta (uso directo) ---")
print(f"theta1 = {theta1}")
print(f"theta2 = {theta2}")
print(f"theta3 = {theta3}")
print(f"theta4 = {theta4}")
print(f"theta5 = {theta5}")
print(f"theta6 = {theta6}")

print("\n--- TPM_surf (dataset) ---")
print(f"Min: {TPM_surf.min():.3f}")
print(f"Max: {TPM_surf.max():.3f}")
print(f"Promedio: {TPM_surf.mean():.3f}")

print("\n--- Diferencia TPM medido - TPM_surf ---")
print(f"Min: {diff.min():.3f}")
print(f"Max: {diff.max():.3f}")
print(f"Promedio: {diff.mean():.3f}")

print("\n--- Error relative (%) ---")
print(f"Max positive: {np.max(err_rel):.2f} %")
print(f"Max negative: {np.min(err_rel):.2f} %")
print(f"Avarage : {np.mean(np.abs(err_rel)):.2f} %")

# === 8) PRUEBA MANUAL: T_test = 110.0 ===
T_test = 110.0
# Puedes cambiar este hex si quieres probar otro valor; yo lo calculo y también pruebo con el hex más cercano del dataset.
C_hex_test = "CE28A0"   # valor ejemplo (tú puedes cambiar aquí)
C_dec_test = hex_to_int_safe(C_hex_test)

# Escalado según min/max del dataset
T_scaled_test = (T_test - T_min) / (T_max - T_min)
C_scaled_test = (C_dec_test - C_min) / (C_max - C_min)

# Indicar si está fuera de rango [0,1]
out_of_range = (T_scaled_test < 0 or T_scaled_test > 1) or (C_scaled_test < 0 or C_scaled_test > 1)

print("\n--- Prueba manual (cuadrático) ---")
print(f"T_test = {T_test} °C")
print(f"C_hex_test = {C_hex_test} -> C_dec_test = {C_dec_test}")
print(f"T_scaled_test = {T_scaled_test:.6f}, C_scaled_test = {C_scaled_test:.6f}")
if out_of_range:
    print("⚠️ ATENCIÓN: prueba manual fuera del rango de entrenamiento (extrapolación). Resultado puede no ser fiable.")

TPM_test = (
    theta1
    + theta2 * T_scaled_test
    + theta3 * C_scaled_test
    + theta4 * (T_scaled_test ** 2)
    + theta5 * (C_scaled_test ** 2)
    + theta6 * (T_scaled_test * C_scaled_test)
)

print(f"TPM_surf (estimado) para T={T_test}°C y C_hex={C_hex_test}: {TPM_test:.6f} %")

# === 9) Extra: determinar C_hex más cercano en dataset (por si quieres comparar) ===
# calcula diferencia absoluta entre C_dec_test y valores del dataset, y muestra la fila más cercana
idx_closest = np.argmin(np.abs(C - C_dec_test))
print("\nFila del dataset con C_dec más cercano:")
print(f"Índice: {idx_closest}, GooseTemp={T[idx_closest]}, GooseCapHex={C_hex_list[idx_closest]}, GooseTPM={TPM_medido[idx_closest]}")
print(f"Entrada dataset (escalada): T_s={T_scaled[idx_closest]:.6f}, C_s={C_scaled[idx_closest]:.6f}, TPM_surf={TPM_surf[idx_closest]:.6f}")
