import pandas as pd
import numpy as np

# === File path ===
file_path = r"C:\Users\rsmed\PycharmProjects\pythonProject\metrics.csv"

# === Load dataset ===
data = pd.read_csv(file_path)

# === Convert hexadecimal capacitance to decimal ===
data['GooseCapDec'] = data['GooseCapHex'].apply(lambda x: int(str(x), 16))

# === Display dataset stats ===
num_rows = len(data)
T_min, T_max = data['GooseTemp'].min(), data['GooseTemp'].max()
C_min, C_max = data['GooseCapDec'].min(), data['GooseCapDec'].max()

print(f"Filas usadas: {num_rows}\n")
print("--- Escala de variables (dataset) ---")
print(f"T: Min={T_min:.3f}  Max={T_max:.3f}")
print(f"C_dec: Min={C_min}  Max={C_max}")
print(f"T_scaled: Min=0.00000  Max=1.00000")
print(f"C_scaled: Min=0.00000  Max=1.00000\n")

# === Model coefficients ===
alpha0 = -1.775454e+03
alpha1 = 1.275336e-04
alpha2 = -5.891628e-07
alpha3 = 9.436783e-08
alpha4 = 2.461430e-13
alpha5 = -3.036403e-03

print("===== Model Ajust =====")
print(f"α0 = {alpha0:.6e}")
print(f"α1 = {alpha1:.6e}")
print(f"α2 = {alpha2:.6e}")
print(f"α3 = {alpha3:.6e}")
print(f"α4 = {alpha4:.6e}")
print(f"α5 = {alpha5:.6e}\n")

# === Model function ===
def predict_tpm(T, C):
    """Predicts TPM using the fitted polynomial regression model"""
    return (alpha0 +
            alpha1 * C +
            alpha2 * T +
            alpha3 * (C * T) +
            alpha4 * (C ** 2) +
            alpha5 * (T ** 2))

# === Select representative samples ===
# Minimum temperature
sample_min = data.loc[data['GooseTemp'].idxmin()]
# Median temperature
sample_med = data.iloc[len(data)//2]
# Maximum temperature
sample_max = data.loc[data['GooseTemp'].idxmax()]

samples = [("Min", sample_min), ("Med", sample_med), ("Max", sample_max)]

print("===== Model Verification =====")
for label, row in samples:
    T = row['GooseTemp']
    C = row['GooseCapDec']
    TPM_real = row['GooseTPM']
    TPM_pred = predict_tpm(T, C)
    print(f"\n[{label} sample]")
    print(f"T = {T:.3f},  C_dec = {C},  TPM_real = {TPM_real:.3f}")
    print(f"TPM_pred (modelo) = {TPM_pred:.3f}")

# === Optional: check overall prediction quality on small subset ===
subset = data.sample(100, random_state=42)
subset['TPM_pred'] = predict_tpm(subset['GooseTemp'], subset['GooseCapDec'])
error = np.mean(np.abs(subset['GooseTPM'] - subset['TPM_pred']))
print(f"\nError medio absoluto (100 muestras aleatorias) = {error:.3f}")
