import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 1. Load dataset ===
file_path = r"C:\Users\rsmed\PycharmProjects\pythonProject\metrics.csv"
df = pd.read_csv(file_path)

# === 2. Convert hexadecimal capacitance to integer ===
# Remove possible '0x' prefix and convert from hex to integer
df['GooseCapInt'] = df['GooseCapHex'].apply(lambda x: int(str(x), 16))

# === 3. Select relevant columns ===
C = df['GooseCapInt'].astype(float)
T = df['GooseTemp'].astype(float)
TPM = df['GooseTPM'].astype(float)

# === 4. Build polynomial terms ===
df['C'] = C
df['T'] = T
df['C_T'] = C * T
df['C2'] = C ** 2
df['T2'] = T ** 2

# === 5. Define X (features) and y (target) ===
X = df[['C', 'T', 'C_T', 'C2', 'T2']]
y = TPM

# === 6. Fit regression model ===
model = LinearRegression()
model.fit(X, y)

# === 7. Extract coefficients ===
alpha_0 = model.intercept_
alpha_1, alpha_2, alpha_3, alpha_4, alpha_5 = model.coef_

# === 8. Predict and evaluate model ===
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# === 9. Display results ===
print("===== Model Ajust =====")
print(f"α0 = {alpha_0:.6e}")
print(f"α1 = {alpha_1:.6e}")
print(f"α2 = {alpha_2:.6e}")
print(f"α3 = {alpha_3:.6e}")
print(f"α4 = {alpha_4:.6e}")
print(f"α5 = {alpha_5:.6e}")
print("\n===== Evaluation =====")
print(f"R²   = {r2:.6f}")
print(f"MAE  = {mae:.6f}")
print(f"RMSE = {rmse:.6f}")

# === 10. Optional: save coefficients to a CSV file ===
coef_df = pd.DataFrame({
    'Coeficiente': ['α0', 'α1', 'α2', 'α3', 'α4', 'α5'],
    'Valor': [alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5]
})
coef_df.to_csv('coeficientes_TPM_modelo.csv', index=False)
print("\nCoeficientes guardados en 'coeficientes_TPM_modelo.csv'")
