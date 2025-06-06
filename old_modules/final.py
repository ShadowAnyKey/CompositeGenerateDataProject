import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# === 1) Путь к данным ===
csv_path = r"C:\Users\kacha\Downloads\tests\38.csv"   # CSV с двумя колонками: Deformation;Standard_Stress

# === 2) Жёстко заданные физические и мета-параметры ===
# Материалы:
fiber_E   = 240e3    # Young’s modulus of carbon fiber, МПа
matrix_E  = 2.7e3    # Young’s modulus of polysulfone matrix, МПа
# (дополнительный квадратичный член, если нужен)
K2        = 0.0      # [МПа], по умолчанию 0

# Геометрия / состав образца:
polymer_solution_pct = 20.0         # % полимера в растворе
length_mm            = 236.0        # мм длина образца
mass_mg              = 261.0        # мг масса образца
fiber_content_pct    = 72.33716475  # % доля углеродного волокна

# Веса для обучения:
# (MLP подгоняет резидуал, MLP.loss=MSE_resid)
# физический закон просто вычитается из данных
# нет дополнительных штрафов здесь
mlp_hidden = (100,100)
random_state = 0

# === 3) Загрузка CSV ===
df = pd.read_csv(csv_path, sep=';', decimal=',', engine='python')
eps_pct = df['Deformation'].astype(float).values        # [%]
stress  = df['Standard_Stress'].astype(float).values    # [МПа]

# Приводим в удобные формы
eps_frac = eps_pct / 100.0   # теперь ε в долях (0…0.015…)

# === 4) Физическая часть: σ_phys = E_eff·ε + K2·ε² ===
VF    = fiber_content_pct / 100.0
E_eff = fiber_E * VF + matrix_E * (1.0 - VF)
stress_phys = E_eff * eps_frac + K2 * eps_frac**2

# === 5) Остатки для обучения MLP ===
residual = stress - stress_phys   # σ_data − σ_phys

# === 6) Собираем признаки для MLP (5-мерный вектор) ===
# eps_pct, polymer_solution_pct, length_mm, mass_mg, fiber_content_pct
X = np.vstack([
    eps_pct,
    np.full_like(eps_pct, polymer_solution_pct),
    np.full_like(eps_pct, length_mm),
    np.full_like(eps_pct, mass_mg),
    np.full_like(eps_pct, fiber_content_pct),
]).T  # shape = (N,5)

# === 7) Масштабирование ===
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)

scaler_y = StandardScaler().fit(residual.reshape(-1,1))
y_scaled = scaler_y.transform(residual.reshape(-1,1)).ravel()

# === 8) Обучаем MLP на остатках ===
mlp = MLPRegressor(
    hidden_layer_sizes=mlp_hidden,
    activation='relu',
    solver='adam',
    max_iter=5000,
    random_state=random_state
)
mlp.fit(X_scaled, y_scaled)

# === 9) Оценка на обучающей выборке ===
resid_pred_train = scaler_y.inverse_transform(
    mlp.predict(X_scaled).reshape(-1,1)
).ravel()
stress_pred_train = stress_phys + resid_pred_train

print("Train total MSE (σ_data vs σ_phys+MLP):", 
      mean_squared_error(stress, stress_pred_train))

# === 10) Генерируем гладкую кривую для вывода ===
eps_test = np.linspace(eps_pct.min(), eps_pct.max(), 300)
X_test   = np.vstack([
    eps_test,
    np.full_like(eps_test, polymer_solution_pct),
    np.full_like(eps_test, length_mm),
    np.full_like(eps_test, mass_mg),
    np.full_like(eps_test, fiber_content_pct),
]).T
X_test_s = scaler_X.transform(X_test)

resid_test = scaler_y.inverse_transform(
    mlp.predict(X_test_s).reshape(-1,1)
).ravel()
stress_test = E_eff * (eps_test/100.0) + K2*(eps_test/100.0)**2 + resid_test

# === 11) Визуализация результата ===
plt.figure(figsize=(8,5))
plt.scatter(eps_pct, stress,   c='k', s=20, label='Data')
plt.plot(    eps_test, stress_test, 'b-', lw=2,     label='Phys+MLP')
plt.xlabel('Deformation [%]')
plt.ylabel('Standard Stress [MPa]')
plt.legend()
plt.tight_layout()
plt.show()