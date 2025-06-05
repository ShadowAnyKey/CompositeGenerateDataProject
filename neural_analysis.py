import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import io

class NeuralAnalyzer:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=5000, random_state=0)

    def load_data(self, csv_files):
        all_eps_pct, all_stress = [], []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, sep=';', decimal=',', engine='python')
            all_eps_pct.extend(df['Deformation'].values)
            all_stress.extend(df['Standard_Stress'].values)
        return np.array(all_eps_pct), np.array(all_stress)

    def fit(self, eps_pct, stress, params):
        VF = params['fiber_content_pct'] / 100.0
        E_eff = 240e3 * VF + 2.7e3 * (1 - VF)
        stress_phys = E_eff * eps_pct / 100.0
        residual = stress - stress_phys

        X = np.vstack([
            eps_pct,
            np.full_like(eps_pct, params['polymer_solution_pct']),
            np.full_like(eps_pct, params['length_mm']),
            np.full_like(eps_pct, params['mass_mg']),
            np.full_like(eps_pct, params['fiber_content_pct'])
        ]).T

        self.scaler_X.fit(X)
        X_scaled = self.scaler_X.transform(X)
        self.scaler_y.fit(residual.reshape(-1, 1))
        y_scaled = self.scaler_y.transform(residual.reshape(-1, 1)).ravel()

        self.model.fit(X_scaled, y_scaled)
        
        predicted_residual = self.scaler_y.inverse_transform(self.model.predict(X_scaled).reshape(-1, 1)).flatten()
        mse = mean_squared_error(stress, stress_phys + predicted_residual)

        return mse

    def predict_curve(self, eps_range, params, num_points=300):
        eps_test = np.linspace(eps_range[0], eps_range[1], num_points)
        VF = params['fiber_content_pct'] / 100.0
        E_eff = 240e3 * VF + 2.7e3 * (1 - VF)
        X_test = np.vstack([
            eps_test,
            np.full_like(eps_test, params['polymer_solution_pct']),
            np.full_like(eps_test, params['length_mm']),
            np.full_like(eps_test, params['mass_mg']),
            np.full_like(eps_test, params['fiber_content_pct'])
        ]).T

        predicted_residual = self.model.predict(self.scaler_X.transform(X_test)).reshape(-1, 1)
        resid_test = self.scaler_y.inverse_transform(predicted_residual).flatten()

        stress_predicted = E_eff * eps_test / 100.0 + resid_test
        return eps_test, stress_predicted

    def generate_multiple_samples(self, params_list, eps_range, num_points=300):
        samples = {}
        for idx, params in enumerate(params_list, start=1):
            eps_test, stress_predicted = self.predict_curve(eps_range, params, num_points)
            samples[f"Sample_{idx}"] = pd.DataFrame({
                "Deformation (%)": eps_test,
                "Predicted Stress (MPa)": stress_predicted.flatten()
            })

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            for sheet_name, df_sample in samples.items():
                df_sample.to_excel(writer, sheet_name=sheet_name, index=False)
        excel_buffer.seek(0)
        return excel_buffer