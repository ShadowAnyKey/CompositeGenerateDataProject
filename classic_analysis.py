import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel
import io

class ClassicAnalyzer:
    def __init__(self, error_threshold=15, pair_only=False):
        self.error_threshold = error_threshold
        self.pair_only = pair_only

    def load_data(self, file_buffer, columns_map):
        df = pd.read_excel(file_buffer, sheet_name='Результаты', header=0, skiprows=[1])
        return df.rename(columns=columns_map)

    def match_data(self, df_virt, df_real):
        matched_results = []
        polymer_values = df_virt['polymer_percent'].dropna().unique()
        for polymer in polymer_values:
            virt_group = df_virt[df_virt['polymer_percent'] == polymer]
            real_group = df_real[df_real['polymer_percent'] == polymer]
            real_copy = real_group.copy()
            if real_group.empty:
                continue
            for _, virt_row in virt_group.iterrows():
                real_copy['diff'] = (
                    np.abs(real_copy['fiber_percent'] - virt_row['fiber_percent'])
                    + (np.abs(real_copy['E_modulus_GPa'] - virt_row['E_modulus_GPa']) / virt_row['E_modulus_GPa']) * 100
                    + (np.abs(real_copy['Fmax_N'] - virt_row['Fmax_N']) / virt_row['Fmax_N']) * 100
                    + (np.abs(real_copy['strength_MPa'] - virt_row['strength_MPa']) / virt_row['strength_MPa']) * 100
                    + (np.abs(real_copy['elongation_percent'] - virt_row['elongation_percent']) / virt_row['elongation_percent']) * 100
                )
                suitable = real_copy[real_copy['diff'] <= self.error_threshold]
                if suitable.empty:
                    continue
                if self.pair_only:
                    closest = suitable.loc[suitable['diff'].idxmin()]
                    matched_results.append((virt_row, closest))
                    real_copy = real_copy.drop(closest.name)
                else:
                    for _, real_row in suitable.iterrows():
                        matched_results.append((virt_row, real_row))
        return matched_results

    def calculate_metrics(self, matched_results):
        metrics = {'Fmax_N': [], 'strength_MPa': [], 'elongation_percent': []}
        match_table = []
        for virt_row, real_row in matched_results:
            match_table.append({
                'virt_polymer%': float(virt_row['polymer_percent']),
                'real_polymer%': float(real_row['polymer_percent']),
                'virt_fiber%': float(virt_row['fiber_percent']),
                'real_fiber%': float(real_row['fiber_percent']),
                'virt_E_modulus_GPa': float(virt_row['E_modulus_GPa']),
                'real_E_modulus_GPa': float(real_row['E_modulus_GPa']),
                'diff': float(real_row['diff'])
            })
            metrics['Fmax_N'].append((real_row['Fmax_N'], virt_row['Fmax_N']))
            metrics['strength_MPa'].append((real_row['strength_MPa'], virt_row['strength_MPa']))
            metrics['elongation_percent'].append((real_row['elongation_percent'], virt_row['elongation_percent']))

        results_metrics = {}
        for param, pairs in metrics.items():
            if pairs:
                y_true, y_pred = zip(*pairs)
                results_metrics[param] = {
                    'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    'MAE': float(mean_absolute_error(y_true, y_pred)),
                    'R2': float(r2_score(y_true, y_pred))
                }
            else:
                results_metrics[param] = {'RMSE': None, 'MAE': None, 'R2': None}
        return results_metrics, match_table, metrics

    def statistical_tests(self, metrics, alpha=0.05):
        stats = {}
        for param, pairs in metrics.items():
            if pairs:
                values = list(zip(*pairs))
                t_stat, p_val = ttest_rel(*values)
                stats[param] = {
                    't_stat': float(t_stat),
                    'p_value': float(p_val),
                    'significant': bool(p_val < alpha)
                }
            else:
                stats[param] = {'t_stat': None, 'p_value': None, 'significant': None}
        return stats

    def evaluate_metrics(self, results_metrics):
        evaluation = {}
        thresholds = {'excellent': 0.8, 'good': 0.6, 'moderate': 0.4}
        for param, m in results_metrics.items():
            r2 = m['R2']
            if r2 is None:
                evaluation[param] = 'No data'
            elif r2 >= thresholds['excellent']:
                evaluation[param] = 'Excellent'
            elif r2 >= thresholds['good']:
                evaluation[param] = 'Good'
            elif r2 >= thresholds['moderate']:
                evaluation[param] = 'Moderate'
            else:
                evaluation[param] = 'Poor'
        return evaluation

    def full_analysis(self, real_buffers, virt_buffers):
        cols_real = {
            'Содержание волокна, %': 'fiber_percent',
            'Eмод': 'E_modulus_GPa',
            'Fmax': 'Fmax_N',
            'sM': 'strength_MPa',
            'dL при Fмакс': 'elongation_percent',
            'Раствор полимера,%': 'polymer_percent'
        }
        cols_virt = {
            'Содержание волокна, %': 'fiber_percent',
            'Eмод, Гпа': 'E_modulus_GPa',
            'Fmax, Н': 'Fmax_N',
            'sM, МПа': 'strength_MPa',
            'dL при Fмакс %': 'elongation_percent',
            'Раствор полимера,%': 'polymer_percent'
        }
        df_real = pd.concat([self.load_data(f, cols_real) for f in real_buffers], ignore_index=True)
        df_virt = pd.concat([self.load_data(f, cols_virt) for f in virt_buffers], ignore_index=True)

        matched = self.match_data(df_virt, df_real)
        results_metrics, matched_experiments, metrics_raw = self.calculate_metrics(matched)
        stats = self.statistical_tests(metrics_raw)
        evals = self.evaluate_metrics(results_metrics)

        return {
            'matched_count':       len(matched),
            'results_metrics':     results_metrics,
            'matched_experiments': matched_experiments,
            'metrics_raw':         metrics_raw,
            'statistical_tests':   stats,
            'metrics_evaluation':  evals
        }

    def generate_excel_report(self, analysis_result):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            pd.DataFrame([analysis_result['results_metrics']]).to_excel(writer, sheet_name='Metrics', index=False)
            pd.DataFrame(analysis_result['matched_experiments']).to_excel(writer, sheet_name='Matched Experiments', index=False)
            pd.DataFrame([analysis_result['metrics_evaluation']]).to_excel(writer, sheet_name='Evaluation', index=False)
        buffer.seek(0)
        return buffer