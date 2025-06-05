import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel

class ClassicAnalyzer:
    def __init__(self, error_threshold=15, pair_only=False):
        self.error_threshold = error_threshold
        self.pair_only = pair_only

    def load_single_file(self, file, is_real=True):
        df = pd.read_excel(file, sheet_name='Результаты', header=0)
        columns_map = {
            'Содержание волокна, %':'fiber_percent',
            'Раствор полимера,%':'polymer_percent',
            'Eмод':'E_modulus_GPa',
            'Eмод, Гпа':'E_modulus_GPa',
            'Fmax':'Fmax_N',
            'Fmax, Н':'Fmax_N',
            'sM':'strength_MPa',
            'sM, МПа':'strength_MPa',
            'dL при Fмакс':'elongation_percent',
            'dL при Fмакс %':'elongation_percent'
        }

        df = df.rename(columns=columns_map)
        required_cols = ['fiber_percent', 'polymer_percent', 'E_modulus_GPa', 'Fmax_N', 'strength_MPa', 'elongation_percent']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствующие столбцы: {missing_cols}")

        return df[required_cols]

    def load_data(self, real_files, virt_files):
        df_real_list = [self.load_single_file(f, is_real=True) for f in real_files]
        df_virt_list = [self.load_single_file(f, is_real=False) for f in virt_files]
        return pd.concat(df_real_list, ignore_index=True), pd.concat(df_virt_list, ignore_index=True)

    def match_data(self, df_real, df_virt):
        matched_results = []
        total_virtual, lost_matches = len(df_virt), 0
        polymer_values = df_virt['polymer_percent'].dropna().unique()

        for polymer in polymer_values:
            virt_group = df_virt[df_virt['polymer_percent'] == polymer]
            real_group = df_real[df_real['polymer_percent'] == polymer].copy()

            if real_group.empty:
                lost_matches += len(virt_group)
                continue

            for _, virt_row in virt_group.iterrows():
                real_group['diff'] = (
                    np.abs(real_group['fiber_percent'] - virt_row['fiber_percent']) +
                    np.abs(real_group['E_modulus_GPa'] - virt_row['E_modulus_GPa']) / virt_row['E_modulus_GPa'] * 100 +
                    np.abs(real_group['Fmax_N'] - virt_row['Fmax_N']) / virt_row['Fmax_N'] * 100 +
                    np.abs(real_group['strength_MPa'] - virt_row['strength_MPa']) / virt_row['strength_MPa'] * 100 +
                    np.abs(real_group['elongation_percent'] - virt_row['elongation_percent']) / virt_row['elongation_percent'] * 100
                )

                suitable_real = real_group[real_group['diff'] <= self.error_threshold]

                if suitable_real.empty:
                    lost_matches += 1
                elif self.pair_only:
                    closest_idx = suitable_real['diff'].idxmin()
                    matched_results.append((virt_row, real_group.loc[closest_idx]))
                    real_group.drop(index=closest_idx, inplace=True)
                else:
                    for _, real_row in suitable_real.iterrows():
                        matched_results.append((virt_row, real_row))

        return matched_results, total_virtual, lost_matches

    def calculate_metrics(self, matched_results):
        metrics = {'Fmax_N': [], 'strength_MPa': [], 'elongation_percent': []}
        match_table = []

        for virt_row, real_row in matched_results:
            match_table.append({
                'virt_polymer%': virt_row['polymer_percent'],
                'real_polymer%': real_row['polymer_percent'],
                'virt_fiber%': virt_row['fiber_percent'],
                'real_fiber%': real_row['fiber_percent'],
                'virt_E_modulus': virt_row['E_modulus_GPa'],
                'real_E_modulus': real_row['E_modulus_GPa'],
                'diff %': real_row['diff']
            })

            for m in metrics:
                metrics[m].append((real_row[m], virt_row[m]))

        results = {}
        for m, pairs in metrics.items():
            y_true, y_pred = zip(*pairs)
            results[f'{m}_RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
            results[f'{m}_MAE'] = mean_absolute_error(y_true, y_pred)
            results[f'{m}_R2'] = r2_score(y_true, y_pred)

        return results, match_table, metrics

    def statistical_tests(self, metrics):
        tests = {}
        for m, pairs in metrics.items():
            real, virt = zip(*pairs)
            t_stat, p_val = ttest_rel(real, virt)
            tests[m] = {
                't_stat': t_stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            }
        return tests

    def full_analysis(self, real_files, virt_files):
        df_real, df_virt = self.load_data(real_files, virt_files)
        matches, total_virtual, lost_matches = self.match_data(df_real, df_virt)

        if not matches:
            raise ValueError("Совпадений нет.")

        results, match_table, metrics = self.calculate_metrics(matches)
        tests = self.statistical_tests(metrics)

        return {
            'matched_count': len(matches),
            'total_virtual': total_virtual,
            'lost_matches': lost_matches,
            'results_metrics': results,
            'match_table': match_table,
            'statistical_tests': tests
        }