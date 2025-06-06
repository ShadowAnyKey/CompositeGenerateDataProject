import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel
import io

st.title('📊 Сравнение виртуальных и реальных экспериментов')

# Загрузка файлов
st.header("1. Загрузка файлов")
real_files = st.file_uploader("Загрузите файлы с реальными экспериментами (Excel)", type=['xlsx'], accept_multiple_files=True)
virt_files = st.file_uploader("Загрузите файлы с виртуальными экспериментами (Excel)", type=['xlsx'], accept_multiple_files=True)

# Настройки сравнения
st.header("2. Настройка параметров сравнения")
error_threshold = st.slider("Порог погрешности (%)", 5, 30, 15, 1)
if error_threshold < 15:
    st.warning("⚠️ Уменьшение порога может снизить количество совпадений!")

pair_only = st.checkbox('✅ Сравнивать строго по парам', value=False)

if real_files and virt_files:
    if st.button('✅ Сравнить'):
        matched_results = []
        total_virtual, total_real, lost_matches = 0, 0, 0

        for virt_file in virt_files:
            df_virt = pd.read_excel(virt_file, sheet_name='Результаты', header=0, skiprows=[1]).rename(columns={
                'Содержание волокна, %':'fiber_percent','Eмод, Гпа':'E_modulus_GPa','Fmax, Н':'Fmax_N',
                'sM, МПа':'strength_MPa','dL при Fмакс %':'elongation_percent','Раствор полимера,%':'polymer_percent'
            })
            total_virtual += len(df_virt)

            for real_file in real_files:
                df_real = pd.read_excel(real_file, sheet_name='Результаты', header=0, skiprows=[1]).rename(columns={
                    'Содержание волокна, %':'fiber_percent','Eмод':'E_modulus_GPa','Fmax':'Fmax_N',
                    'sM':'strength_MPa','dL при Fмакс':'elongation_percent','Раствор полимера,%':'polymer_percent'
                })
                total_real += len(df_real)

                polymer_values = df_virt['polymer_percent'].dropna().unique()

                for polymer in polymer_values:
                    virt_group = df_virt[df_virt['polymer_percent'] == polymer]
                    real_group = df_real[df_real['polymer_percent'] == polymer]
                    real_copy = real_group.copy()

                    if real_group.empty:
                        lost_matches += len(virt_group)
                        continue

                    for idx, virt_row in virt_group.iterrows():
                        real_copy['diff'] = (
                            np.abs(real_copy['fiber_percent'] - virt_row['fiber_percent']) +
                            (np.abs(real_copy['E_modulus_GPa'] - virt_row['E_modulus_GPa']) / virt_row['E_modulus_GPa']) * 100 +
                            (np.abs(real_copy['Fmax_N'] - virt_row['Fmax_N']) / virt_row['Fmax_N']) * 100 +
                            (np.abs(real_copy['strength_MPa'] - virt_row['strength_MPa']) / virt_row['strength_MPa']) * 100 +
                            (np.abs(real_copy['elongation_percent'] - virt_row['elongation_percent']) / virt_row['elongation_percent']) * 100
                        )

                        suitable_real = real_copy[real_copy['diff'] <= error_threshold]

                        if suitable_real.empty:
                            lost_matches += 1
                        elif pair_only:
                            closest_idx = suitable_real['diff'].idxmin()
                            matched_results.append((virt_row, real_copy.loc[closest_idx]))
                            lost_matches += (len(suitable_real) - 1)
                            real_copy = real_copy.drop(index=closest_idx)
                        else:
                            for _, real_row in suitable_real.iterrows():
                                matched_results.append((virt_row, real_row))

        if matched_results:
            metrics = {'Fmax_N':[],'strength_MPa':[],'elongation_percent':[]}
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

            results_df = pd.DataFrame([results])
            st.subheader("📌 Итоговые метрики:")
            st.dataframe(results_df)

            # Оценка точности данных
            st.subheader("🔍 Автоматическая оценка точности данных")

            def evaluate_metrics(results):
                conclusions = []
                thresholds = {'excellent': 0.8, 'good': 0.6, 'moderate': 0.4}
                for param in ['Fmax_N_R2', 'strength_MPa_R2', 'elongation_percent_R2']:
                    r2 = results[param]
                    param_name = param.replace("_R2", "")
                    if r2 >= thresholds['excellent']:
                        conclusions.append((param_name, "✅ Отличная точность"))
                    elif r2 >= thresholds['good']:
                        conclusions.append((param_name, "🟡 Хорошая точность, используйте осторожно"))
                    elif r2 >= thresholds['moderate']:
                        conclusions.append((param_name, "🟠 Умеренная точность, требуется проверка"))
                    else:
                        conclusions.append((param_name, "🔴 Низкая точность, не рекомендуется"))
                return conclusions

            conclusions = evaluate_metrics(results)
            for param, conclusion in conclusions:
                st.write(f"**{param}**: {conclusion}")

            if all("✅" in c[1] for c in conclusions):
                st.success("👍 Виртуальные данные точные и отлично подходят для анализа.")
            elif any("🔴" in c[1] for c in conclusions):
                st.error("⚠️ Виртуальные данные имеют низкую точность, не рекомендуются без дополнительной доработки.")
            else:
                st.info("🔍 Виртуальные данные имеют приемлемую точность, рекомендуется дополнительная проверка.")

            st.subheader("📈 Диаграммы рассеяния (виртуальные vs реальные)")
            for m in metrics:
                plt.figure(figsize=(5,4))
                plt.scatter(*zip(*metrics[m]), alpha=0.7)
                plt.xlabel('Реальные значения')
                plt.ylabel('Виртуальные значения')
                plt.title(f'Диаграмма рассеяния: {m}')
                plt.grid(True)
                st.pyplot(plt)

            st.subheader("📦 Ящик с усами погрешностей")
            for m in metrics:
                errors = np.array([r-v for r,v in metrics[m]])
                plt.figure(figsize=(4,3))
                sns.boxplot(x=errors)
                plt.title(f'Ящик с усами ошибок: {m}')
                plt.xlabel('Ошибка')
                plt.grid(True)
                st.pyplot(plt)

            st.subheader("🔬 Статистическая значимость (парный t-тест)")
            for m in metrics:
                t_stat, p_value = ttest_rel(*zip(*metrics[m]))
                signif = "✅ значимо" if p_value < 0.05 else "⚠️ незначимо"
                st.write(f"**{m}**: t={t_stat:.3f}, p-value={p_value:.3f} ({signif})")

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Metrics', index=False)
                pd.DataFrame(match_table).to_excel(writer, sheet_name='Matched Experiments', index=False)

            st.download_button("📥 Скачать отчёт Excel", data=buffer, file_name='final_report.xlsx', mime='application/vnd.ms-excel')
        else:
            st.error("🚫 Совпадений нет. Проверьте порог или данные.")