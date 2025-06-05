import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
from neural_analysis import NeuralAnalyzer

st.title("📊 Интерфейс анализа и генерации данных")

API_URL = "http://127.0.0.1:8000"

mode = st.sidebar.selectbox("Выберите режим анализа:", [
    "Классический анализ",
    "Нейросетевой анализ",
    "Анализ Excel-образцов"
])

if mode == "Классический анализ":
    st.header("🔍 Классический анализ (реальные vs виртуальные)")

    real_files = st.file_uploader("Загрузите реальные эксперименты", type="xlsx", accept_multiple_files=True)
    virt_files = st.file_uploader("Загрузите виртуальные эксперименты", type="xlsx", accept_multiple_files=True)

    error_threshold = st.slider("Порог погрешности (%)", 5, 30, 15)
    pair_only = st.checkbox("Строгое парное сравнение", False)

    if real_files and virt_files and st.button("Запустить классический анализ"):
        files = [('real_files', (f.name, f.getvalue())) for f in real_files] + \
                [('virt_files', (f.name, f.getvalue())) for f in virt_files]

        response = requests.post(f"{API_URL}/classic-analysis/", files=files, data={
            'error_threshold': error_threshold,
            'pair_only': pair_only
        })

        if response.ok:
            res_json = response.json()
            st.success(f"Найдено совпадений: {res_json['matched_count']}")
            st.subheader("Метрики анализа:")
            st.json(res_json['metrics_summary'])
            st.subheader("Статистические тесты:")
            st.json(res_json['statistical_tests'])
        else:
            st.error("Ошибка анализа: проверьте файлы или сервер.")

elif mode == "Нейросетевой анализ":
    st.header("🧠 Анализ с помощью нейросети")

    csv_files = st.file_uploader("Загрузите CSV-файлы для обучения", type="csv", accept_multiple_files=True)
    polymer_solution_pct = st.number_input("Раствор полимера (%)", value=20.0)
    length_mm = st.number_input("Длина образца (мм)", value=236.0)
    mass_mg = st.number_input("Масса образца (мг)", value=261.0)
    fiber_content_pct = st.number_input("Содержание волокна (%)", value=72.34)
    num_samples = st.number_input("Сколько образцов сгенерировать?", min_value=1, max_value=20, value=3, step=1)

    if csv_files and st.button("Запустить нейросетевой анализ"):
        files = [('csv_files', (f.name, f.getvalue())) for f in csv_files]

        response = requests.post(
            f"{API_URL}/neural-analysis/",
            files=files,
            data={
                'polymer_solution_pct': polymer_solution_pct,
                'length_mm': length_mm,
                'mass_mg': mass_mg,
                'fiber_content_pct': fiber_content_pct,
                'num_samples': num_samples
            }
        )

        if response.ok:
            excel_buffer = io.BytesIO(response.content)

            st.download_button(
                label="Скачать Excel с несколькими образцами",
                data=excel_buffer,
                file_name="multiple_predicted_samples.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("Файл успешно создан и готов для скачивания.")
        else:
            st.error("Ошибка анализа: проверьте данные или сервер.")

elif mode == "Анализ Excel-образцов":
    st.header("📈 Анализ Excel-файлов (по листам)")

    excel_files = st.file_uploader("Загрузите Excel-файлы с образцами", type="xlsx", accept_multiple_files=True)
    #tol = st.number_input("Порог падения деформации (%)", min_value=0.0, value=0.05, step=0.01, format="%f")
    skip_initial_rows = st.number_input("Сколько строк пропустить сверху файла?", min_value=0, value=3, step=1)

    show_only_good = st.checkbox("Показывать только хорошие образцы", False)

    if excel_files and st.button("Запустить анализ Excel-файлов"):
        files = [('excel_files', (f.name, f.getvalue())) for f in excel_files]

        response = requests.post(
            f"{API_URL}/excel-sample-analysis/",
            files=files,
            data={
                'tol': tol,
                'skip_initial_rows': skip_initial_rows
            }
        )

        if response.ok:
            results = response.json()
            for file_name, sheets in results.items():
                st.subheader(f"Файл: {file_name}")

                data_for_table = []
                for sheet_result in sheets:
                    if 'error' in sheet_result:
                        if not show_only_good:
                            data_for_table.append({
                                'Лист': sheet_result['sheet'],
                                'Падений деформации': 'Ошибка',
                                'Финальное падение': 'Ошибка',
                                'Наличие пика': 'Ошибка',
                                'Статус': sheet_result['error']
                            })
                    else:
                        status = "✅ Хороший" if sheet_result['is_good_sample'] else "❌ Плохой"
                        if not show_only_good or sheet_result['is_good_sample']:
                            data_for_table.append({
                                'Лист': sheet_result['sheet'],
                                'Падений деформации': str(sheet_result['n_drops']),
                                'Финальное падение': str(sheet_result['final_drop']),
                                'Наличие пика': str(sheet_result['has_peak']),
                                'Статус': status
                            })

                if data_for_table:
                    df_results = pd.DataFrame(data_for_table).astype(str)
                    st.dataframe(df_results)
                else:
                    st.info("Нет подходящих образцов для отображения.")
        else:
            st.error("Ошибка анализа: проверьте файлы или сервер.")