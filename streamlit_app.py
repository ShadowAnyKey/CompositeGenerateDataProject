import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import io
import seaborn as sns
import configparser
import os

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
        files = []
        for f in real_files:
            files.append(('real_files', (f.name, f.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')))
        for f in virt_files:
            files.append(('virt_files', (f.name, f.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')))

        data = {
            'error_threshold': error_threshold,
            'pair_only': 'true' if pair_only else 'false'
        }

        response = requests.post(f"{API_URL}/classic-analysis/", files=files, data=data)

        if response.ok:
            res_json = response.json()

            st.success(f"✅ Найдено совпадений: {res_json['matched_count']}")

            # Метрики анализа в таблице
            st.subheader("📈 Метрики анализа:")
            # Преобразуем словарь метрик в DataFrame: строки — параметры, столбцы — метрики
            metrics_df = pd.DataFrame(res_json['results_metrics']).T
            # Переупорядочим и отформатируем колонки для отображения
            metrics_df = metrics_df[['RMSE', 'MAE', 'R2']]
            metrics_df.index.name = 'Параметр'
            st.dataframe(metrics_df)

            # Оценка точности данных
            st.subheader("🧐 Оценка точности данных:")
            eval_df = pd.DataFrame(list(res_json['metrics_evaluation'].items()), columns=['Параметр', 'Оценка'])
            st.dataframe(eval_df)

            # Статистические тесты
            st.subheader("📊 Статистические тесты:")
            tests_data = []
            for param, values in res_json['statistical_tests'].items():
                tests_data.append({
                    'Параметр': param,
                    't-статистика': values['t_stat'],
                    'p-значение': values['p_value'],
                    'Статистически значимо?': "✅ Да" if values['significant'] else "❌ Нет"
                })
            tests_df = pd.DataFrame(tests_data)
            st.dataframe(tests_df)

            # Сопоставленные эксперименты
            st.subheader("🔗 Сопоставленные эксперименты:")
            matched_df = pd.DataFrame(res_json['matched_experiments'])
            st.dataframe(matched_df)

            # 1) Scatter «реал vs виртуал» по основным параметрам
            st.subheader("📈 Диаграммы рассеяния (виртуальные vs реальные)")
            for m, pairs in res_json['metrics_raw'].items():
                fig, ax = plt.subplots(figsize=(5,4))
                real_vals, virt_vals = zip(*pairs)
                ax.scatter(real_vals, virt_vals, alpha=0.7)
                ax.plot([min(real_vals+virt_vals), max(real_vals+virt_vals)]*2,
                        [min(real_vals+virt_vals), max(real_vals+virt_vals)]*2,
                        linestyle='--')
                ax.set_xlabel("Реальные значения")
                ax.set_ylabel("Виртуальные значения")
                ax.set_title(f"Диаграмма рассеяния: {m}")
                ax.grid(True)
                st.pyplot(fig)

            # 2) Boxplot распределения ошибок (Real – Virt)
            st.subheader("📦 Ящик с усами погрешностей")
            for m, pairs in res_json['metrics_raw'].items():
                errors = [r - v for r, v in pairs]
                fig, ax = plt.subplots(figsize=(4,3))
                sns.boxplot(x=errors, ax=ax)
                ax.set_title(f"Ящик с усами ошибок: {m}")
                ax.set_xlabel("Ошибка (Real − Virt)")
                ax.grid(True)
                st.pyplot(fig)

        else:
            st.error("Ошибка анализа: проверьте файлы или сервер.")

elif mode == "Нейросетевой анализ":
    # Чтение конфигурации
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.cfg')
    if not os.path.exists(config_path):
        st.error(f"Файл конфигурации не найден: {config_path}")
        st.stop()
    config.read(config_path)
    if 'neural_analysis_ui' not in config:
        st.error("В файле config.cfg отсутствует секция [neural_analysis_ui]")
        st.stop()
    ui_cfg = config['neural_analysis_ui']

    # Значения по умолчанию из cfg
    default_polymer = ui_cfg.getfloat('polymer_solution_pct', 20.0)
    default_length = ui_cfg.getfloat('length_mm', 236.0)
    default_mass = ui_cfg.getfloat('mass_mg', 261.0)
    default_fiber = ui_cfg.getfloat('fiber_content_pct', 72.34)

    st.header("🧠 Анализ с помощью нейросети")

    csv_files = st.file_uploader("Загрузите CSV-файлы для обучения", type="csv", accept_multiple_files=True)
    polymer_solution_pct = default_polymer  # Раствор полимера (%)
    length_mm = default_length            # Длина образца (мм)
    mass_mg = default_mass               # Масса образца (мг)
    fiber_content_pct = default_fiber     # Содержание волокна (%)
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
    skip_initial_rows = st.number_input("Сколько строк пропустить сверху файла?", min_value=0, value=3, step=1)

    show_only_good = st.checkbox("Показывать только хорошие образцы", False)

    if excel_files and st.button("Запустить анализ Excel-файлов"):
        files = [('excel_files', (f.name, f.getvalue())) for f in excel_files]

        data = {'skip_initial_rows': skip_initial_rows}

        response = requests.post(
            f"{API_URL}/excel-sample-analysis/",
            files=files,
            data=data
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