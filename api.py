from fastapi import FastAPI, File, UploadFile, Form
from classic_analysis import ClassicAnalyzer
from neural_analysis import NeuralAnalyzer
import pandas as pd
import numpy as np
import io
from sample_analysis import analyze_excel_file
import tempfile
from fastapi.responses import StreamingResponse

app = FastAPI(title="Combined Analysis API")

@app.post("/excel-sample-analysis/")
async def excel_sample_analysis(
    excel_files: list[UploadFile] = File(...),
    tol: float = Form(0.05),
    skip_initial_rows: int = Form(3)
):
    results = {}
    with tempfile.TemporaryDirectory() as tmpdirname:
        for excel_file in excel_files:
            file_location = f"{tmpdirname}/{excel_file.filename}"
            with open(file_location, "wb") as f:
                f.write(await excel_file.read())
            results[excel_file.filename] = analyze_excel_file(
                file_location, tol=tol, skip_initial_rows=skip_initial_rows
            )
    return results

# Эндпоинт для классического анализа
@app.post("/classic-analysis/")
async def classic_analysis(
    virt_file: UploadFile = File(...),
    real_file: UploadFile = File(...),
    error_threshold: int = Form(15),
    pair_only: bool = Form(False)
):
    virt_bytes = await virt_file.read()
    real_bytes = await real_file.read()

    virt_buffer = io.BytesIO(virt_bytes)
    real_buffer = io.BytesIO(real_bytes)

    # Важно! Перемотка указателя в начало файла
    virt_buffer.seek(0)
    real_buffer.seek(0)

    analyzer = ClassicAnalyzer(error_threshold=error_threshold, pair_only=pair_only)
    df_virt, df_real = analyzer.load_data(virt_buffer, real_buffer)

    matches = analyzer.match_data(df_virt, df_real)

    response = {
        "matched_count": len(matches),
        "matches": [
            {
                "virt_fiber_percent": float(match[0]['fiber_percent']),
                "real_fiber_percent": float(match[1]['fiber_percent']),
                "diff": float(match[1]['diff'])
            } for match in matches
        ]
    }

    return response

# Эндпоинт для анализа с нейросетью
@app.post("/neural-analysis/")
async def neural_analysis_multiple_samples(
    csv_files: list[UploadFile] = File(...),
    polymer_solution_pct: float = Form(20.0),
    length_mm: float = Form(236.0),
    mass_mg: float = Form(261.0),
    fiber_content_pct: float = Form(72.34),
    num_samples: int = Form(3)
):
    analyzer = NeuralAnalyzer()

    csv_buffers = [io.BytesIO(await f.read()) for f in csv_files]
    eps_pct, stress = analyzer.load_data(csv_buffers)

    base_params = {
        'polymer_solution_pct': polymer_solution_pct,
        'length_mm': length_mm,
        'mass_mg': mass_mg,
        'fiber_content_pct': fiber_content_pct
    }

    analyzer.fit(eps_pct, stress, base_params)

    params_list = []
    for _ in range(num_samples):
        params_variation = base_params.copy()
        params_variation['fiber_content_pct'] += np.random.uniform(-2.0, 2.0)
        params_variation['polymer_solution_pct'] += np.random.uniform(-1.0, 1.0)
        params_list.append(params_variation)

    excel_buffer = analyzer.generate_multiple_samples(
        params_list=params_list,
        eps_range=(np.min(eps_pct), np.max(eps_pct)),
        num_points=300
    )

    return StreamingResponse(
        excel_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=multiple_predicted_samples.xlsx"}
    )