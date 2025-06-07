from fastapi import FastAPI, File, UploadFile, Form
from classic_analysis import ClassicAnalyzer
from neural_analysis import NeuralAnalyzer
import pandas as pd
import numpy as np
import io
from sample_analysis import analyze_multiple_excel_files
import tempfile
from fastapi.responses import StreamingResponse

app = FastAPI(title="Combined Analysis API")

@app.post("/excel-sample-analysis/")
async def excel_sample_analysis(
    excel_files: list[UploadFile] = File(...),
    skip_initial_rows: int = Form(3)
):
    results = {}
    with tempfile.TemporaryDirectory() as tmpdirname:
        for excel_file in excel_files:
            file_location = f"{tmpdirname}/{excel_file.filename}"
            with open(file_location, "wb") as f:
                f.write(await excel_file.read())
            results[excel_file.filename] = analyze_multiple_excel_files(
                [file_location], skip_initial_rows=skip_initial_rows
            )[file_location]
    return results

# Эндпоинт для классического анализа
@app.post("/classic-analysis/")
async def classic_analysis(
    real_files: list[UploadFile] = File(...),
    virt_files: list[UploadFile] = File(...),
    error_threshold: int = Form(15),
    pair_only: bool = Form(False)
):
    print(f"pair_only: {pair_only}") 
    analyzer = ClassicAnalyzer(error_threshold=error_threshold, pair_only=pair_only)

    real_buffers = [io.BytesIO(await f.read()) for f in real_files]
    virt_buffers = [io.BytesIO(await f.read()) for f in virt_files]

    analysis_result = analyzer.full_analysis(real_buffers, virt_buffers)

    return analysis_result

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