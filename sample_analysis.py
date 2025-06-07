import pandas as pd

def analyze_single_dataframe(df: pd.DataFrame, sheet_name: str = ""):
    if df.shape[1] < 2:
        raise ValueError(f"Лист '{sheet_name}': нужно ≥2 колонки, найдено {df.shape[1]}.")

    deform = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    stress = pd.to_numeric(df.iloc[:, 1], errors='coerce')

    if deform.isna().any() or stress.isna().any():
        raise ValueError(f"Лист '{sheet_name}': найдены нечисловые значения.")

    idx_peak = int(stress.idxmax())
    has_peak = bool(idx_peak < len(stress) - 1)

    post_peak_deform = deform.iloc[idx_peak + 1:]
    diffs_post = post_peak_deform.diff().fillna(0)
    n_drops = int((diffs_post < 0).sum())
    final_drop = bool(deform.iloc[-1] < deform.iloc[idx_peak])

    return {
        'sheet': sheet_name,
        'n_drops': n_drops,
        'final_drop': final_drop,
        'has_peak': has_peak,
        'is_good_sample': not final_drop and has_peak
    }

def analyze_excel_file(path: str, skip_initial_rows: int = 3):
    results = []
    with pd.ExcelFile(path) as xls:
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None, skiprows=skip_initial_rows)
                result = analyze_single_dataframe(df, sheet_name)
            except Exception as e:
                result = {
                    'sheet': sheet_name,
                    'error': f"{str(e)}"
                }
            results.append(result)
    return results

def analyze_multiple_excel_files(paths: list, skip_initial_rows: int = 3):
    results = {}
    for path in paths:
        results[path] = analyze_excel_file(path, skip_initial_rows)
    return results