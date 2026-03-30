# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python project centered on PatchTST forecasting experiments. Top-level scripts contain the main workflows:

- `main.py`: loads precomputed arrays and runs training/evaluation.
- `data_preparation.py` and `data_preparation_single.py`: build `X`, `y`, split metadata, and preprocessing artifacts under `tsai/data/`.
- `PatchTST.py`, `ST_PatchTST_model.py`, and `CT_PatchTST_model.py`: model definitions plus training helpers.
- `tsai/data/stations_data*`: raw station CSV inputs.
- `tsai/models/`: exported learners and model artifacts.

Keep new experiment scripts at the repository root unless they justify a reusable package.

## Build, Test, and Development Commands

Use the local Conda environment `aqi-pre`. Required packages include `tsai`, `torch`, `pandas`, `numpy`, `scipy`, and `scikit-learn`.

In Codex, run Python via Conda without `conda activate`:

- `/opt/miniforge/bin/conda run -n aqi-pre python data_preparation.py` — generate multi-station training assets in `tsai/data/`.
- `/opt/miniforge/bin/conda run -n aqi-pre python data_preparation_single.py` — generate single-station assets for the baseline model.
- `/opt/miniforge/bin/conda run -n aqi-pre python main.py` — train the configured model and print evaluation metrics.
- `/opt/miniforge/bin/conda run -n aqi-pre python -m py_compile *.py` — run a quick syntax check across top-level modules.

These scripts expect data files already present under `tsai/data/stations_data/` or `tsai/data/stations_data_Guangzhou/`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for model classes, and explicit imports for local modules. Keep tensor-shape assumptions documented near reshaping code. Favor small helper functions over adding more logic to `main.py`. No formatter or linter is configured here, so keep edits conservative and consistent with surrounding code.

## Testing Guidelines
There is no automated test suite yet. Treat validation as a required smoke test:

- Run `python -m py_compile *.py` before committing.
- Re-run the relevant data preparation script when changing feature engineering.
- Run `python main.py` or the affected training function and confirm shape prints, learning-rate search, and metric output complete without errors.

Name any future tests `test_*.py` so they are easy to adopt under `pytest`.

## Commit & Pull Request Guidelines
Git history is minimal (`initial commit`), so use short imperative commit messages such as `fix station merge logic` or `add ST PatchTST evaluation`. In pull requests, include:

- a brief summary of the modeling or data change,
- the exact command(s) used for validation,
- any dataset assumptions or regenerated artifacts,
- screenshots only when notebook visualizations changed.

Do not commit generated `.npz`, `.pkl`, `.pt`, or `models/` outputs; they are already ignored.
