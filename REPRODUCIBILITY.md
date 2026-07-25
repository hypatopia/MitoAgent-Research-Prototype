# Reproducibility notes

## Recommended clean environment

Create a fresh Python environment and install:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-ui.txt
```

Optional sensitivity methods:

```bash
python -m pip install -r requirements-sensitivity.txt
```

For an exact local freeze after your validated real-data run:

```bash
python -m pip freeze > requirements-lock.txt
```

## Verification commands

```bash
pytest -q -m "not slow"
python run_all.py --smoke
python run_real_data.py --input-dir data_samples --mode smoke --skip-profiles --skip-sensitivity
python -c "import app.streamlit_app"
```

## Reportable real-data mode

Use:

```bash
python run_real_data.py --input-dir data_real --mode publication_real_data
```

The `publication_real_data` tier records all numerical budgets in result provenance. Fast and smoke outputs are diagnostic only and should not be used in manuscript tables or Results text.

## Optional dependency behavior

SALib is optional. Without SALib, Morris and Sobol sensitivity analyses are skipped with an explicit warning. Calibration, numerical diagnostics, identifiability, validation, reporting, and manuscript exports still run.
