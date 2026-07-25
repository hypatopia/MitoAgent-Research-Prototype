# Real-data runbook for paper-result production

This package is configured for the first-author decision that the **primary analysis uses chamber-level traces**. For each real dataset, chamber A and chamber B are analysed separately. Averaged A/B traces are available only as a **supplementary sensitivity analysis** after event alignment and chamber agreement have been inspected.

## 1. Put real files in `data_real/`

Accepted formats: `.xlsx`, `.xls`, `.csv`.

Recommended names:

```text
data_real/dataset_I_real.xlsx
data_real/dataset_II_real.xlsx
data_real/dataset_III_real.xlsx
```

Each file should contain time, chamber A oxygen, chamber B oxygen, and intervention labels for oligomycin, each FCCP dose, and rotenone/antimycin.

## 2. Install dependencies

Core pipeline and UI:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-ui.txt
```

Optional Morris/Sobol sensitivity analyses:

```bash
python -m pip install -r requirements-sensitivity.txt
```

If SALib is not installed, calibration, identifiability, diagnostics, validation, reports, and manuscript exports still run. Morris/Sobol sensitivity outputs are skipped with a warning.

## 3. Run a smoke check first

```bash
python run_real_data.py --input-dir data_real --mode smoke --skip-profiles --skip-sensitivity
```

Expected output folders:

```text
results_real/
figures_real/
manuscript/tables_real/
reports_real/
exports_real/
```

## 4. Run the recommended publication real-data workflow

```bash
python run_real_data.py --input-dir data_real --mode publication_real_data
```

This runs chamber-level analyses for every detected chamber in every Excel/CSV file.

## 5. Optional supplementary averaged-chamber analysis

Use only after checking chamber agreement:

```bash
python run_real_data.py --input-dir data_real --mode publication_real_data --include-averaged
```

Averaged traces are labelled `A_B_average_supplementary` and should not replace the primary chamber-level analysis.

## 6. Outputs to use for manuscript updates

Calibration summaries:

```text
results_real/calibration/
manuscript/tables_real/real_calibration_summary.csv
```

Numerical diagnostics:

```text
results_real/diagnostics/
```

Identifiability:

```text
results_real/identifiability/
```

Sensitivity, if SALib is installed:

```text
results_real/sensitivity/
```

Validation:

```text
results_real/validation/
```

Figures:

```text
figures_real/
```

Manuscript draft text:

```text
manuscript/real_results_paragraphs.md
```

Machine-readable run summary:

```text
exports_real/real_data_run_summary.json
```

## 7. Chamber averaging decision rule

Do not average chambers before primary modeling. Average only as supplementary if:

- chamber A and B event times are aligned;
- basal slopes are similar;
- FCCP responses are comparable;
- post-inhibition residuals are comparable;
- chamber-transfer diagnostics do not show major disagreement.

## 8. Manuscript framing

Recommended wording:

> Each dataset contained paired chamber measurements. To preserve technical-replicate information and avoid masking chamber-specific artifacts, all primary analyses were performed at the chamber level. Calibration, residual analysis, numerical diagnostics, identifiability analysis, sensitivity analysis, and validation diagnostics were conducted separately for each chamber. Dataset-level summaries were then derived from the paired chamber results. Averaged-chamber traces were considered only as a secondary sensitivity analysis after confirming event alignment and chamber-level agreement.
