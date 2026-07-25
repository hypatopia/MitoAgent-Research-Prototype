# Using real measured Oroboros data

## Step 1 — Place files

Create a folder for real measured data:

```bash
mkdir -p real_data
```

Place `.xlsx` or `.csv` exports there.

## Step 2 — Inspect column names

Recommended columns:

```text
Time [s]
O2 conc.A [nmol/mL]
O2 conc.B [nmol/mL]
Event
```

A different naming convention can work if the time, oxygen, and event columns are detectable.

## Step 3 — Load and inspect events

```python
from data_io.loader import load_dataset

ds = load_dataset("real_data/experiment_001.xlsx")
for ch in ds.chambers:
    print(ch.label)
    print("start:", ch.t_start)
    print("oligo:", ch.t_oligo)
    print("fccp:", ch.t_fccp)
    print("rot/ant:", ch.t_inhibit)
    print("end:", ch.t_end)
    print("warnings:", ch.metadata.get("event_warnings", []))
```

If warnings show missing intervention labels, fix the source file or provide explicit column arguments before calibration.

## Step 4 — Choose chamber A or B

The CLI uses a zero-based chamber index:

```bash
python -m agent.cli analyze real_data/experiment_001.xlsx --chamber 0 --fast --out results/agent_reports/experiment_001_A_report.json
python -m agent.cli analyze real_data/experiment_001.xlsx --chamber 1 --fast --out results/agent_reports/experiment_001_B_report.json
```

## Step 5 — Run preprocessing

The package protects intervention windows during outlier rejection. Use preprocessing before calibration and inspect warnings.

```python
from data_io.preprocess import preprocess

ch_clean, warnings = preprocess(ds.chambers[0], do_outliers=True, do_smooth=False)
print(warnings)
```

## Step 6 — Export calibration-ready data

```python
from data_io.loader import export_calibration_ready

export_calibration_ready(ch_clean, "results/calibration_ready/experiment_001_A.csv", dataset=ds.name)
```

## Step 7 — Run the workflow

Diagnostic mode:

```bash
python -m agent.cli analyze real_data/experiment_001.xlsx --chamber 0 --fast --out results/agent_reports/experiment_001_A_report.json
```

Publication-level analysis should be run only after CHUNK 5–8 calibration, identifiability, sensitivity, and validation upgrades are complete.

## Step 8 — Replace demo results

Do not mix demo/synthetic and real-data outputs in publication claims. For real-data publication results, regenerate:

- calibration outputs
- phase summaries
- identifiability profiles
- sensitivity results
- validation diagnostics
- figures
- tables
- final report

## Step 9 — Interpretation limitations

Real OCR data alone still cannot prove disease status, isolated Complex IV dysfunction, membrane potential, redox state, or proton-gradient dynamics. Treat MitoAgent hypothesis output as candidate hypotheses requiring experimental confirmation.
