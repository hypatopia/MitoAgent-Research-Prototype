# Real Oroboros data format guide

This guide describes how to prepare real measured Oroboros OCR/O₂ exports for the 3-state OCR-informed model and MitoAgent workflow.

## 1. Accepted file formats

Preferred formats:

- Excel: `.xlsx` or `.xls`
- CSV: `.csv`

Optional demo/developer format:

- paired NumPy arrays: `*_t.npy` and `*_o.npy`, with event times supplied separately

## 2. Minimum schema

Each Excel/CSV file should contain one time column, one or more oxygen columns, and preferably one sparse event-label column.

Example:

| Time [s] | O2 conc.A [nmol/mL] | O2 conc.B [nmol/mL] | Event |
|---:|---:|---:|---|
| 0 | 200.1 | 199.8 |  |
| 210 | 188.4 | 187.9 | start |
| 300 | 176.2 | 175.6 | oligomycin |
| 480 | 150.4 | 149.7 | FCCP_1 |
| 660 | 121.0 | 120.9 | rotenone+antimycin |
| 826.2 | 118.2 | 118.0 | end |

## 3. Time column detection

The loader searches for column names containing:

- `time`
- `[s]`
- `sec`
- `time_s`

If no named time column is found, it falls back to the first numeric monotone column.

## 4. Oxygen column detection

The loader searches for numeric columns containing terms such as:

- `O2`
- `oxygen`
- `chamber`
- `O2 conc.A`
- `O2 conc.B`

If no named oxygen column is found, it falls back to non-time numeric columns.

## 5. Chamber A/B handling

Each oxygen column becomes one `ChamberTrace`. Chamber A/B labels are inferred from the oxygen column name. If the column name is ambiguous, the original column name is retained.

## 6. Event-label requirements

The model requires at least oligomycin and rotenone/antimycin labels to construct the protocol. FCCP labels are strongly recommended and are required for uncoupling/FCCP-related interpretation.

Accepted event labels are case-insensitive and include:

| Event | Accepted examples |
|---|---|
| Start | `start`, `begin`, `t0`, `baseline` |
| Oligomycin | `oligo`, `oligomycin` |
| FCCP | `FCCP`, `FCCP_1`, `FCCP2`, `uncoupler` |
| Rotenone/antimycin | `rot`, `rotenone`, `antimycin`, `rot/ant`, `rotenone+antimycin`, `RA` |
| End | `end`, `stop`, `finish` |

## 7. Custom event labels

If your export uses custom labels, either rename them in the source file or map them before loading. For example:

- `Inhibitor A` → `oligomycin`
- `Uncoupler dose 1` → `FCCP_1`
- `Rot+AA` → `rotenone+antimycin`

Do not rely on vague comments as event labels.

## 8. Missing or ambiguous labels

The loader records event warnings in `ChamberTrace.metadata["event_warnings"]`. If intervention labels are missing, the workflow should stop before mechanistic interpretation and ask the user to inspect or correct event parsing.

## 9. Multiple FCCP injections

The parser supports arbitrary FCCP titration counts. Examples:

- one FCCP injection: `FCCP_1`
- two injections: `FCCP_1`, `FCCP_2`
- four injections: `FCCP_1`, `FCCP_2`, `FCCP_3`, `FCCP_4`

All FCCP times are sorted by time before protocol construction.

## 10. Preprocessing safeguards

Outlier rejection protects windows around oligomycin, FCCP, and rotenone/antimycin events. Sharp intervention transitions are treated as protocol signal, not electrode glitches.

## 11. Calibration-ready export

After loading a chamber, export a calibration-ready CSV:

```python
from data_io.loader import load_dataset, export_calibration_ready

ds = load_dataset("real_data/experiment_001.xlsx")
ch = ds.chambers[0]
export_calibration_ready(ch, "results/calibration_ready/experiment_001_A.csv", dataset=ds.name)
```

## 12. Interpretation boundary

Real measured data are required for biological interpretation. Demo/synthetic data are only parser and workflow demonstrations.
