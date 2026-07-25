# MitoAgent Streamlit UI

Run:

```bash
streamlit run app/streamlit_app.py
```

The UI uses a modern side-navigation layout with grouped pages:

- Project & Data
- Model & Analysis
- Interpretation
- Outputs
- Learning & Settings

The UI is designed for interactive inspection and single-session analysis. For reproducible paper-result production across all real datasets, use:

```bash
python run_real_data.py --input-dir data_real --mode publication_real_data
```

The UI uses user-facing scientific labels for parameters. Raw code names are reserved for developer/API contexts and machine-readable JSON/YAML exports.
