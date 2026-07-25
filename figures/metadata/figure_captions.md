# Final figure captions and source-data links

## fig1_model_overview: Model overview and biological scope

- Script: `figures/make_fig_step1.py`
- Source data/modules: `core/reduced_model.py; core/protocols.py`
- Caption: Biological and modeling scope. The workflow observes O2(t)/OCR only and models CIV-mediated oxygen consumption in intact respiratory-chain context; it does not measure membrane potential, redox state, pH, proton gradient, or isolated Complex IV activity.
- Status: diagnostic/demo unless regenerated on real measured data.

## fig2_reduced_model: 3-state model formulation

- Script: `figures/make_fig_step2.py`
- Source data/modules: `core/reduced_model.py; manuscript/model.tex`
- Caption: 3-state model formulation. The model states are r, o, and kappa; smooth intervention functions encode oligomycin, FCCP additions, and rotenone/antimycin inhibition. kappa is a latent OCR-permissiveness factor, not measured membrane potential.
- Status: diagnostic/demo unless regenerated on real measured data.

## fig3_data_pipeline: Data pipeline and event parsing

- Script: `figures/make_fig_step3.py`
- Source data/modules: `data_io/loader.py; data_io/preprocess.py; data_samples/*.xlsx`
- Caption: Data pipeline for demo and real Oroboros-style data. Excel/CSV inputs are parsed, chamber columns and event labels are detected, injection windows are protected during preprocessing, and calibration-ready exports are generated with warnings.
- Status: diagnostic/demo unless regenerated on real measured data.

## fig4_calibration: Calibration fits and residuals

- Script: `figures/make_fig_step4.py`
- Source data/modules: `results/calibration/*.json; results/calibration/calibration_summary.csv; results/calibration/phase_summary_dataset_*.json`
- Caption: Calibration fits and residuals for demonstration traces. Fit overlays and residual panels demonstrate software execution only; biological interpretation requires real measured data.
- Status: diagnostic/demo unless regenerated on real measured data.

## fig5_identifiability: Identifiability diagnostics

- Script: `figures/make_fig_step5.py`
- Source data/modules: `results/identifiability/fim_dataset_I.json; results/identifiability/profiles_dataset_I.json; results/identifiability/parameter_interpretability_flags.csv`
- Caption: Identifiability diagnostics. FIM eigenvalues are local diagnostics; fast-mode scans are not publication-grade profile likelihoods. OCR-only data do not support interpretation of all parameters.
- Status: diagnostic/demo unless regenerated on real measured data.

## fig6_sensitivity: Sensitivity diagnostics

- Script: `figures/make_fig_step6.py`
- Source data/modules: `results/sensitivity/morris_dataset_I.json; results/sensitivity/sobol_auc_dataset_I.json; results/sensitivity/time_resolved_sobol_dataset_I.npz`
- Caption: Sensitivity diagnostics. Morris and Sobol outputs identify influential parameters and phases, but sensitivity does not prove identifiability. Sobol total-order indices include interactions and are not additive.
- Status: diagnostic/demo unless regenerated on real measured data.

## fig7_validation: Validation diagnostics

- Script: `figures/make_fig_step7.py`
- Source data/modules: `results/validation/*.json; results/validation/*.csv`
- Caption: Validation diagnostics. Chamber transfer, within-trace holdout, LODO status, and parametric-bootstrap predictive checks are workflow diagnostics, not proof of biological generalization.
- Status: diagnostic/demo unless regenerated on real measured data.

## fig8_agent_architecture: MitoAgent architecture

- Script: `figures/make_fig_step8.py`
- Source data/modules: `agent/*.py; app/streamlit_app.py`
- Caption: MitoAgent architecture. The deterministic backend generates numerical outputs; the optional natural-language layer routes questions and explains structured outputs with caveats.
- Status: diagnostic/demo unless regenerated on real measured data.

## fig9_streamlit_ui_overview: Streamlit interface overview

- Script: `figures/make_fig_step9.py`
- Source data/modules: `app/streamlit_app.py; app/components/*.py`
- Caption: Streamlit interface overview schematic. User-facing tabs expose deterministic backend outputs; UI does not create independent numerical results.
- Status: diagnostic/schematic.

## fig10_ask_mitoagent: Ask MitoAgent interpretation pathway

- Script: `figures/make_fig_step10.py`
- Source data/modules: `agent/ask_agent.py; agent/question_router.py; agent/safety_rules.py; agent/interpretation.py`
- Caption: Ask MitoAgent interpretation pathway. Natural-language answers are generated only from structured deterministic backend outputs; the optional LLM does not produce scientific results.
- Status: diagnostic/schematic.

