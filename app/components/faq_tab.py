"""FAQ tab for MitoAgent."""
from __future__ import annotations

FAQ = [
# ── New: tier system, genuine profiles, raw-state diagnostics ─────────
("What is the run-tier selector in the sidebar?",
 "MitoAgent now has three explicit run tiers — smoke, fast, and publication — defined in core/run_settings.py. Each tier sets every numerical budget (DE iterations, profile-likelihood grid, Morris/Sobol N, bootstrap n_boot, within-trace refit settings) from one source of truth. Smoke is for CI sanity checks. Fast is for interactive development. Only publication is reportable, meaning only the publication tier's numbers may legitimately be quoted in a manuscript table."),
("What does the 'Reportable' / 'NOT reportable' badge mean?",
 "The sidebar shows a green 'Reportable' badge when you are running on the publication tier, and an amber 'NOT reportable' badge for smoke or fast. The badge is reproduced on the Dashboard and the Status Card. It is a deliberate engineering guardrail: smoke/fast tiers use budgets too small to produce defensible numbers (for example, Sobol N=4 can produce negative first-order indices), so the UI refuses to mark those numbers as suitable for a report."),
("Why does the Identifiability page distinguish 'genuine profile likelihood' from 'fixed-parameter scan'?",
 "A genuine profile likelihood re-optimises all other parameters at every grid point. On a sloppy likelihood surface (FIM condition ~1e15 or higher for OCR-only inference), other parameters move freely along the flat valley, so the profile likelihood stays low far from the MAP and confidence intervals are correspondingly wide (or one-sided). A fixed-parameter scan holds the others fixed at the MAP, so the likelihood rises steeply along the scanned axis — producing CIs that are too narrow by construction. Only the publication tier runs genuine profiles; fast and smoke run fixed scans, written to a distinct filename (fixed_scans_*.json) so they cannot be confused with profile likelihoods."),
("What does the 'MAP in CI' column mean on the Identifiability page?",
 "It tells you whether the calibration MAP (the parameter value the optimizer returned) falls inside the profile likelihood's confidence interval. On a sloppy OCR-only surface, MAP-outside-CI is a strong non-identifiability signal: the calibration found one point of the flat valley, but profile re-optimisation finds different points are equally good for that parameter. For the bundled dataset_I, 5 of 8 parameters show MAP-outside-CI — direct evidence of the OCR-only practical non-identifiability that the manuscript discusses."),
("What is the difference between 'raw' and 'clipped' state variables?",
 "The ODE solver can produce small negative excursions (round-off) or larger non-physical values during integration. MitoAgent's simulate() now keeps both: r_raw, o_raw, kappa_raw (raw solver output, used by every diagnostic) and r, o, kappa (clipped to physical ranges, used for plotting and calibration residuals). Conservation drift and negative-state counts are computed on the raw arrays, so non-physical excursions cannot be masked by upstream sanitisation. If your real data triggers a 'raw state went negative' warning, the diagnostic is doing its job — investigate before trusting the fit."),
("Why might calibration RMSE be stable but fitted parameters change every run?",
 "Because the OCR-only inverse problem is practically non-identifiable. Benchmarks on dataset_I show calibration RMSE varies ~1% across random seeds (the optimizer reliably finds the same flat valley), but individual parameters can vary 100%+ across seeds because the valley is long and flat. This is not a bug; it is the structural property of OCR-only inference. Calibration's job is to find the valley; the Identifiability page is what characterises it honestly. Do not interpret individual fitted parameter values without checking their profile-likelihood verdict and 'MAP in CI' status."),
("How do I know which tier the result files on disk came from?",
 "Every results/*.json file carries a 'diagnostic_level' field recording the tier ('smoke', 'fast', 'publication', or 'publication-dim12-reduced' for dataset_III). The UI also surfaces tier and reportable status in the structured report exported via Report Builder or Export Results, so downstream readers can audit which numbers came from which tier."),
("What is the 'publication-dim12-reduced' tier I see in dataset_III's profile JSON?",
 "Dataset_III has 4 FCCP injections, which means 12 free parameters versus 9–10 for the other datasets. The inner re-optimisation at every profile-likelihood grid point is therefore about 2× more expensive. To keep the analysis tractable, we use a per-dataset budget override: n_grid=11 (instead of 15), n_restarts_constrained=1 (instead of 2). The 'reduced' in this string refers to the reduced profile-likelihood budget for this high-dimensional case, NOT to a different model — the same 3-state OCR-informed model is used for all datasets. The profile is still genuine: inner re-optimisation, multiple restarts at constrained points. The override is auditable in the JSON's diagnostic_level field."),
("Why is RMSE_test smaller than RMSE_train on the within-trace holdout page?",
 "For an Oroboros stress-test trace, the train segment covers basal, oligomycin, and FCCP-ramp; the held-out test tail is post-inhibition (rotenone + antimycin block the chain). The post-inhibition tail has near-zero OCR by construction, so its observational variance is intrinsically low — RMSE against a flat low-OCR target is small regardless of model fit. The diagnostic is intervention-phase extrapolation, not predictive-performance comparison. The UI now displays an info note explaining this whenever it applies."),
# ── Original FAQ continues ───────────────────────────────────────────
("What is MitoAgent?", "A reproducible analysis platform for 3-state OCR-informed mitochondrial stress-test modeling, diagnostics, reporting, hypothesis prioritization, and experimental-design guidance."),
("Is MitoAgent an AI agent or just a GUI?", "It is more than a GUI: it includes a deterministic backend, CLI, Python API, Streamlit UI, deterministic interpretation layer, and optional LLM-assisted language layer."),
("What are the three goals of MitoAgent?", "Reproducible analysis, cautious hypothesis prioritization, and experimental-design guidance."),
("Can MitoAgent diagnose Alzheimer’s disease?", "No. It cannot diagnose disease from OCR traces. It can only prioritize candidate hypotheses requiring experimental confirmation."),
("Can MitoAgent prove Complex IV dysfunction?", "No. OCR stress tests do not directly measure isolated Complex IV enzymatic activity. Use targeted assays/additional observables for CIV-specific claims."),
("What does CIV-mediated OCR mean?", "Oxygen reduction is mediated through Complex IV in intact respiratory-chain context, but the OCR trace also reflects upstream supply, coupling, protocol effects, and assay conditions."),
("Is this an isolated Complex IV assay?", "No. It is an OCR stress-test modeling framework, not an isolated enzyme assay."),
("What is κ?", "κ is a latent effective respiratory-drive/OCR-permissiveness factor."),
("Is κ membrane potential?", "No. κ is not measured Δψm, true protonmotive force, pH, redox state, or proton gradient."),
("Why can’t OCR alone identify all mechanisms?", "Oxygen traces are one observable affected by multiple coupled processes. Different mechanisms can produce similar OCR patterns."),
("What does identifiability mean?", "Identifiability asks whether the data constrain a parameter enough to interpret it, not merely whether an optimizer returns a value."),
("What is the difference between FIM and profile likelihood?", "FIM is a local diagnostic near the fitted point. Profile likelihood evaluates practical identifiability by scanning/refitting parameter values."),
("Why are some parameters weakly identified or one-sided?", "OCR-only data may constrain only combinations of parameters or only one side of a parameter range. Such parameters should not be used as biological endpoints."),
("What does sensitivity analysis tell me?", "It ranks which parameters or phases the output responds to most strongly. It does not prove that those parameters are identifiable."),
("Does sensitivity analysis prove identifiability?", "No. A parameter can strongly affect model output and still be weakly identified if other parameters can compensate for it."),
("What is Morris screening?", "A global screening method that ranks influential parameters using elementary effects. It is useful for prioritization and debugging."),
("What is Sobol analysis?", "A variance-based global sensitivity method that estimates first-order and total-order parameter effects on a chosen output metric."),
("Why are Sobol total-order indices not additive?", "Total-order indices include interactions, so their sum can exceed one and should not be read as exclusive explained variance."),
("What does validation mean in this tool?", "Validation means technical and workflow diagnostics, such as predictive envelopes or holdouts. It is not proof of biological generalization."),
("What is a technical-replicate transfer check?", "It tests whether a fit from one technical trace/chamber can predict another related technical trace. It is not disease/generalization proof."),
("What is a parametric-bootstrap predictive check?", "It simulates repeated traces from fitted parameters plus an iid observation-noise model to check compatibility with observed residual spread."),
("Why is this not called posterior predictive checking?", "The implemented check uses fitted parameters plus iid noise simulation; it is a parametric-bootstrap predictive check, not posterior sampling."),
("What does LODO validation mean?", "Leave-one-dataset-out validation fits a pooled pattern on all but one dataset and checks transfer to the held-out dataset. It requires suitable replicate datasets."),
("Can I use my own Oroboros data?", "Yes, if the file has time, oxygen columns, and recognizable intervention labels. Always verify parsed events."),
("What data format is required?", "Excel or CSV with a time column, one or more oxygen/O2 columns, and event labels or event timing that can be mapped to start, oligomycin, FCCP, rotenone/antimycin, and end."),
("What if my event labels are different?", "Map custom labels to recognized intervention names before relying on calibration. Always inspect the event table."),
("What if event labels are missing?", "The tool will warn. Calibration can still run only if protocol timing is known; missing or wrong events undermine interpretation."),
("What if my trace has only one FCCP injection?", "The parser supports one FCCP injection and creates a corresponding α₁ response term."),
("What if my trace has multiple FCCP injections?", "The parser supports arbitrary FCCP injections and creates corresponding αⱼ terms for each dose/time."),
("What if calibration fails?", "Check event labels, preprocessing, downsampling, optimizer settings, seed, bounds, and model-scope limitations. Then recalibrate and rerun identifiability."),
("What if the model fits poorly?", "Inspect residuals by phase. Poor fit can reflect wrong events, insufficient optimizer effort, outliers, or real model-scope limits requiring additional observables."),
("What if a parameter is non-identifiable?", "Do not interpret it as a biological endpoint. Use it as a fitted nuisance quantity unless additional measurements or profile results support interpretation."),
("How should I interpret hypothesis-generation output?", "As a prioritized list of candidate explanations and follow-up ideas, not as final biological conclusions."),
("Are the generated hypotheses proven?", "No. They are candidate hypotheses requiring experimental confirmation."),
("What follow-up measurements are recommended?", "Often membrane-potential proxy, redox-state measurement, targeted Complex IV assay, pH/proton-related readout, longer FCCP plateau, denser sampling, and more replicates."),
("How can I export results?", "Use Report Builder for HTML, PDF, JSON, or YAML reports. JSON/YAML are best for reproducibility; HTML/PDF are best for human review."),
("How can I reproduce the paper figures?", "Use the CLI/run_all figure-generation commands documented in the execution guide. The UI is for guided inspection, not the only reproduction path."),
("How can I cite MitoAgent?", "Use the manuscript citation once finalized. Until then, cite the repository/archive version and include the software version or commit hash."),
("Does MitoAgent require an LLM or API key?", "No. All scientific analyses run without an LLM. The LLM layer is optional and only helps with language interaction if configured."),
("What is the difference between CLI, Python API, and Streamlit UI?", "The UI is best for interactive inspection. CLI/API are better for batch analysis, reproducible reruns, automation, logs, and manuscript regeneration."),
("What is the role of the optional LLM layer?", "It can route questions and explain structured backend outputs in natural language. It is not the scientific engine."),
("Does the LLM produce scientific results?", "No. Simulations, estimates, diagnostics, sensitivity indices, validation outputs, and figures come from deterministic Python backend modules."),
("What happens if I ask about an analysis that has not been run?", "Ask MitoAgent should say the required analysis is missing and recommend running it before interpreting the result."),
("How does Ask MitoAgent preserve reproducibility?", "It answers from structured backend outputs and reports the evidence used, caveats, answer mode, and recommended next action."),
]


def render(st):
    st.header("FAQ")
    st.caption(f"{len(FAQ)} frequently asked questions for users, reviewers, and reproducibility engineers.")
    query = st.text_input("Search FAQ", value="", help="Filter questions by keyword, such as calibration, LLM, κ, CLI, validation, or report.")
    q = query.lower().strip()
    shown = 0
    for question, answer in FAQ:
        if q and q not in question.lower() and q not in answer.lower():
            continue
        shown += 1
        with st.expander(question):
            st.write(answer)
    if q:
        st.caption(f"Showing {shown} matching question(s).")
