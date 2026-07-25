"""Runbook-style Help Hub for MitoAgent Streamlit UI."""
from __future__ import annotations


def render(st):
    st.header("Help Hub — model, agent, UI, and reproducibility runbook")
    st.markdown(
        """
<div class="mito-card">
<h4>How to use this hub</h4>
<p>This is the learning and operating manual for MitoAgent. Use it before running real Oroboros data, before interpreting fitted parameters, and before exporting a report.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    sections = st.tabs([
        "Overview",
        "Model + equations",
        "Run tiers & reportability",
        "Workflow runbook",
        "Calibration help",
        "Identifiability help",
        "Sensitivity help",
        "Validation help",
        "Diagnostics help",
        "Hypothesis tab",
        "Design guidance tab",
        "Ask MitoAgent",
        "Optional NL agent",
        "UI vs CLI/API",
        "Report Builder",
        "Real data",
        "Warnings",
        "Methods references",
    ])

    with sections[0]:
        st.subheader("What MitoAgent is")
        st.markdown("""
MitoAgent is a reproducible analysis-orchestration platform for 3-state OCR-informed mitochondrial stress-test modeling. It combines:

- deterministic backend modules for data loading, event parsing, preprocessing, simulation, calibration, diagnostics, identifiability, sensitivity, validation, figures, and reports;
- a Streamlit interface for guided inspection and interpretation;
- a CLI/Python API for reproducible batch analysis;
- an optional natural-language layer for explaining already-generated backend outputs.

Its three goals are **reproducible analysis**, **hypothesis prioritization**, and **experimental-design guidance**.
""")
        st.subheader("What MitoAgent is not")
        st.markdown("""
MitoAgent is not a diagnostic tool for Alzheimer’s disease, not proof of Complex IV dysfunction, not an isolated Complex IV enzyme assay, not a direct membrane-potential measurement, and not a black-box LLM system that invents numerical results.
""")

    with sections[1]:
        st.subheader("3-state OCR-informed model")
        st.markdown("""
The model represents **CIV-mediated OCR in intact respiratory-chain context**. Oxygen/OCR is the measured observable; upstream supply and protonmotive/coupling effects are represented phenomenologically.

- **κ** is a latent effective respiratory-drive / OCR-permissiveness factor.
- **κ is not measured Δψm**, true protonmotive force, pH, redox state, or proton-gradient measurement.
- **Vₘₐₓ** is an effective CIV-mediated OCR-capacity parameter, not isolated Complex IV enzymatic activity.
- FCCP response amplitudes such as **α₁** are FCCP-response indicators, not direct proton-leak measurements.

### Mathematical formulation

The model uses three state variables: an effective reduced cytochrome-*c*/reductant pool state **r(t)**, oxygen concentration **o(t)**, and latent respiratory-drive factor **κ(t)**. The direct observable is oxygen concentration; OCR is derived from the model flux.
""")
        st.latex(r"\frac{dr}{dt} = 2v_{\mathrm{sup}} - 2v_{\mathrm{CIV}}")
        st.latex(r"\frac{do}{dt} = -\frac{1}{2}v_{\mathrm{CIV}}")
        st.latex(r"\frac{d\kappa}{dt} = \frac{\kappa_{\mathrm{eq}}(t)-\kappa}{\tau_{\kappa}}")
        st.latex(r"\mathrm{OCR}(t)=\frac{1}{2}v_{\mathrm{CIV}}(t)")
        st.markdown("""
The CIV-mediated oxygen-reduction flux is bounded by oxygen availability, reduced-pool availability, and the latent drive term. Protocol injections are represented through smooth event functions for oligomycin, FCCP doses, and rotenone/antimycin.

Key user-facing parameters include:

| Symbol | Meaning |
|---|---|
| **kₛ** | Effective upstream supply rate |
| **cₜ** | Total reduced-pool scale |
| **Vₘₐₓ** | Maximum effective CIV-mediated OCR capacity |
| **Kₒ** | Oxygen affinity scale |
| **Kᵣ** | Reduced-pool affinity scale |
| **γₒ** | Oligomycin response factor |
| **τₖ** | Drive relaxation time |
| **r₀** | Initial reduced-pool state |
| **αⱼ** | FCCP response amplitude for injection *j* |
| **σₒᵦₛ** | Within-trace observational-noise estimate |
""")

    with sections[2]:
        st.subheader("Run tiers and the reportable badge")
        st.markdown("""
MitoAgent has **three explicit run tiers** defined in
`core/run_settings.py`. The sidebar selector and a coloured badge
("Reportable" / "NOT reportable") tell you which tier you are on at all
times. The badge is reproduced on the Dashboard and on the Status Card.

| Tier | Purpose | Reportable? | DE / Sobol / Profile budgets |
|---|---|---|---|
| **smoke** | CI / import sanity. Cannot produce defensible numbers. | No | DE 2×2 · Sobol N=4 · fixed-scan only |
| **fast** | Interactive development and learning. | No | DE 25×10 · Sobol N=32 · fixed-scan only |
| **publication** | Benchmark-validated. The only tier whose numbers may be quoted in a manuscript. | **Yes** | DE 25×12 · Sobol N=512 · **genuine** profile likelihoods (inner re-optimisation), n_grid=15 |

### What changes when you switch tiers

- **Calibration page** pulls `de_maxiter`, `de_popsize`, and `de_polish`
  from the tier; the widget defaults follow the selector.
- **Identifiability page** branches on `run_cfg.profile_real`. On the
  publication tier it runs **genuine profile likelihoods** (every other
  parameter is re-optimised at every grid point). On fast/smoke it runs
  a fixed-other-parameter scan and shows a prominent warning that the
  resulting CIs are too narrow on sloppy surfaces.
- **Sensitivity page** uses tier-specific Morris `N_trajectories`,
  Sobol `N_base`, and time-resolved Sobol `N`. Publication N_base=512 is
  the benchmarked threshold above which first-order indices are stable
  on this model.
- **Validation page** uses tier-specific bootstrap `n_boot` and
  within-trace holdout DE budget. The within-trace holdout always
  refits parameters on the training segment (no residual-split
  shortcut).
- **Status card** and **structured export** record the tier and the
  `reportable` flag in JSON/YAML so downstream readers can audit which
  numbers came from which tier.

### Reportable vs not reportable

The "Reportable" badge is a deliberate guardrail. Smoke and fast tier
budgets are intentionally small (smoke takes seconds; fast takes
~30 seconds) and produce diagnostics that are useful for development
but not defensible in a manuscript. For example, Sobol with N=4 can
produce first-order indices that are negative within their confidence
intervals. Use fast for interactive work; switch to publication
before quoting any number externally.

### Tier overrides for high-dimensional datasets

Datasets with many FCCP injections have more free parameters. The
profile-likelihood inner re-optimisation cost scales with that
dimensionality. To keep the publication-tier analysis tractable on
dataset_III (4 FCCP injections, 12 parameters), the package applies a
`publication-dim12-reduced` override: `n_grid=11` and
`n_restarts_constrained=1` instead of the 15 / 2 defaults. The
override is auditable in every JSON's `diagnostic_level` field. The
profile is still genuine; only the grid and restarts are trimmed.
""")

    with sections[3]:
        st.subheader("Recommended workflow")
        st.markdown("""
**Before you start.** Set the sidebar **Analysis mode** to the
appropriate tier. Use **fast** for interactive learning; switch to
**publication** before any output is intended for a manuscript. The
sidebar badge tells you whether the current tier is reportable.

1. **Load data** and verify chamber selection.
2. **Inspect event parsing** before any fitting.
3. **Preprocess** with injection-window protection.
4. **Simulate** with default or previously calibrated parameters.
5. **Calibrate** and inspect residuals by phase. Calibration-page
   widgets now default to the tier's benchmarked DE budget.
6. **Run numerical diagnostics** before interpretation. Raw negative-state
   warnings (if any) come from the unclipped solver output and should
   be taken seriously.
7. **Run identifiability** before treating fitted parameters as
   meaningful. On the publication tier the page runs **genuine profile
   likelihoods**; on fast/smoke it runs a fixed-other-parameter scan
   and warns you that the resulting CIs are too narrow. Inspect the
   **MAP-in-CI** column — ✗ indicates the calibration MAP sits outside
   the profile CI, a strong non-identifiability signal.
8. **Run sensitivity** to identify high-information parameters/phases.
9. **Run validation** to assess technical/workflow transfer, not
   biological truth. The within-trace holdout always refits parameters
   on the training segment.
10. **Use hypothesis/design guidance** to plan follow-up experiments.
11. **Ask MitoAgent** targeted interpretation questions.
12. **Export a report** in HTML/PDF for review or JSON/YAML for
    reproducibility. Exports record `tier` and `reportable` so
    downstream readers can audit which tier each number came from.
13. **Regenerate manuscript figures** on the Manuscript Figures page
    (under OUTPUTS). It runs the publication figure scripts from
    `figures/make_fig_step*.py` against the current `results/` tree and
    previews each figure inline, so you can reproduce Figs 4–6 (and any
    other figure in the manuscript) from inside the UI.
""")
        st.caption("Use smoke for CI sanity, fast for learning/debugging, publication for any number intended for a manuscript.")

    with sections[4]:
        st.subheader("How users can improve calibration")
        st.markdown("""
The Calibration page defaults to the **current tier's benchmarked DE
budget**. Switching the sidebar tier reloads the widgets to the new
defaults (the widgets are tier-keyed so values do not carry stale
state across switches).

If the plot shows that one phase fits poorly, users do **not** have to passively accept the first fit. They can:

- verify event labels and injection times;
- inspect whether the poor fit occurs near an injection transition;
- increase the number of downsampled fitting points;
- increase differential-evolution iterations/population size;
- rerun with a different random seed;
- switch from fast mode to publication mode;
- adjust preprocessing/outlier settings cautiously;
- check whether the phase is outside the model’s scope;
- run identifiability after recalibration before interpreting parameter values.

If repeated settings still leave structured residuals, the correct conclusion may be that the data need additional observables or that the model cannot explain that phase.
""")

    with sections[5]:
        st.subheader("Identifiability help")
        st.markdown("""
The Identifiability page answers a question calibration cannot: **given
this trace, which fitted parameters are actually interpretable?**

### What the page shows

1. **FIM diagnostic** — a local sensitivity-matrix check at the fitted
   point. It reports the raw and clipped condition numbers and an
   eigenvalue-spectrum plot. The FIM is a local indicator only; it
   cannot assign per-parameter verdicts on its own. The UI is honest
   about this and does not print "weak" against every row.
2. **Profile-likelihood / fixed-parameter scan section** — the heading
   changes based on the current run tier:
   - On **publication** tier the heading reads
     *"Genuine profile likelihood (inner re-optimisation)"*. Every
     other parameter is re-optimised at every grid point. CIs are
     correct even on sloppy surfaces.
   - On **fast or smoke** tier the heading reads
     *"Fixed-other-parameter scan (NOT a profile likelihood)"* with
     an amber warning that CIs from this path are too narrow on sloppy
     surfaces. Use this for quick debugging only.
3. **Per-parameter verdict table** populated by your session's runs,
   sorted by verdict (identifiable → one-sided → weakly identified →
   non-identifiable → unresolved). Each row shows MAP, CI low, CI high,
   and the **MAP-in-CI** flag (✓ or ✗) plus the analysis source
   (genuine profile vs fixed scan).
4. **Profile-likelihood plot** for the currently-selected parameter.

### What 'MAP outside CI' means

If the MAP-in-CI flag is ✗ for a parameter, the calibration MAP sits
outside the profile likelihood's confidence interval. On a sloppy
OCR-only likelihood surface this is a **strong non-identifiability
signal**: the calibration optimizer found one point of the flat
valley, but profile re-optimisation finds different points are equally
good. For the bundled dataset_I, 5 of 8 parameters are MAP-outside-CI.

### Verdict categories

- **identifiable** — two-sided CI present and reasonably tight.
- **one-sided** — only one side of the CI is bounded; the other side is
  effectively unconstrained.
- **weakly identified** — the likelihood degrades slowly along the
  scanned axis.
- **non-identifiable** — the likelihood is flat along the scanned axis.
- **unresolved** — optimizer failed too many times to assign a verdict.

Do not interpret a non-identifiable or weakly-identified parameter as
a biological endpoint. Treat it as a fitted nuisance quantity and rely
on the identifiable ones plus, in real data, additional measurements.
""")

    with sections[6]:
        st.subheader("Sensitivity help")
        st.markdown("""
The Sensitivity page tells you **which parameters and protocol phases
the model output responds to most strongly**, which is different from
identifiability.

### What the page runs

- **Morris screening** — global elementary-effects screening with
  `N_trajectories` set from the current tier (publication = 20).
- **Sobol AUC** — variance-based first-order (S₁) and total-order (Sₜ)
  indices on the trace AUC, with `N_base` from the current tier
  (publication = 512, the benchmarked threshold above which S₁ is
  non-negative within CI on this smooth 4-event model).
- **Time-resolved Sobol** — per-timepoint Sₜ across the protocol,
  with variance-degenerate post-inhibition points flagged.

### Sensitivity vs identifiability

A parameter can be highly sensitive (the output moves a lot when it
moves) and still be non-identifiable, because other parameters can
compensate for it. The two analyses are complementary, not
substitutes. Always read the Sensitivity page alongside the
Identifiability page.

### Why total-order indices can sum to more than one

Total-order Sobol indices include all interactions involving a
parameter, so the same interaction contributes to multiple parameters'
Sₜ values. Do not read total-order indices as "fraction of variance
exclusively explained". Use first-order S₁ for that interpretation,
and use the gap between S₁ and Sₜ as evidence of how interaction-rich
the parameter is.

### Tier caveats

If the sidebar shows a **NOT reportable** tier, the page displays an
amber warning that N_base is below the publication threshold and the
first-order indices may be unstable.
""")

    with sections[7]:
        st.subheader("Validation help")
        st.markdown("""
The Validation page runs **diagnostics that separate technical
transfer from biological generalisation**, not proof of biological
validity.

### What the page runs

- **Parametric-bootstrap predictive check** — simulates `n_boot`
  traces (publication = 500) from the fitted parameters plus an iid
  observation-noise model and reports empirical 90% coverage. The
  configurable warning band (sidebar) flags coverage that falls
  outside an acceptable range.
- **Within-trace holdout (refit-based)** — fits parameters on the
  first 70% of the trace (basal + oligomycin + FCCP-ramp) and
  evaluates RMSE on the held-out tail (post-inhibition). At every
  tier the holdout **actually refits**; tiers only set the inner DE
  budget. There is no residual-split shortcut.

### Why is RMSE_test smaller than RMSE_train?

For an Oroboros stress-test trace, the held-out post-inhibition tail
has near-zero OCR by construction (rotenone + antimycin block the
chain). Its observational variance is intrinsically low, so RMSE
against that target is small regardless of model fit. The UI displays
an info note when this happens. The diagnostic is **intervention-phase
extrapolation**, not predictive-performance comparison.

### What this page is NOT

- Not biological validation.
- Not a posterior predictive check (the bootstrap uses fitted
  parameters plus iid noise, not posterior samples).
- Not proof that the model generalises across disease vs control or
  across cell lines.

LODO (leave-one-dataset-out) transfer requires real replicate datasets
and is deferred until your real Oroboros data are available.
""")

    with sections[8]:
        st.subheader("Numerical Diagnostics tab")
        st.markdown("""
The Diagnostics tab is a **numerical audit**, not a biological interpretation page. It checks whether the current simulation/fitted parameters are numerically credible enough to inspect further.

It currently reports:

- solver convergence;
- oxygen monotone non-increasing check;
- κ finite-range check;
- tolerance robustness;
- Jacobian/stiffness indicators;
- cytochrome/reductant-pool drift indicator;
- raw negative-state counts for r and o;
- warnings.

### Raw vs clipped states (important for honest diagnostics)

The ODE solver can produce small negative excursions (round-off) or
larger non-physical values during integration. MitoAgent's `simulate()`
keeps both versions in the result object:

- `r_raw`, `o_raw`, `kappa_raw` — the **raw solver output**. These are
  what `conservation_check`, `detect_instability`, and the negative-state
  counts inspect. If the solver excurses into non-physical regions, the
  diagnostics will see it here.
- `r`, `o`, `kappa` — clipped to physical ranges. These are used for
  plotting and calibration residuals only.

Earlier versions clipped states before returning, which made conservation
drift and the negative-state count vacuously zero by construction. The
fix means diagnostics now report honest numbers: if a "raw state went
negative" warning appears on your real data, investigate it before
trusting the fit.

The dropdown/expander is for reviewer and reproducibility details. The main UI shows badges and readable metrics; the raw object is intentionally hidden under a machine-readable audit expander.
""")

    with sections[9]:
        st.subheader("Hypothesis tab")
        st.markdown("""
The Hypothesis tab turns phase-level OCR summaries, calibration quality, identifiability warnings, sensitivity results, and validation diagnostics into **candidate hypotheses requiring experimental confirmation**.

Users can use it to ask: which phase looks most informative, whether an FCCP phenotype deserves follow-up, and which ambiguity should be tested experimentally. It must not be read as disease diagnosis or proof of Complex IV dysfunction.
""")

    with sections[10]:
        st.subheader("Design Guidance tab")
        st.markdown("""
The Design Guidance tab converts uncertainty into concrete next experiments. It recommends additional measurements, replicates, or protocol refinements such as membrane-potential proxy, redox-state readout, targeted Complex IV activity assay, pH/proton-related readout, longer FCCP plateau, denser sampling around injections, and additional biological/technical replicates.
""")

    with sections[11]:
        st.subheader("Ask MitoAgent tab")
        st.markdown("""
Ask MitoAgent answers natural-language questions using structured backend outputs only. Good questions include:

- Which parameters should I trust?
- Why is my FCCP response low?
- Can this trace suggest Complex IV dysfunction?
- What follow-up experiment would reduce uncertainty?
- Why did calibration fit poorly?

Every answer reports answer mode, backend evidence used, caveats, and recommended next action.
""")

    with sections[12]:
        st.subheader("Optional Natural-Language Agent tab")
        st.markdown("""
This tab checks/configures optional LLM-assisted explanation. The default mode is deterministic/offline and requires no API key.

Users can:

- check whether an LLM provider appears configured;
- run the deterministic command router;
- use LLM-assisted explanation if a provider is configured.

Users cannot make the LLM estimate parameters, invent diagnostics, override calibration/identifiability/sensitivity/validation, diagnose disease, or prove mechanism.

To enable LLM-assisted mode, configure the provider expected by `agent/llm_driver.py` through environment variables or local secrets, restart Streamlit, and use it only for language explanation of backend outputs.
""")

    with sections[13]:
        st.subheader("UI versus CLI/API")
        st.markdown("""
**Use the UI** for interactive inspection, event checking, visual diagnostics, education, and one-session reports.

**Use CLI/API** for batch runs, exact reruns, real-data studies with many files, automated tests, logs, versioned outputs, and manuscript regeneration.

Example:

```bash
python -m agent.cli analyze data_samples/dataset_I.xlsx --out results/agent_reports/dataset_I_report.json --fast
```
""")

    with sections[14]:
        st.subheader("Report Builder")
        st.markdown("""
The Report Builder creates customized reports from the current session. Use:

- **HTML** for polished browser review with tables and figures;
- **PDF** for printable review;
- **JSON** for exact machine-readable reproducibility;
- **YAML** for readable configuration/versioning.

Select all sections for a comprehensive report or choose a focused report for calibration, diagnostics, identifiability, hypothesis, or design guidance.
""")

    with sections[15]:
        st.subheader("Real Oroboros data")
        st.markdown("""
Real data should include a time column, oxygen columns, and recognizable event labels for start, oligomycin, FCCP injections, rotenone/antimycin, and end. Always verify parsed events before calibration. Wrong events can create residual patterns that look like model failure.
""")

    with sections[16]:
        st.subheader("Warning types")
        st.markdown("""
- **Data warnings**: missing/ambiguous columns or events.
- **Numerical warnings**: solver, oxygen monotonicity, κ range, tolerance/stiffness issues.
- **Identifiability warnings**: weak, one-sided, flat, unresolved, optimizer failure, high FIM condition number.
- **Sensitivity warnings**: total-order indices are not additive; sensitivity does not prove identifiability.
- **Validation warnings**: coverage or transfer issues; not biological proof.
- **Unsupported-claim warnings**: disease diagnosis or mechanism proof requested from OCR-only data.
""")


    with sections[17]:
        st.subheader("Method references and learning links")
        st.markdown("""
These links are provided for interested readers who want the statistical and numerical background behind the analysis tabs.

| Topic | What it supports in MitoAgent | Reference / link |
|---|---|---|
| Fisher Information Matrix | Local identifiability/sloppiness diagnostic | Raue et al., *Bioinformatics* 2009, profile likelihood and practical identifiability: https://doi.org/10.1093/bioinformatics/btp358 |
| Profile likelihood | Practical identifiability and confidence-bound assessment | Raue et al., *Bioinformatics* 2009: https://doi.org/10.1093/bioinformatics/btp358 |
| Differential evolution | Global optimizer used before local polishing | Storn and Price, *Journal of Global Optimization* 1997: https://doi.org/10.1023/A:1008202821328 |
| Morris screening | Elementary-effects global sensitivity screening | Morris, *Technometrics* 1991: https://doi.org/10.1080/00401706.1991.10484804 |
| Sobol sensitivity indices | First-order and total-order variance-based sensitivity | Sobol, *Mathematical Modeling and Computational Experiment* 1993; Saltelli et al. global sensitivity analysis overview: https://doi.org/10.1016/S0010-4655(02)00280-1 |
| SALib | Python implementation used for sensitivity workflows | Herman and Usher, *Journal of Open Source Software* 2017: https://doi.org/10.21105/joss.00097 |
| Oroboros / high-resolution respirometry | Experimental context for OCR traces | Gnaiger, OXPHOS analysis resources: https://wiki.oroboros.at/index.php/MiPNet |
""")
