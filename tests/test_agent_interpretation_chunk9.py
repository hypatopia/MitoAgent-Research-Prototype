from agent.ask_agent import ask
from agent.reporting import enrich_report, build_analysis_status


def test_ask_mitoagent_refuses_disease_claim():
    out = ask('Can this diagnose Alzheimer disease?', {'data': {'n_samples': 1}, 'warnings_by_category': {}})
    assert out['answer_mode'] == 'deterministic_offline'
    assert 'disease_diagnosis' in out['unsupported_claims_refused']
    assert 'cannot diagnose' in out['answer'].lower()


def test_enriched_report_has_hypothesis_and_design_guidance():
    report = enrich_report({'data': {'n_samples': 10, 'n_fccp': 1}, 'warnings_by_category': {}, 'calibration': {'rmse_calib': 1.0}})
    assert report['hypothesis_prioritization']['status'] == 'generated'
    assert report['experimental_design_guidance']['status'] == 'generated'
    assert build_analysis_status(report)['Ask MitoAgent'] == 'deterministic/offline'
