import csv, json
from pathlib import Path


def test_identifiability_outputs_exist_after_chunk6():
    base = Path('results/identifiability')
    assert (base / 'fim_dataset_I.json').exists()
    assert (base / 'profiles_dataset_I.json').exists()
    assert (base / 'identifiability_summary.csv').exists()
    assert (base / 'parameter_interpretability_flags.csv').exists()


def test_fim_records_raw_and_clipped_condition_numbers():
    data = json.loads(Path('results/identifiability/fim_dataset_I.json').read_text())
    assert 'eigvals_raw' in data and 'eigvals_clipped' in data
    assert 'condition_raw' in data and 'condition_clipped' in data
    assert data['method'] == 'fisher_information'
    assert isinstance(data.get('warnings'), list)


def test_profile_flags_are_cautious():
    rows = list(csv.DictReader(open('results/identifiability/parameter_interpretability_flags.csv')))
    assert rows
    flags = {r['interpretability_flag'] for r in rows}
    assert flags <= {'interpretable', 'weak', 'one-sided', 'flat', 'unresolved', 'optimizer failure'}
    assert any(r['interpretability_flag'] in {'weak', 'one-sided', 'flat', 'unresolved'} for r in rows)
