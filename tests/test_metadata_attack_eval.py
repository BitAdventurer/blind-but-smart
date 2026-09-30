"""Small synthetic evaluator checks; no paper records or benchmark runs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "metadata_attack_eval.py"
spec = importlib.util.spec_from_file_location("metadata_attack_eval", SCRIPT)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def record(**changes):
    row = dict(allocator="H", family="secret_text", pair_id="pair-1", side="R", rep=0,
               k=5, filter_status="EXEC_PROP", target_cell=np.nan,
               **{name: 2.62 for name in evaluator.EPS_COLS})
    row.update(changes)
    return row


def pair_records():
    rows = [record(side=side, rep=rep) for side in ("R", "Rp") for rep in range(3)]
    rows += [record(pair_id="pair-2", side=side, rep=rep, eps_01=4.0 if side == "Rp" else 2.0)
             for side in ("R", "Rp") for rep in range(3)]
    return pd.DataFrame(rows)


def cell_records():
    return pd.DataFrame([record(family="grounding", pair_id=np.nan, side=np.nan, rep=rep,
                               target_cell=cell, **{f"eps_{cell:02d}": 4.0})
                         for cell in (1, 2, 3) for rep in range(10)])


def test_pair_determinism_counts_and_exclusions():
    df = pd.concat([pair_records(), pd.DataFrame([
        record(pair_id="insufficient", side="R"),
        record(pair_id="insufficient", side="Rp"),
        record(pair_id=np.nan, side=np.nan),
    ])], ignore_index=True)
    result = evaluator.evaluate(df, "pair")
    assert result == evaluator.evaluate(df, "pair")
    group = result["results"][0]
    assert (group["n_input"], group["n_analyzed"], group["n_excluded"]) == (3, 2, 1)
    assert group["n_input_records"] == 15
    assert group["record_exclusion_reasons"] == {"missing_pair_id": 1}
    assert group["exclusion_reasons"] == {"fewer_than_two_records_on_a_side": 1}
    pairs = {pair["pair_id"]: pair for pair in group["pairs"]}
    assert pairs["pair-1"]["records_per_side"] == {"R": 3, "Rp": 3}
    assert pairs["pair-1"]["accuracy"] == .5
    assert pairs["pair-2"]["accuracy"] == 1
    assert group["accuracy"] == .75
    assert group["ci95"] == [.5, 1]
    assert result["method"]["resampling_unit"] == "screen_pair"
    assert result["method"]["refit_during_bootstrap"] is False


def test_cell_argmax_logreg_and_count_units():
    df = pd.concat([cell_records(), pd.DataFrame([record()])], ignore_index=True)
    result = evaluator.evaluate(df, "cell")
    group = result["results"][0]
    assert (group["n_input"], group["n_analyzed"], group["n_excluded"]) == (31, 30, 1)
    assert group["exclusion_reasons"] == {"missing_target_cell": 1}
    assert group["counts_by_family"] == [
        {"family": "grounding", "n_input": 30, "n_analyzed": 30, "n_excluded": 0},
        {"family": "secret_text", "n_input": 1, "n_analyzed": 0, "n_excluded": 1},
    ]
    for attack in ("argmax", "logreg"):
        assert group["attacks"][attack] == {"accuracy": 100, "ci95": [100, 100], "n_correct": 30, "n_analyzed": 30}
    assert result["method"]["resampling_unit"] == "record"
    assert result["method"]["resampled_values"] == "fixed_correctness_indicators"
    assert result == evaluator.evaluate(df, "cell")


def test_batched_bootstrap_matches_original_draw_order():
    outcomes = np.array([0., .5, 1., 1., .5])
    rng = np.random.default_rng(8)
    original = outcomes[rng.integers(len(outcomes), size=(2000, len(outcomes)))].mean(axis=1)
    expected = np.percentile(original, [2.5, 97.5])
    assert evaluator.bootstrap_interval(outcomes, np.random.default_rng(8)) == expected.tolist()


def test_fixed_binary_bootstrap_interval_magnitude():
    hits = np.zeros(18178)
    hits[:1345] = 1
    low, high = evaluator.bootstrap_interval(hits, np.random.default_rng(8))
    assert .069 < low < .072
    assert .076 < high < .079
    assert high - low < .009


@pytest.mark.parametrize("column,value", [
    ("filter_status", "unknown"), ("filter_status", 2), ("eps_01", np.inf),
    ("eps_01", np.nan), ("eps_01", -1), ("k", 1.5), ("side", "other"),
    ("rep", -1), ("rep", np.inf), ("allocator", ""),
])
def test_rejects_invalid_pair_inputs(column, value):
    df = pair_records()
    df[column] = df[column].astype(object)
    df.loc[0, column] = value
    with pytest.raises(ValueError):
        evaluator.evaluate(df, "pair")


def test_duplicate_repetitions_and_missing_columns():
    df = pair_records()
    with pytest.raises(ValueError, match="duplicate repetition"):
        evaluator.evaluate(pd.concat([df, df.iloc[[0]]], ignore_index=True), "pair")
    with pytest.raises(ValueError, match="missing columns"):
        evaluator.evaluate(df.drop(columns="eps_01"), "pair")


@pytest.mark.parametrize("target", [0, 26, 1.5, np.inf])
def test_rejects_invalid_target_cells(target):
    df = cell_records()
    df["target_cell"] = df.target_cell.astype(float)
    df.loc[0, "target_cell"] = target
    with pytest.raises(ValueError, match="target_cell"):
        evaluator.evaluate(df, "cell")


def test_insufficient_cell_classes_for_five_fold():
    with pytest.raises(ValueError, match="five-fold"):
        evaluator.evaluate(cell_records().iloc[:3], "cell")


def test_cli_json_provenance(tmp_path):
    source, output = tmp_path / "records.csv", tmp_path / "results.json"
    pair_records().to_csv(source, index=False)
    completed = subprocess.run([sys.executable, str(SCRIPT), str(source), "--attack", "pair", "--output", str(output)],
                               capture_output=True, text=True, check=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(completed.stdout) == report
    assert report["input"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["input"]["kind"] == "provided_csv"
    assert report["seed"] == evaluator.RNG_SEED
    assert report["method"]["bootstrap_replicates"] == 2000
    assert report["method"]["percentiles"] == [2.5, 97.5]


def test_demo_is_clearly_synthetic(capsys):
    evaluator.main(["--demo", "--attack", "pair"])
    report = json.loads(capsys.readouterr().out)
    assert report["input"] == {"kind": "synthetic_demo", "sha256": None, "paper_measurements": False}


def test_cli_preserves_pair_ids_and_protects_input(tmp_path, capsys):
    source = tmp_path / "records.csv"
    df = pair_records()
    df["pair_id"] = df.pair_id.replace({"pair-1": "01", "pair-2": "1"})
    df.to_csv(source, index=False)
    evaluator.main([str(source), "--attack", "pair"])
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["n_analyzed"] == 2
    assert {pair["pair_id"] for pair in report["results"][0]["pairs"]} == {"01", "1"}
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        evaluator.main([str(source), "--attack", "pair", "--output", str(source)])
    assert source.read_bytes() == before
