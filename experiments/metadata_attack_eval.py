#!/usr/bin/env python3
"""Evaluate recorded metadata; never generate or recover paper measurements.

CSV columns: allocator, family, pair_id, side (R/Rp), rep, k,
filter_status (EXEC_PROP/EXEC_FALLBACK or 0/1), eps_01..eps_25,
target_cell (1..25). Pair IDs or target cells may be blank for the other attack.
Pair accuracy uses one held-out record per side within each pair; its chance
reference is 0.5, which does not establish absence of leakage. Cell accuracy
uses the first maximal-budget cell and five-fold out-of-fold logistic predictions.
Both intervals resample fixed evaluated outcomes 2,000 times, without refitting.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RNG_SEED = 20260912
N_BOOT = 2000
EPS_COLS = [f"eps_{i:02d}" for i in range(1, 26)]
FEATS = EPS_COLS + ["k", "filter_status"]


def validate(df, attack):
    """Reject malformed measurements; account for explicitly absent attack IDs."""
    required = {"allocator", "family", *FEATS}
    required.update({"pair_id", "side", "rep"} if attack == "pair" else {"target_cell"})
    if missing := required - set(df.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("input contains no records")
    df = df.copy().reset_index(drop=True)
    for name in ("allocator", "family"):
        if df[name].isna().any() or df[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name} must be nonempty")
        df[name] = df[name].astype(str)
    df["filter_status"] = df["filter_status"].map(lambda value: {"EXEC_PROP": 0, "EXEC_FALLBACK": 1}.get(str(value).upper(), value))
    for name in FEATS:
        df[name] = pd.to_numeric(df[name], errors="raise")
        if not np.isfinite(df[name].to_numpy(float)).all():
            raise ValueError(f"{name} must contain only finite numbers")
    if not df.filter_status.isin([0, 1]).all():
        raise ValueError("filter_status must be EXEC_PROP, EXEC_FALLBACK, 0, or 1")
    if (df[EPS_COLS] < 0).any().any():
        raise ValueError("executed budgets must be nonnegative")
    if ((df.k < 1) | (df.k != np.floor(df.k))).any():
        raise ValueError("k must be a positive integer")
    if attack == "pair":
        df["pair_id"] = df.pair_id.replace(r"^\s*$", np.nan, regex=True)
        selected = df.pair_id.notna()
        if not df.loc[selected, "side"].isin(["R", "Rp"]).all():
            raise ValueError("pair side must be R or Rp")
        reps = pd.to_numeric(df.loc[selected, "rep"], errors="raise")
        if not np.isfinite(reps.to_numpy(float)).all() or ((reps < 0) | (reps != np.floor(reps))).any():
            raise ValueError("pair rep must be a finite nonnegative integer")
        df.loc[selected, "rep"] = reps
        if df.loc[selected].duplicated(["allocator", "family", "pair_id", "side", "rep"]).any():
            raise ValueError("duplicate repetition within allocator/family/pair/side")
    else:
        df["target_cell"] = pd.to_numeric(df.target_cell, errors="raise")
        cells = df.target_cell.dropna().to_numpy(float)
        if not np.isfinite(cells).all() or ((cells < 1) | (cells > 25) | (cells != np.floor(cells))).any():
            raise ValueError("target_cell must be an integer in 1..25 or blank")
    return df


def bootstrap_interval(values, rng):
    """Percentiles of fixed-outcome means, with bounded resampling storage."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a nonempty finite vector")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("bootstrap accuracies must be in [0, 1]")
    means = np.empty(N_BOOT)
    # Bound the index/sample arrays to about one million entries per batch.
    batch = max(1, min(64, 1_000_000 // len(values)))
    for start in range(0, N_BOOT, batch):
        stop = min(start + batch, N_BOOT)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5], method="linear").tolist()


def pair_accuracy(df, rng):
    """Balanced accuracy on two held-out records, with fitting within this pair."""
    r, rp = df[df.side == "R"], df[df.side == "Rp"]
    held_r, held_rp = int(rng.integers(len(r))), int(rng.integers(len(rp)))
    train = pd.concat([r.drop(r.index[held_r]), rp.drop(rp.index[held_rp])])
    test = pd.concat([r.iloc[[held_r]], rp.iloc[[held_rp]]])
    x_train, x_test = train[FEATS].to_numpy(float), test[FEATS].to_numpy(float)
    y_train = (train.side == "Rp").to_numpy(int)
    if np.all(x_train == x_train[0]):
        accuracy = 0.5  # Any constant prediction scores 1/2 on this balanced test.
    else:
        scale = StandardScaler().fit(x_train)
        model = LogisticRegression(C=1.0, max_iter=1000).fit(scale.transform(x_train), y_train)
        accuracy = float((model.predict(scale.transform(x_test)) == [0, 1]).mean())
    return accuracy, {"R": int(r.iloc[held_r].rep), "Rp": int(rp.iloc[held_rp].rep)}


def attack_pair(df, rng):
    results = []
    for (allocator, family), group in df.groupby(["allocator", "family"], sort=False):
        details, accuracies = [], []
        missing = int(group.pair_id.isna().sum())
        for pair_id, pair in group[group.pair_id.notna()].groupby("pair_id", sort=True):
            reps = {side: sorted(int(x) for x in pair.loc[pair.side == side, "rep"]) for side in ("R", "Rp")}
            entry = {"pair_id": str(pair_id), "records_per_side": {side: len(ids) for side, ids in reps.items()}, "rep_ids_per_side": reps}
            if min(map(len, reps.values())) < 2:
                entry.update(status="excluded", reason="fewer_than_two_records_on_a_side")
            else:
                accuracy, held = pair_accuracy(pair, rng)
                accuracies.append(accuracy)
                entry.update(status="analyzed", accuracy=accuracy, held_out_rep=held)
            details.append(entry)
        n = len(accuracies)
        results.append({
            "allocator": allocator, "family": family, "count_unit": "screen_pair",
            "n_input": len(details), "n_analyzed": n, "n_excluded": len(details) - n,
            "n_input_records": len(group), "n_records_without_pair_id": missing,
            "exclusion_reasons": {"fewer_than_two_records_on_a_side": len(details) - n},
            "record_exclusion_reasons": {"missing_pair_id": missing},
            "accuracy": float(np.mean(accuracies)) if n else None,
            "ci95": bootstrap_interval(accuracies, rng) if n else None,
            "accuracy_unit": "fraction", "pairs": details,
        })
    if not any(result["n_analyzed"] for result in results):
        raise ValueError("no analyzable pairs; each pair needs at least two records on each side")
    return results


def attack_cell(df, rng, seed):
    results = []
    for allocator, group in df.groupby("allocator", sort=False):
        data = group[group.target_cell.notna()]
        counts = [{"family": family, "n_input": len(part), "n_analyzed": int(part.target_cell.notna().sum()),
                   "n_excluded": int(part.target_cell.isna().sum())}
                  for family, part in group.groupby("family", sort=False)]
        result = {"allocator": allocator, "count_unit": "record", "n_input": len(group),
                  "n_analyzed": len(data), "n_excluded": len(group) - len(data),
                  "exclusion_reasons": {"missing_target_cell": len(group) - len(data)}, "counts_by_family": counts,
                  "accuracy_unit": "percent", "attacks": {}}
        if len(data):
            x = data[EPS_COLS].to_numpy(float)
            y = data.target_cell.to_numpy(int)
            classes, sizes = np.unique(y, return_counts=True)
            if len(classes) < 2 or sizes.min() < 5:
                raise ValueError(f"{allocator}: five-fold stratified logistic evaluation requires at least two target classes and five records per observed class")
            predictions = np.empty_like(y)
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for train, test in folds.split(x, y):
                scale = StandardScaler().fit(x[train])
                model = LogisticRegression(C=1.0, max_iter=2000).fit(scale.transform(x[train]), y[train])
                predictions[test] = model.predict(scale.transform(x[test]))
            for name, predicted in (("argmax", x.argmax(axis=1) + 1), ("logreg", predictions)):
                hits = (predicted == y).astype(float)
                result["attacks"][name] = {"accuracy": float(100 * hits.mean()), "ci95": [100 * v for v in bootstrap_interval(hits, rng)],
                                           "n_correct": int(hits.sum()), "n_analyzed": len(hits)}
            result["target_cell_counts"] = {str(label): int(n) for label, n in zip(classes, sizes)}
        results.append(result)
    if not any(result["n_analyzed"] for result in results):
        raise ValueError("no records with target_cell")
    return results


def evaluate(df, attack, seed=RNG_SEED):
    df = validate(df, attack)
    rng = np.random.default_rng(seed)
    results = attack_pair(df, rng) if attack == "pair" else attack_cell(df, rng, seed)
    return {
        "attack": attack, "seed": seed,
        "method": {
            "bootstrap_replicates": N_BOOT, "percentiles": [2.5, 97.5], "percentile_method": "linear",
            "resampling_unit": "screen_pair" if attack == "pair" else "record",
            "resampled_values": "fixed_pair_accuracies" if attack == "pair" else "fixed_correctness_indicators",
            "refit_during_bootstrap": False, "rng": "numpy.default_rng/PCG64",
            "pair_split": "one_random_record_per_side_held_out_within_each_pair" if attack == "pair" else None,
            "cell_split": None if attack == "pair" else {"folds": 5, "stratified": True, "shuffle": True, "random_state": seed},
            "logistic_regression": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000 if attack == "pair" else 2000,
                                    "standardization": "fit_on_training_partition_only"},
            "argmax_tie_rule": None if attack == "pair" else "first_cell_in_1_based_order",
            "interpretation": "Conditional on evaluated records/pairs and fitted attackers; chance-level accuracy does not establish absence of leakage.",
        },
        "results": results,
    }


def make_demo(rng):
    """Small synthetic fixture; its outcomes are not paper results."""
    rows = []
    for allocator in ("CLEAN", "H", "TMS"):
        for pair in range(4):
            for side in ("R", "Rp"):
                for rep in range(3):
                    eps = np.full(25, 2.62)
                    if allocator == "CLEAN" and side == "Rp":
                        eps[0] = 4
                    elif allocator == "H":
                        eps += rng.uniform(-0.1, 0.1, 25)
                    rows.append(dict(allocator=allocator, family="synthetic_pair", pair_id=f"demo_{pair}", side=side,
                                     rep=rep, k=5, filter_status=0, target_cell=np.nan, **dict(zip(EPS_COLS, eps))))
        for cell in range(1, 26):
            for rep in range(5):
                eps = np.full(25, 2.62)
                if allocator == "CLEAN":
                    eps[cell - 1] = 4
                elif allocator == "H":
                    eps += rng.uniform(-0.1, 0.1, 25)
                rows.append(dict(allocator=allocator, family="synthetic_grounding", pair_id=np.nan, side=np.nan,
                                 rep=rep, k=5, filter_status=0, target_cell=cell, **dict(zip(EPS_COLS, eps))))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path)
    parser.add_argument("--attack", choices=["pair", "cell"], required=True)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--output", type=Path, help="Save machine-readable results and input/method provenance")
    parser.add_argument("--demo", action="store_true", help="Use synthetic fixtures, never paper measurements")
    args = parser.parse_args(argv)
    if bool(args.csv) == args.demo:
        parser.error("give exactly one CSV path or --demo")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if args.csv and args.output and args.csv.resolve() == args.output.resolve():
        parser.error("output must not overwrite the input CSV")
    try:
        if args.demo:
            df = make_demo(np.random.default_rng(args.seed))
            provenance = {"kind": "synthetic_demo", "sha256": None, "paper_measurements": False}
        else:
            raw = args.csv.read_bytes()
            df = pd.read_csv(io.BytesIO(raw), dtype={name: str for name in ("allocator", "family", "pair_id", "side")})
            provenance = {"kind": "provided_csv", "filename": args.csv.name, "sha256": hashlib.sha256(raw).hexdigest(),
                          "verification": "computed_from_supplied_records; provenance_of_records_not_certified"}
        report = evaluate(df, args.attack, args.seed)
        report["input"] = provenance
        report["software"] = {"numpy": np.__version__, "pandas": pd.__version__}
        from importlib.metadata import version
        report["software"]["scikit_learn"] = version("scikit-learn")
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
