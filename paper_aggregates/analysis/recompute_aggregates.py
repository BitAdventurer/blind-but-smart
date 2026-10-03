#!/usr/bin/env python3
"""Recompute published aggregate arithmetic; this does not rerun experiments."""
import argparse
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

TABLES = Path(__file__).resolve().parent.parent / "tables"
COLUMNS = ("G_H", "G_TMS", "G_CB", "G_Independent", "G_Likelihood",
           "A_H", "A_TMS", "A_CB", "A_Independent")
CONTRASTS = (
    ("Grounding", "H--TMS", "G_H", "G_TMS"),
    ("Grounding", "H--CB", "G_H", "G_CB"),
    ("Action Step", "H--TMS", "A_H", "A_TMS"),
    ("Action Step", "H--CB", "A_H", "A_CB"),
    ("Grounding", "H--Independent", "G_H", "G_Independent"),
    ("Action Step", "H--Independent", "A_H", "A_Independent"),
    ("Grounding", "Full--Likelihood only", "G_H", "G_Likelihood"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(name, hashes):
    path = TABLES / name
    data = path.read_bytes()
    hashes[name] = hashlib.sha256(data).hexdigest()
    body = data.decode("utf-8-sig").split(r"\begin{tabular}", 1)[1]
    body = body.split(r"\end{tabular}", 1)[0]
    return [[part.strip() for part in line.rsplit(r"\\", 1)[0].split("&")]
            for line in body.splitlines() if "&" in line]


def check_round(label, value, printed, checks):
    printed = printed.strip()
    places = len(printed.split(".")[1]) if "." in printed else 0
    with localcontext() as context:
        context.prec = 50
        number = (Decimal(value.numerator) / Decimal(value.denominator)
                  if isinstance(value, F) else Decimal(str(value)))
        rounded = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    require(rounded == Decimal(printed), f"{label}: computed {number}, printed {printed}")
    checks.append(label)


def average(values):
    return sum(values, F(0)) / len(values)


def conditional_interval(differences):
    # Inclusive-tail inversion at alpha/2=.025. Empty subset contributes 1/1024.
    means = []
    for mask in range(1, 1 << len(differences)):
        subset = [d for i, d in enumerate(differences) if mask & (1 << i)]
        means.append(average(subset))
    means.sort()
    return means[24], means[-25]


def upper_tail(differences, margin=F(1)):
    centered = [d - margin for d in differences]
    observed = sum(centered)
    count = 0
    for mask in range(1 << len(centered)):
        signed = sum(value if mask & (1 << i) else -value
                     for i, value in enumerate(centered))
        count += signed >= observed  # Fraction preserves exact ties.
    return F(count, 1 << len(centered))


def holm(pvalues):
    adjusted = [F(0)] * len(pvalues)
    running = F(0)
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, min(F(1), (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def compute():
    hashes, checks = {}, []
    c6 = read_rows("C_6.tex", hashes)
    families = [row for row in c6 if row[0].isdigit()]
    require([int(row[0]) for row in families] == list(range(1, 11)), "C.6: expected families 1-10")
    require(all(len(row) == 10 for row in families), "C.6: column schema changed")
    values = {name: [F(row[i + 1]) for row in families] for i, name in enumerate(COLUMNS)}
    published_means = next(row[1:] for row in c6 if row[0] == "Mean")
    means = {name: average(column) for name, column in values.items()}
    for name, printed in zip(COLUMNS, published_means):
        check_round(f"C.6 mean {name}", means[name], printed, checks)

    c7 = {(row[0], row[1]): row[2:] for row in read_rows("C_7.tex", hashes)
          if row[0] in {"Grounding", "Action Step"}}
    require(len(c7) == 7, "C.7: expected seven contrasts")
    differences = [[a - b for a, b in zip(values[left], values[right])]
                   for _, _, left, right in CONTRASTS]
    pvalues = [upper_tail(d) for d in differences[:4]]
    adjusted = holm(pvalues)
    contrasts = []
    for index, ((task, name, _, _), d) in enumerate(zip(CONTRASTS, differences)):
        mean, interval = average(d), conditional_interval(d)
        printed = c7[(task, name)]
        label = f"C.7 {task} {name}"
        check_round(label + " mean", mean, printed[0], checks)
        endpoints = re.findall(r"-?\d+(?:\.\d+)?", printed[1])
        require(len(endpoints) == 2, label + ": expected two interval endpoints")
        for side, value, target in zip(("lower", "upper"), interval, endpoints):
            check_round(label + " " + side, value, target, checks)
        result = {"task": task, "contrast": name, "family_differences_pp": list(map(float, d)),
                  "mean_pp": float(mean), "conditional_interval_pp": list(map(float, interval)),
                  "exact_interval": list(map(str, interval)),
                  "rounding_envelope_pp": [float(interval[0] - F("0.1")), float(interval[1] + F("0.1"))]}
        if index < 4:
            check_round(label + " p_upper_1pp", pvalues[index], printed[2], checks)
            check_round(label + " p_Holm", adjusted[index], printed[3], checks)
            result.update(p_upper_1pp=float(pvalues[index]), p_Holm=float(adjusted[index]))
        else:
            require(printed[2:] == ["---", "---"], label + ": supplementary tests should be absent")
        contrasts.append(result)

    screenspot = {}
    for row in read_rows("D_6.tex", hashes):
        if len(row) != 5 or not re.fullmatch(r"[\d,]+", row[1]):
            continue
        name = row[0].split("$")[0]
        n = int(row[1].replace(",", ""))
        counts = [int(value.strip().replace(",", "")) for value in row[2].split("/")]
        require(len(counts) == 3 and all(0 <= c <= n for c in counts), name + ": invalid seed counts")
        accuracies = [F(100 * c, n) for c in counts]
        mean, sd = average(accuracies), statistics.stdev(accuracies)
        printed = re.findall(r"\d+(?:\.\d+)?", row[3])
        require(len(printed) == 2, name + ": expected mean and SD")
        check_round("D.6 " + name + " mean", mean, printed[0], checks)
        check_round("D.6 " + name + " SD", sd, printed[1], checks)
        screenspot[name] = {"eligible_n": n, "correct_counts": counts,
                            "seed_accuracy_pct": list(map(float, accuracies)),
                            "mean_pct": float(mean), "sample_sd_pp": sd}
    require(len(screenspot) == 7, "D.6: expected seven methods")
    h, tms = screenspot["H"], screenspot["TMS"]
    require(h["eligible_n"] == tms["eligible_n"], "D.6: H/TMS denominators differ")
    count_differences = [a - b for a, b in zip(h["correct_counts"], tms["correct_counts"])]
    seed_differences = [F(100 * value, h["eligible_n"]) for value in count_differences]
    seed_summary = {"correct_count_differences": count_differences,
                    "seed_differences_pp": list(map(float, seed_differences)),
                    "mean_pp": float(average(seed_differences)),
                    "sample_sd_pp": statistics.stdev(seed_differences),
                    "scope": "Descriptive differences aligned by listed seed columns; fitted components fixed; no interval or refit inference."}

    f5 = read_rows("F_5.tex", hashes)
    allocations = {row[0]: list(map(F, row[-2:])) for row in f5
                   if row[0] in {"H (joint)", "Independent-500k", "Independent-1M"}}
    require(len(allocations) == 3, "F.5: expected three methods")
    difference = [a - b for a, b in zip(allocations["H (joint)"], allocations["Independent-1M"])]
    printed = next(row[-2:] for row in f5 if "H $-$ Independent-1M" in row[0])
    for task, value, target in zip(("Grounding", "Action"), difference, printed):
        check_round("F.5 H-Independent-1M " + task, value, target, checks)
    for method, columns in (("H (joint)", ("G_H", "A_H")), ("Independent-500k", ("G_Independent", "A_Independent"))):
        require(allocations[method] == [means[key] for key in columns], "F.5/C.6 inconsistent baseline " + method)
    extra_training = [a - b for a, b in zip(allocations["Independent-1M"], allocations["Independent-500k"])]

    f6 = read_rows("F_6.tex", hashes)
    actor_updates = {row[0]: list(map(F, row[1:])) for row in f6
                     if row[0] in {"H (matched rerun)", "H-ActorTMS"}}
    require(len(actor_updates) == 2, "F.6: expected two methods")
    actor_difference = [a - b for a, b in zip(actor_updates["H (matched rerun)"], actor_updates["H-ActorTMS"])]
    # Appendix F.5 reports a 0.9 pp difference for both tasks in Table F.6.
    # Keep this companion independent of the manuscript's section files.
    for task, value in zip(("Grounding", "Action"), actor_difference):
        check_round("F.6 H-ActorTMS " + task, value, "0.9", checks)
    return {"status": "pass", "rounded_numeric_checks_passed": len(checks), "input_sha256": hashes,
            "scope": "Arithmetic from published aggregates only; not an experimental rerun or raw-record verification.",
            "exclusions": ["Raw predictions and fitting/evaluation execution", "G attack intervals", "Retention reconstruction",
                           "F.5 intervals: Independent-1M family-level inputs are absent",
                           "F.6 approximate intervals: paired family inputs and repeat counts are not supplied"],
            "C6_means_pct": {name: float(value) for name, value in means.items()}, "C7_contrasts": contrasts,
            "D6_methods": screenspot, "D6_H_minus_TMS": seed_summary,
            "F5": {"H_minus_Independent1M_pp": list(map(float, difference)),
                   "Independent1M_minus_500k_pp": list(map(float, extra_training)), "intervals_recomputed": False},
            "F6": {"status": "author_confirmed_experimental_summary", "mean_accuracy_pct": {name: list(map(float, values)) for name, values in actor_updates.items()},
                   "H_minus_ActorTMS_pp": list(map(float, actor_difference)), "intervals_recomputed": False,
                   "experimental_status_confirmed": True},
            "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args()
    try:
        result = json.dumps(compute(), indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(result, encoding="utf-8")
        else:
            print(result, end="")
    except (ValueError, OSError, KeyError, IndexError, StopIteration) as error:
        print(f"Aggregate check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
