# Recompute published aggregate statistics

For supplied Appendix G allocation records, use the separate [metadata evaluator](../../experiments/README.md). It reports actual analyzed/excluded counts and bootstrap provenance; it does not reconstruct the published attack results from rounded tables.

This companion uses Python 3's standard library only. It reads the adjacent manuscript tables directly, independently of the working directory. From the repository root:

```text
python paper_aggregates/analysis/recompute_aggregates.py
python paper_aggregates/analysis/recompute_aggregates.py --output paper_aggregates/analysis/computed_aggregates.json
```

The default writes JSON to standard output. `--output` writes to the supplied path; its parent directory must already exist. A mismatch or missing required input exits with a nonzero status. The saved `computed_aggregates.json` records the input-table SHA-256 hashes so it can be checked against subsequent manuscript changes.

## What is recalculated

- **C.6:** Nine accuracy means over the ten published family rows; evaluation repeats remain nested in their family means.
- **C.7:** Seven paired mean differences and individual conditional intervals. Decimal family inputs are represented by exact fractions. All 1,023 nonempty subset means are enumerated; the 25th smallest and 25th largest give the inclusive-tail sign-flip inversion endpoints at 0.025 per tail. The script enumerates all 1,024 signs for the original four tests at +1 pp, including ties, and applies Holm adjustment to those four tests only.
- **Rounding sensitivity:** Each paired difference can change by at most 0.1 pp when the original family accuracies were rounded to the nearest 0.1 pp. Expanding each interval endpoint by 0.1 pp gives the deterministic envelope described in Appendix C.4. This is not a new confidence level or trajectory-resampling interval.
- **D.6:** Accuracy means and sample SDs (`n - 1` denominator) from each method's three integer seed counts and eligible population. H–TMS differences are aligned by the listed seed columns and summarized descriptively. These are evaluation seeds with fitted components fixed, not additional model fits.
- **F.5:** The two H–Independent-1M differences, the Independent-1M versus Independent-500k differences, and consistency of the original baselines with C.6.
- **F.6:** The two H–H-ActorTMS mean differences and their agreement with the published +0.9 pp contrast in each task. The author confirmed the accuracies and approximate intervals as actual experimental results on 2026-10-02. This script checks their mean-difference arithmetic only.

Every displayed C.6 mean, C.7 mean/interval/test value, D.6 mean/SD, F.5 H–Independent-1M point difference, and F.6 H–H-ActorTMS point difference is checked at its published decimal precision. Fraction arithmetic preserves sign-flip ties; displayed decimal checks use nearest rounding with half-up ties. No checked value currently depends on a half-way rounding convention.

## Limits

This companion reproduces aggregate arithmetic, **not experiments**. It does not recover raw predictions, verify fitting independence or same-example pairing, reconstruct the evaluation protocol, or establish the sign-flip assumptions. Conditional coverage still requires the symmetry and location-shift assumptions stated in Appendix C.4 and does not account for newly sampled trajectories.

The script does not reconstruct retention denominators, attack intervals in Appendix G, the F.5 confidence intervals, or the approximate F.6 intervals. Independent-1M family inputs are absent; F.6 paired family inputs and repeat counts have not been supplied. Author confirmation of actual results is distinct from independent interval recomputation. The script does not manufacture missing data, infer unreported repeat counts, train models, call an API, or run benchmarks.
