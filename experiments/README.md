# Recorded metadata evaluation

`metadata_attack_eval.py` evaluates supplied records for Appendix G. It preserves
the pair-level and record-level percentile bootstrap (2,000 replicates), with
fixed evaluated outcomes and no attacker refitting inside bootstrap replicates.
It does not recover missing measurements or tune intervals to published values.

Install the optional dependencies with `python -m pip install -e ".[metadata,test]"`.
For the standalone LaTeX bundle, which has no package manifest, use
`python -m pip install "numpy>=2,<3" "pandas>=2,<3" "scikit-learn>=1.5,<2" "pytest>=8"`.
From the repository root:

```powershell
python experiments/metadata_attack_eval.py records.csv --attack pair --output pair-results.json
python experiments/metadata_attack_eval.py records.csv --attack cell --output cell-results.json
python experiments/metadata_attack_eval.py --demo --attack pair --output synthetic-demo.json
python -m pytest tests/test_metadata_attack_eval.py -q
```

The CSV columns are documented at the top of the evaluator. Pair evaluation
requires at least two distinct repetition IDs on each side of every analyzed
pair. Missing pair IDs and insufficient repetitions are counted explicitly.
Cell evaluation uses records with a target cell; each observed class needs five
records for the five-fold stratified evaluation. Standardization is fitted only
on the training partition. Tied maximal budgets select the lowest cell index.

JSON includes the input SHA256, seed, method, software versions, analyzed and
excluded counts, exclusion reasons, and per-pair repetition counts/IDs. Pair
counts refer to supplied pairs, not automatically to another experiment's pair
inventory. Missing pair IDs are counted as records because their pair count is
unknown. Cell counts are also broken down by family while the classifier is
fitted across the allocator's eligible records, matching the original design.
Keep raw records and identifying pair IDs out of public artifacts. `--demo`
outputs are explicitly synthetic and must never be cited as paper results.
Chance-level accuracy is an empirical reference, not proof of no leakage.
