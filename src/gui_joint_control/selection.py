"""Development selection from actual closed-loop trajectory records.

The callback is an evaluator, not a table of reported paper scores. This module
does not run a VLM or infer missing run bindings. It computes the manuscript's
eligible-slot return and keeps the selected and latest training states separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterable, Mapping


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class DevelopmentScore:
    """Return is equal-trajectory/equal-replicate; resources use eligible totals."""

    mean_return: float
    mean_regional_budget: float
    mean_candidate_count: float
    eligible_slots: int
    trajectories: int
    replicates: int
    evaluation_grid_sha256: str

    def ranking_key(self, iteration: int) -> tuple:
        return (-self.mean_return, self.mean_regional_budget, self.mean_candidate_count, iteration)


def score_development(episodes: Iterable[Mapping]) -> DevelopmentScore:
    """Score full 56-slot rollouts, including -1 for each exhausted eligible slot.

Each episode has trajectory_id, replicate and records. Records use the runtime
schema (slot, eligible, invoked, correct, executed_budgets, candidate_count).
The invocation reward is recomputed without entropy or the replay tail penalty;
otherwise exhausted slots would be counted twice. Every replicate must contain
the same nonempty trajectory population and eligibility masks. Zero-eligible
trajectories are retained in the grid audit but excluded from trajectory means.
"""
    by_replicate: dict[str, dict[str, tuple]] = {}
    budget_totals, candidate_totals = [], []
    eligible_total = 0
    grid = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("Development episodes must be objects")
        trajectory = episode.get("trajectory_id")
        replicate = episode.get("replicate")
        if not isinstance(trajectory, str) or not trajectory:
            raise ValueError("Every development episode requires a nonempty trajectory_id")
        if isinstance(replicate, bool) or not isinstance(replicate, (str, int)) or str(replicate) == "":
            raise ValueError("Every development episode requires an explicit replicate ID")
        replicate = str(replicate)
        population = by_replicate.setdefault(replicate, {})
        if trajectory in population:
            raise ValueError("Duplicate development trajectory/replicate")
        records = list(episode.get("records", []))
        if len(records) != 56:
            raise ValueError("Development episodes must contain all 56 fixed transcript slots")
        eligibility, rewards = [], []
        episode_budget = episode_candidates = 0.0
        for slot, record in enumerate(records):
            if not isinstance(record, Mapping) or type(record.get("slot")) is not int or record["slot"] != slot:
                raise ValueError("Development records must be ordered by all slots 0..55")
            if type(record.get("eligible")) is not bool or type(record.get("invoked")) is not bool:
                raise ValueError("eligible and invoked must be explicit booleans")
            eligible, invoked = record["eligible"], record["invoked"]
            eligibility.append(eligible)
            if invoked and not eligible:
                raise ValueError("An ineligible slot cannot invoke the mechanism")
            if not eligible:
                continue
            if not invoked:
                rewards.append(-1.0)
                continue
            if type(record.get("correct")) is not bool:
                raise ValueError("Invoked records require an offline boolean correctness label")
            budgets = record.get("executed_budgets")
            if not isinstance(budgets, (list, tuple)) or len(budgets) != 25:
                raise ValueError("Invoked records require 25 executed regional budgets")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                   not math.isfinite(value) or not 1.5 <= value <= 5 for value in budgets):
                raise ValueError("Executed regional budgets must be finite and in [1.5,5]")
            count = record.get("candidate_count")
            if type(count) is not int or not 1 <= count <= 20:
                raise ValueError("Invoked candidate_count must be an integer in 1..20")
            budget_sum = math.fsum(budgets)
            reward = float(record["correct"]) + .5 * (math.log(5) - math.log(budget_sum / 25)) / (math.log(5) - math.log(1.5)) - .1 * (count - 1) / 19
            rewards.append(reward)
            episode_budget += budget_sum
            episode_candidates += count
        discounted = math.fsum((.99 ** index) * reward for index, reward in enumerate(rewards))
        population[trajectory] = (tuple(eligibility), discounted, len(rewards))
        grid.append([replicate, trajectory, eligibility])
        budget_totals.append(episode_budget)
        candidate_totals.append(episode_candidates)
        eligible_total += len(rewards)
    if not by_replicate or not eligible_total:
        raise ValueError("Development selection requires nonempty eligible trajectories")
    first = next(iter(by_replicate.values()))
    for population in by_replicate.values():
        if population.keys() != first.keys() or any(population[key][0] != first[key][0] for key in first):
            raise ValueError("Development replicates must share trajectory IDs and eligibility masks")
    eligible_trajectories = sum(value[2] > 0 for value in first.values())
    means = [math.fsum(value[1] for value in population.values() if value[2]) / eligible_trajectories
             for population in by_replicate.values()]
    return DevelopmentScore(
        mean_return=math.fsum(means) / len(means),
        mean_regional_budget=math.fsum(budget_totals) / (25 * eligible_total),
        mean_candidate_count=math.fsum(candidate_totals) / eligible_total,
        eligible_slots=eligible_total, trajectories=eligible_trajectories,
        replicates=len(by_replicate),
        evaluation_grid_sha256=sha256(_canonical(sorted(grid)).encode("utf-8")).hexdigest(),
    )


class CheckpointManager:
    """Select only on development rollouts; preserve a separate resumable last state.

``evaluate(trainer)`` must return split='development', manifest_sha256, episodes.
The callback must run each candidate's own feedback/ledger and must not update
the trainer. An Independent trainer is evaluated as its composed pair; no
component is selected using the other component's reward or a Bench outcome.
``binding`` must include dev_manifest_sha256 and may include all other run IDs.
On resume, construct with resume=True in the same directory, then load last.pt
into the trainer and call validate_resume before continuing.
"""

    def __init__(self, directory: str | Path, *, binding: Mapping, interval: int = 10_000, resume: bool = False):
        if type(interval) is not int or interval <= 0:
            raise ValueError("Selection interval must be a positive integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binding = json.loads(_canonical(dict(binding)))
        manifest = self.binding.get("dev_manifest_sha256")
        if not isinstance(manifest, str) or len(manifest) != 64 or any(c not in "0123456789abcdef" for c in manifest):
            raise ValueError("binding.dev_manifest_sha256 must be an explicit lowercase SHA-256")
        self.interval = interval
        self.state_path = self.directory / "selection.json"
        self.selected_path = self.directory / "selected.pt"
        self.last_path = self.directory / "last.pt"
        if resume:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("format_version") != 1 or self.state.get("binding") != self.binding or self.state.get("interval") != interval:
                raise ValueError("Selection resume bindings or interval differ")
            for field, path in (("best", self.selected_path), ("last", self.last_path)):
                entry = self.state.get(field)
                if entry is not None and (not path.is_file() or file_sha256(path) != entry["checkpoint_sha256"]):
                    raise ValueError(f"{field} checkpoint does not match its persisted selection state")
        else:
            if any(path.exists() for path in (self.state_path, self.selected_path, self.last_path)):
                raise FileExistsError("Selection output already exists; use an explicit resume")
            self.state = {"format_version": 1, "binding": self.binding, "interval": interval,
                          "criterion": "maximum equal-trajectory/replicate eligible return, lower eligible regional budget, lower eligible count, earlier iteration",
                          "return_discount": .99, "best": None, "last": None, "evaluations": []}

    @property
    def best(self):
        return self.state["best"]

    def validate_resume(self, trainer) -> None:
        last = self.state["last"]
        if last is None or trainer.iteration != last["iteration"] or trainer.method != last["method"]:
            raise ValueError("Resume the exact last checkpoint before continuing selection")
        for field in ("seed", "config_sha256"):
            if field in last and getattr(trainer, field, None) != last[field]:
                raise ValueError(f"Resumed trainer {field} differs from selection state")
        completed = self.state["evaluations"][-1]["iteration"] if self.state["evaluations"] else 0
        if completed != (trainer.iteration // self.interval) * self.interval:
            raise ValueError("Interrupted or missing development selection; do not silently resume past it")

    def _persist(self) -> None:
        _atomic_json(self.state_path, self.state)

    def save_last(self, trainer) -> None:
        if type(trainer.iteration) is not int or trainer.iteration < 0:
            raise ValueError("Trainer iteration must be a nonnegative integer")
        previous = self.state["last"]
        if previous and (trainer.method != previous["method"] or trainer.iteration < previous["iteration"]):
            raise ValueError("Cannot roll back or change a selection run's last checkpoint")
        trainer.save_checkpoint(self.last_path)
        self.state["last"] = {"iteration": trainer.iteration, "method": trainer.method,
                              "checkpoint_sha256": file_sha256(self.last_path)}
        for field in ("seed", "config_sha256"):
            if hasattr(trainer, field):
                self.state["last"][field] = getattr(trainer, field)
        self._persist()

    def consider(self, trainer, evaluate: Callable) -> dict | None:
        iteration = trainer.iteration
        if type(iteration) is not int or iteration <= 0:
            raise ValueError("Only completed positive training iterations can be selected")
        if iteration % self.interval:
            return None
        evaluations = self.state["evaluations"]
        if evaluations and iteration <= evaluations[-1]["iteration"]:
            raise ValueError("Checkpoint selection iterations must increase; do not reevaluate on resume")
        expected = self.interval if not evaluations else evaluations[-1]["iteration"] + self.interval
        if iteration != expected:
            raise ValueError("A scheduled development selection checkpoint was skipped")
        result = evaluate(trainer)
        if trainer.iteration != iteration:
            raise ValueError("Development evaluation must not advance training")
        if not isinstance(result, Mapping) or result.get("split") != "development":
            raise ValueError("Checkpoint selection requires an actual development evaluation")
        if result.get("manifest_sha256") != self.binding["dev_manifest_sha256"]:
            raise ValueError("Development evaluator used a different manifest")
        score = score_development(result.get("episodes", []))
        if evaluations and score.evaluation_grid_sha256 != evaluations[0]["score"]["evaluation_grid_sha256"]:
            raise ValueError("Development trajectory/replicate grid changed between checkpoints")
        entry = {"iteration": iteration, "method": trainer.method, "score": asdict(score)}
        if self.best and trainer.method != self.best["method"]:
            raise ValueError("A selection run cannot switch methods")
        better = self.best is None or score.ranking_key(iteration) < DevelopmentScore(**self.best["score"]).ranking_key(self.best["iteration"])
        self.save_last(trainer)
        if better:
            _atomic_copy(self.last_path, self.selected_path)
            self.state["best"] = {**entry, "checkpoint_sha256": file_sha256(self.selected_path)}
        self.state["evaluations"].append({**entry, "selected": better})
        self._persist()
        return {**entry, "selected": better}


def select_retrieval_threshold(evaluate: Callable[[float], Mapping], *, manifest_sha256: str) -> dict:
    """Evaluate gates 0,.25,...,2 on development; equal returns favor larger gates.

The callback has the same result schema as CheckpointManager's evaluator and
must produce fresh actual closed-loop records at the supplied threshold. This
helper never substitutes a default threshold when development is unavailable.
"""
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in manifest_sha256):
        raise ValueError("An explicit development manifest SHA-256 is required")
    entries, grid = [], None
    for quarter in range(9):
        threshold = quarter / 4
        result = evaluate(threshold)
        if not isinstance(result, Mapping) or result.get("split") != "development" or result.get("manifest_sha256") != manifest_sha256:
            raise ValueError("Retrieval selection must use the bound development manifest")
        score = score_development(result.get("episodes", []))
        if grid is not None and score.evaluation_grid_sha256 != grid:
            raise ValueError("Retrieval gate candidates used different development grids")
        grid = score.evaluation_grid_sha256
        entries.append({"threshold": threshold, "score": asdict(score)})
    best = max(entries, key=lambda entry: (entry["score"]["mean_return"], entry["threshold"]))
    return {"format_version": 1, "split": "development", "manifest_sha256": manifest_sha256,
            "selected_threshold": best["threshold"], "criterion": "maximum eligible return, then larger threshold",
            "evaluations": entries, "reproduces_historical_results": False}
