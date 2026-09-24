"""Recorded-trajectory execution and local replay collection.

Clean features and probes remain client-side. Executor callbacks receive only
the refinement release and public text. Target labels are scored afterwards.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import math
import numpy as np
from .privacy import PrivacyLedger
from .action_evaluation import ActionPrediction, ActionScores, ActionSlot


@dataclass(frozen=True)
class Slot:
    instruction: str
    target_box: tuple[float, float, float, float] | None
    eligible: bool = True
    recorded: bool = True


@dataclass(frozen=True)
class Prediction:
    point: tuple[float, float] | None
    feedback: float


class TrajectoryExecutionError(ValueError):
    """A failed invocation with its already-spent disclosure preserved."""

    def __init__(self, message, records, *, used_budget):
        super().__init__(message)
        self.records = records
        self.used_budget = used_budget


def correct_grounding(point, box) -> bool:
    if point is None or box is None:
        return False
    x, y = point
    return bool(math.isfinite(x) and math.isfinite(y) and box[0] <= x <= box[2] and box[1] <= y <= box[3])


def run_trajectory(slots: list[Slot], feature_loader: Callable,
                   allocator: Callable, executor: Callable, *, ledger=None, executor_context=False):
    """Return local evaluation records and trusted replay for one trajectory.

Evaluation records include offline correctness labels. They must not be
confused with the public mechanism transcript represented by StepRelease.

Callbacks: feature_loader(zero_based_slot)->clean25x256; allocator(obs28)->
(budgets25,k); executor(release25x256, instruction,k)->Prediction. Executor never
receives target boxes, probes, the original image or controller observation.
"""
    if not 1 <= len(slots) <= 56:
        raise ValueError("One to 56 recorded slots are required")
    for slot in slots:
        if slot.eligible and (not slot.recorded or not slot.instruction or slot.target_box is None):
            raise ValueError("Eligible slots require public text, a recorded screen and an offline target")
        if slot.target_box is not None:
            box = slot.target_box
            if len(box) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in box) or box[0] > box[2] or box[1] > box[3]:
                raise ValueError("Target box must be ordered normalized coordinates")
    def execute(release, slot, k, previous_feedback):
        return (executor(release, slot.instruction, (), k, previous_feedback) if executor_context
                else executor(release, slot.instruction, k))

    def score(prediction, slot):
        if not isinstance(prediction, Prediction):
            raise ValueError("Grounding executor must return Prediction")
        return correct_grounding(prediction.point, slot.target_box), {"point": prediction.point}

    return _run_trajectory(slots, feature_loader, allocator, execute, score, ledger=ledger)


def run_action_trajectory(slots: list[ActionSlot], feature_loader: Callable,
                          allocator: Callable, executor: Callable, *, evaluator: Callable, ledger=None):
    """Recorded-screen Action evaluation with explicit offline scoring.

    ``executor(release, request, recorded_history, k, prior_feedback)`` receives
    only released/public inputs. ``evaluator(action, ActionSlot)`` is called
    afterwards, returning ActionScores. It is the caller's pinned official
    scorer, not an inferred replacement. Histories are supplied dataset thoughts
    in chronological order; generated predictions never update them.
    """
    if not callable(evaluator):
        raise TypeError("An explicitly bound offline Action evaluator is required")
    if not 1 <= len(slots) <= 56:
        raise ValueError("One to 56 recorded slots are required")
    for slot in slots:
        if not isinstance(slot, ActionSlot):
            raise TypeError("Action trajectories require ActionSlot")
        if slot.eligible and (not slot.recorded or not slot.request or slot.reference_action is None):
            raise ValueError("Eligible Action slots require request, recorded screen and offline reference")
        if any(not isinstance(text, str) or not text for text in slot.history):
            raise ValueError("Action history must contain dataset-recorded thought strings")

    def execute(release, slot, k, previous_feedback):
        return executor(release, slot.request, slot.history, k, previous_feedback)

    def score(prediction, slot):
        if not isinstance(prediction, ActionPrediction):
            raise ValueError("Action executor must return ActionPrediction")
        action = prediction.action
        if (not isinstance(action, dict) or set(action) != {"function", "arguments", "status"}):
            raise ValueError("Action prediction must be a parsed function/arguments/status object")
        if action["function"] == "INVALID" or action["status"] == "INVALID":
            scores = ActionScores(False, False, False)
        else:
            scores = evaluator(action, slot)
            if not isinstance(scores, ActionScores):
                raise TypeError("Offline Action evaluator must return ActionScores")
        return scores.step, {"action": dict(action), "function_correct": scores.function,
                             "arguments_correct": scores.arguments, "status_correct": scores.status}

    return _run_trajectory(slots, feature_loader, allocator, execute, score, ledger=ledger, action=True)


def _run_trajectory(slots, feature_loader, allocator, execute, score, *, ledger=None, action=False):
    ledger = ledger or PrivacyLedger([s.eligible for s in slots], [s.recorded for s in slots])
    expected_eligible=tuple(s.eligible for s in slots)+(False,)*(56-len(slots))
    expected_recorded=tuple(s.recorded for s in slots)+(False,)*(56-len(slots))
    if ledger.slot_index != 0 or ledger.eligible_mask != expected_eligible or ledger.recorded_mask != expected_recorded:
        raise ValueError("Ledger must start fresh with the exact fixed trajectory masks")
    transitions=[]; records=[]; previous_feedback=0.0; eligible_index=0
    for t in range(56):
        slot = slots[t] if t < len(slots) else Slot("", None, False, False)
        if slot.eligible:
            eligible_index += 1
        captured={}
        def trusted_allocate(observation):
            captured['observation']=np.asarray(observation, dtype=np.float32).copy()
            return allocator(observation)
        release=ledger.release_step(lambda:feature_loader(t), trusted_allocate, prior_feedback=previous_feedback)
        record={'slot':t, 'eligible':slot.eligible, 'invoked':release.invoked,
                'status':release.status, 'correct':False, 'candidate_count':release.k,
                'used_budget':release.used_budget, 'remaining_budget':release.remaining_budget,
                'executed_budgets':release.executed_budgets.tolist() if release.invoked else None,
                'refinement_scales':release.refinement_scales.tolist() if release.invoked else None}
        if action:
            record.update(function_correct=False, arguments_correct=False, status_correct=False)
        if release.invoked:
            try:
                prediction=execute(release.release, slot, release.k, previous_feedback)
                if not hasattr(prediction, 'feedback') or not math.isfinite(prediction.feedback) or not 0 <= prediction.feedback <= 2:
                    raise ValueError("Executor must return protected feedback in [0,2]")
                correct, details=score(prediction, slot)
            except Exception as error:
                record.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
                raise TrajectoryExecutionError(str(error), records+[record], used_budget=ledger.used_budget) from error
            mean_budget=float(np.mean(release.executed_budgets))
            reward=float(correct)+.5*(math.log(5)-math.log(mean_budget))/(math.log(5)-math.log(1.5))-.1*(release.k-1)/19
            transitions.append({'observation':captured['observation'], 'executed_budgets':release.executed_budgets.copy(),
                'candidate_count':release.k, 'reward':reward, 'success':correct,
                'remaining_budget':release.remaining_budget+float(np.sum(release.executed_budgets)),
                'eligible_index':eligible_index})
            previous_feedback=prediction.feedback
            record.update(correct=correct, mean_regional_budget=mean_budget, **details)
        records.append(record)
    for i, transition in enumerate(transitions):
        if i+1 < len(transitions):
            next_record=transitions[i+1]
            transition.update(next_observation=next_record['observation'],next_remaining_budget=next_record['remaining_budget'],terminal=False)
        else:
            # Eligible slots after final invocation are misses, not removed rows.
            tail=eligible_index-transition['eligible_index']
            transition['reward'] -= sum(.99**step for step in range(1,tail+1))
            transition.update(next_observation=np.zeros(28,dtype=np.float32),next_remaining_budget=0.0,terminal=True)
        del transition['eligible_index']
    return records, transitions


def save_replay(transitions, path):
    if not transitions:
        raise ValueError("No invoked transitions to save")
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=('observation','executed_budgets','candidate_count','reward','next_observation','terminal','remaining_budget','next_remaining_budget','success')
    arrays={key:np.asarray([row[key] for row in transitions]) for key in fields}
    metadata_fields=('slot_id','next_slot_id')
    if any(key in row for key in metadata_fields for row in transitions):
        if any(key not in row or not isinstance(row[key], str) for key in metadata_fields for row in transitions):
            raise ValueError("Replay public slot IDs must be present as strings on every transition")
        for key in metadata_fields:
            arrays[key]=np.asarray([row[key] for row in transitions], dtype=np.str_)
    for key in ('source_manifest_sha256','task','family_id'):
        if any(key in row for row in transitions):
            if any(key not in row or not isinstance(row[key], str) or not row[key] for row in transitions):
                raise ValueError(f"Replay {key} metadata must be a nonempty string on every transition")
            arrays[key]=np.asarray([row[key] for row in transitions], dtype=np.str_)
    with path.open('xb') as stream:
        np.savez_compressed(stream,**arrays)


def behavior_allocator(rng):
    """Public behavior law for an immutable buffer; no correctness input."""
    def allocate(_observation):
        return rng.uniform(1.5,5.0,size=25),int(rng.integers(1,21))
    return allocate


def inference_allocator(trainer, *, slot_id_provider=None, tms_schedule=None):
    """Delegate all method-specific composition to the bound trainer.

    Single-head variants require their evaluation TMS schedule and an explicit
    public slot-ID provider. No counter or constant counterpart is inferred from
    the observation. The privacy ledger alone applies the affordability filter.
    """
    import torch
    def allocate(observation):
        state=torch.tensor(np.asarray(observation)[None],dtype=torch.float32,device=trainer.device)
        slot_ids = None
        if slot_id_provider is not None:
            identifier = slot_id_provider(observation)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Evaluation slot-ID provider must return a nonempty public ID")
            slot_ids = [identifier]
        with torch.inference_mode():
            action=trainer.proposal_action(state, slot_ids=slot_ids, tms_schedule=tms_schedule)
        return action['budgets'][0].cpu().numpy(),int(action['candidate_count'][0])
    return allocate
