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


def correct_grounding(point, box) -> bool:
    if point is None or box is None:
        return False
    x, y = point
    return bool(math.isfinite(x) and math.isfinite(y) and box[0] <= x <= box[2] and box[1] <= y <= box[3])


def run_trajectory(slots: list[Slot], feature_loader: Callable,
                   allocator: Callable, executor: Callable, *, ledger=None):
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
        if release.invoked:
            prediction=executor(release.release, slot.instruction, release.k)
            if not isinstance(prediction, Prediction) or not math.isfinite(prediction.feedback) or not 0 <= prediction.feedback <= 2:
                raise ValueError("Executor must return Prediction with protected feedback in [0,2]")
            correct=correct_grounding(prediction.point, slot.target_box)
            mean_budget=float(np.mean(release.executed_budgets))
            reward=float(correct)+.5*(math.log(5)-math.log(mean_budget))/(math.log(5)-math.log(1.5))-.1*(release.k-1)/19
            transitions.append({'observation':captured['observation'], 'executed_budgets':release.executed_budgets.copy(),
                'candidate_count':release.k, 'reward':reward, 'success':correct,
                'remaining_budget':release.remaining_budget+float(np.sum(release.executed_budgets)),
                'eligible_index':eligible_index})
            previous_feedback=prediction.feedback
            record.update(correct=correct, point=prediction.point, mean_regional_budget=mean_budget)
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
    with path.open('xb') as stream:
        np.savez_compressed(stream,**arrays)


def behavior_allocator(rng):
    """Public behavior law for an immutable buffer; no correctness input."""
    def allocate(_observation):
        return rng.uniform(1.5,5.0,size=25),int(rng.integers(1,21))
    return allocate


def inference_allocator(trainer):
    """Compose trained heads into proposals; the ledger applies the filter."""
    import torch
    def allocate(observation):
        state=torch.tensor(np.asarray(observation)[None],dtype=torch.float32,device=trainer.device)
        with torch.inference_mode():
            if trainer.method in ('Independent','Independent-1M'):
                budget=trainer.bundles['disclosure']['actor'].evaluation_action(state)['budgets']
                count=trainer.bundles['count']['actor'].evaluation_action(state)['candidate_count']
            else:
                action=trainer.bundles['joint']['actor'].evaluation_action(state)
                budget,count=action['budgets'],action['candidate_count']
        return budget[0].cpu().numpy(),int(count[0])
    return allocate
