"""Explicit NEW prompt policy. The caller supplies the real full-prompt renderer.

This module never substitutes a character count for a tokenizer and never drops
an overflowing example from the evaluation denominator.
"""
from dataclasses import dataclass
import math
from typing import Callable, Sequence


class PromptTooLong(ValueError):
    """Abort the new run: the current instruction cannot fit this protocol."""


@dataclass(frozen=True)
class Retrieval:
    text: str
    score: float


@dataclass(frozen=True)
class PromptPayload:
    task: str
    current: str
    history: tuple[str, ...]
    retrieval: tuple[Retrieval, ...]

    @property
    def scoring_text(self) -> str:
        return "\n".join((self.current,) + self.history)


@dataclass(frozen=True)
class PreparedPrompt:
    payload: PromptPayload
    input_ids: tuple[int, ...]
    removed_history: int
    removed_retrieval: int


def prepare_prompt(task: str, current: str, history: Sequence[str],
                   retrieval: Sequence[Retrieval],
                   render_token_ids: Callable[[PromptPayload], Sequence[int]],
                   token_limit: int = 4096, max_previous_steps: int = 10) -> PreparedPrompt:
    """Re-render with the real tokenizer/template after every whole-entry trim.

The callback must return ALL model input token IDs, including wrapper, visual
placeholders and generation prefix. Its revision must be saved with the run.
It must neither add to nor modify the supplied scoring text. Input embedding
lookup is deliberately outside this policy utility.
"""
    if task not in ("G", "A") or not isinstance(current, str) or not current:
        raise ValueError("G/A and a non-empty current instruction are required")
    if isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit < 1:
        raise ValueError("token_limit must be a positive integer")
    if isinstance(max_previous_steps, bool) or not isinstance(max_previous_steps, int) or max_previous_steps < 0:
        raise ValueError("max_previous_steps must be a nonnegative integer")
    if any(not isinstance(x, str) or not x for x in history):
        raise ValueError("history must contain non-empty original strings")
    if any(not isinstance(x, Retrieval) or not isinstance(x.text, str) or not x.text or not math.isfinite(x.score) for x in retrieval):
        raise ValueError("retrieval entries require text and finite scores")
    # No normalization of the original Unicode strings.
    for text in [current, *history, *(x.text for x in retrieval)]:
        text.encode("utf-8", errors="strict")
    retained = list(history[-max_previous_steps:]) if task == "A" and max_previous_steps else []
    removed_history = len(history) - len(retained)
    retained_retrieval = list(retrieval)
    removed_retrieval = 0
    while True:
        payload = PromptPayload(task, current, tuple(retained), tuple(retained_retrieval))
        token_ids = tuple(render_token_ids(payload))
        if not token_ids or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in token_ids):
            raise ValueError("renderer must return non-empty actual integer token IDs")
        if len(token_ids) <= token_limit:
            return PreparedPrompt(payload, token_ids, removed_history, removed_retrieval)
        if retained:
            retained.pop(0)
            removed_history += 1
        elif retained_retrieval:
            index = min(range(len(retained_retrieval)), key=lambda i: (retained_retrieval[i].score, -i))
            retained_retrieval.pop(index)
            removed_retrieval += 1
        else:
            raise PromptTooLong("Current instruction plus fixed wrapper/visual tokens exceeds the cap; abort run, do not silently exclude this example")
