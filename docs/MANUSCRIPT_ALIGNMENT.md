# Current-manuscript alignment

This update aligns the runtime with the current author-specified JDC manuscript derived from the 44-page parameter-unified handoff and later edits. It does not modify reported results or claim recovery of historical experiments.

| Contract | Implementation |
|---|---|
| 256-wide joint/standalone; 170-wide Independent; exact counts | controller.py |
| AdamW; Kaiming/Xavier/zero bias; zero-based Gumbel; float64 density | controller.py and reference configuration |
| Critic→actor→Polyak; CB no successor; immutable replay | trainer.py / replay.py |
| Development-mean TMS with selected H and population bindings | tms.py / cli.py |
| Every-10k development selection; selected versus resumable last state | selection.py |
| Grounding and Action released-input paths | executor.py / runtime.py / evaluation.py |
| Frozen Train-only retrieval with public exclusion views | retrieval.py |
| Explicit Action schema/alias/types and official scorer boundary | action_evaluation.py / scoring.py |
| Ten families and three post-fit evaluations | protocol.py; supplied frozen executor stage boundary |

Current H/CB online/target/stored counts are 261192/169986/431178; combined Independent 247254/167284/414538. Other frozen models/projection are excluded. Configuration drift and version1 checkpoints are rejected.

Tests cover actual parameter shapes, initialization, optimizer sequencing, TMS assignment/filtering, seed isolation, checkpoint resume, development denominator/ties, Action/retrieval boundaries and synthetic end-to-end drivers. Tiny Qwen tests use randomly initialized small models, never benchmark fallbacks. CI runs CPU checks and a three-update smoke.

Experimenters still supply data/splits, paired fitted family artifacts, roots, schema/official Action evaluator, retrieval bank source records and development gate, and same-screen pairing definitions. These are not inferred where the source is unspecified. No historical table constants are accepted as newly measured results.

One RTX 5090 is author-confirmed. CPU checks do not verify historical GPU timing or software versions. See the runtime guide and protocol document for supported scope and explicit prerequisites.
