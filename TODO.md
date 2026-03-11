# TODO

Last updated: 2026-03-11

This file tracks what is already implemented in the repository, what is partially complete, and what still needs to be built to reach a real v1 training run for the 800M code model described in PROJECT.md.

## Status legend

- [x] Done
- [~] Partially done or scaffolded only
- [ ] Not started

## Current snapshot

- [x] Greenfield repository bootstrap is complete.
- [x] The core package layout exists under `src/eight_hundred_m/`.
- [x] The project has a working config system, model scaffold, tokenizer scaffold, data scaffold, training scaffold, CLI, and tests.
- [~] The repo supports planning, manifest handling, external data adaptation, packing, and smoke validation.
- [ ] The repo does not yet run a real end-to-end training job.
- [ ] The repo does not yet include a production training loop with optimizer, scheduler, checkpoint resume, and evaluation.

## Completed foundation work

### Planning and project scaffolding

- [x] Create `PLAN.md` with scope, milestones, fixed assumptions, token budgets, and risk tracking.
- [x] Create `pyproject.toml` with runtime and development dependencies.
- [x] Create package entrypoints in `src/eight_hundred_m/__init__.py` and `src/eight_hundred_m/__main__.py`.
- [x] Add `pyrightconfig.json` for `src`-layout analysis.

### Model and configuration layer

- [x] Implement `TokenizerConfig` in `src/eight_hundred_m/config.py`.
- [x] Implement `TrainingStageConfig` in `src/eight_hundred_m/config.py`.
- [x] Implement `ModelConfig` in `src/eight_hundred_m/config.py`.
- [x] Add config validation and serialization helpers.
- [x] Add JSON config loaders for tokenizer, model, and training stages.
- [x] Add frozen model artifact at `configs/model/800m.json`.
- [x] Freeze the base tokenizer contract in `configs/tokenizer/base_bpe.json`.
- [x] Freeze staged training settings in `configs/train/stages.json`.

### Model architecture scaffold

- [x] Implement `RMSNorm`.
- [x] Implement `SwiGLU`.
- [x] Implement RoPE helpers.
- [x] Implement grouped-query attention.
- [x] Implement QK normalization hooks.
- [x] Implement layer scaling.
- [x] Implement value residual plumbing.
- [x] Implement per-head gating hooks.
- [x] Implement a shape-safe decoder-only model scaffold in `src/eight_hundred_m/modeling.py`.
- [~] The model scaffold is implemented, but a real training-path smoke using packed data still needs to be wired up.

### Tokenizer pipeline scaffold

- [x] Add tokenizer spec dataclasses in `src/eight_hundred_m/tokenizer/spec.py`.
- [x] Add tokenizer corpus manifest artifact at `configs/tokenizer/corpus_manifest.json`.
- [x] Implement tokenizer corpus discovery in `src/eight_hundred_m/tokenizer/corpus.py`.
- [x] Fix corpus discovery to support brace-glob patterns.
- [x] Implement normalization loading hooks for tokenizer training input.
- [x] Implement tokenizer validation helpers for whitespace, path, diff, and tool transcript fidelity.
- [x] Implement tokenizer training plan helpers in `src/eight_hundred_m/tokenizer/trainer.py`.
- [x] Implement tokenizer runtime loading in `src/eight_hundred_m/tokenizer/runtime.py`.
- [~] Tokenizer training code exists, but a real tokenizer artifact still needs to be trained on the target machine.

### Data manifest and planning layer

- [x] Add base data manifest artifact at `configs/data/base_manifest.json`.
- [x] Implement `DataFilterPolicy` in `src/eight_hundred_m/data/manifest.py`.
- [x] Implement `DataSourceManifestEntry` in `src/eight_hundred_m/data/manifest.py`.
- [x] Extend the data manifest schema for external offline sources.
- [x] Implement data-file discovery in `src/eight_hundred_m/data/planning.py`.
- [x] Implement generated and vendor filtering.
- [x] Implement shard planning helpers.
- [x] Implement planning summary and shard artifact writing.
- [~] File-level planning is working, but parser-valid filtering, near-duplicate dedup, and stronger quality filters are still pending.

### External Stage 1 data ingestion

- [x] Add language normalization helpers in `src/eight_hundred_m/data/languages.py`.
- [x] Freeze the selected Stage 1 external language set.
- [x] Add offline Stage 1 external manifest at `configs/data/stage1_external_manifest.json`.
- [x] Add `The Stack v2 dedup` record adaptation.
- [x] Add `CommitPackFT` record adaptation.
- [x] Restrict `CommitPackFT` Stage 1 ingestion to `new_contents` mode.
- [x] Implement JSONL record loading for offline dataset slices.
- [x] Implement provenance-carrying normalized external sample objects.
- [x] Implement filtering and summarization for external samples.
- [x] Implement writing external sample shard artifacts.
- [~] The adapter layer is ready, but real offline dataset runs still need to happen on the training machine.

### Packing and smoke validation

- [x] Implement heuristic token estimation for Stage 1 packing.
- [x] Implement sample text normalization before packing.
- [x] Implement packed sample artifact writing.
- [x] Implement packed sample artifact loading.
- [x] Add optional tokenizer-backed token counting for exact packing.
- [x] Record the token counting method used for packed samples.
- [x] Implement smoke-batch validation against a tokenizer artifact and target sequence length.
- [~] Packing is production-usable once a real tokenizer artifact is available.

### Training runner scaffold

- [x] Implement stage runtime planning in `src/eight_hundred_m/training/stages.py`.
- [x] Implement run-plan and optimizer-step estimation in `src/eight_hundred_m/training/runner.py`.
- [x] Implement run directory initialization in `src/eight_hundred_m/training/loop.py`.
- [x] Implement run-plan artifact writing.
- [x] Implement per-stage checkpoint placeholder writing.
- [x] Implement Stage 1 external data preparation during run initialization.
- [x] Allow training initialization to use tokenizer-backed packing when a tokenizer artifact is supplied.
- [~] The training loop is still a skeleton and does not yet execute forward/backward optimization steps.

### CLI surface

- [x] Add `tokenizer-summary` command.
- [x] Add `data-plan` command.
- [x] Add `external-pack` command.
- [x] Add `run-summary` command.
- [x] Add `init-run` command.
- [x] Add `smoke-batch` command.
- [x] Support `--tokenizer-artifact` in pack and run-init flows.

### Test coverage

- [x] Add config tests.
- [x] Add model tests.
- [x] Add manifest tests.
- [x] Add pipeline planning tests.
- [x] Add training runner tests.
- [x] Add CLI and run loop tests.
- [x] Add external ingest tests.
- [x] Add tokenizer runtime and smoke-batch tests.
- [~] Current tests validate scaffolding and smoke behavior, not real training quality or training stability.

## Immediate next work

### Ready to do now on the training machine

- [ ] Create and validate the real Python environment with PyTorch 2.10 and ROCm 7.2.
- [ ] Install runtime dependencies from `pyproject.toml`.
- [ ] Train the project-owned BPE tokenizer and save a real tokenizer artifact.
- [ ] Run tokenizer validation on the trained artifact and inspect preservation metrics.
- [ ] Run Stage 1 external packing with `--tokenizer-artifact` so sample lengths use exact token counts.
- [ ] Run `smoke-batch` on the packed artifacts to confirm sequence-length fit before training.
- [ ] Run `init-run` with the real tokenizer artifact and external manifest to produce the first realistic run directory.

### Highest-priority implementation gaps

- [ ] Add a tensorization layer that converts packed sample artifacts into model-ready token batches.
- [ ] Add attention-mask and label construction for causal LM training.
- [ ] Add a tiny forward-pass smoke that loads packed data and runs the actual model.
- [ ] Add a loss computation smoke test with realistic batch shapes.
- [ ] Replace checkpoint placeholder writing with real checkpoint serialization.
- [ ] Implement a minimal single-node training loop.
- [ ] Add optimizer setup for the baseline `AdamW` path.
- [ ] Add learning-rate scheduler setup with warmup and cosine decay.
- [ ] Add gradient clipping.
- [ ] Add mixed-precision handling.
- [ ] Add periodic evaluation hooks.
- [ ] Add resume-from-checkpoint support.

## Short-term backlog

### Data quality and preprocessing

- [ ] Implement exact permissive-license allowlist enforcement.
- [ ] Add parser-valid filtering where feasible.
- [ ] Add stronger generated/minified file detection.
- [ ] Add file-level dedup verification reports.
- [ ] Add near-duplicate dedup beyond file identity.
- [ ] Add quality scoring and rejection criteria for external samples.
- [ ] Add token-aware sampling reports after real tokenizer training.

### Training system hardening

- [ ] Add proper batch packing for mixed sequence lengths.
- [ ] Add stage-aware dataloaders.
- [ ] Add metrics logging for throughput, loss, tokens/sec, and memory.
- [ ] Add EMA support if it remains in scope for v1.
- [ ] Add validation dataset wiring.
- [ ] Add reproducible seed handling and run metadata capture.
- [ ] Add better checkpoint metadata for resume and analysis.

### Tokenizer and formatting follow-ups

- [ ] Benchmark one reputable external code tokenizer for compression and fidelity only.
- [ ] Decide whether to add chat-format wrapper tokens in a later SFT-specific tokenizer revision.
- [ ] Add tokenizer training/validation CLI commands if the current workflow feels too manual.

## Medium-term milestones

### Stage 2 code continued pretraining

- [ ] Finalize Stage 2 data mix and sequence-length policy.
- [ ] Add repository-context formatting beyond the current basic structure.
- [ ] Add fill-in-the-middle formatting for Stage 2.
- [ ] Add constrained completion evaluation inputs.

### Stage 3 repair and debugging specialization

- [ ] Add lint-failure to patch dataset format.
- [ ] Add type-failure to patch dataset format.
- [ ] Add failing-test to patch dataset format.
- [ ] Add stack-trace to diagnosis and patch dataset format.
- [ ] Add diff-completion tasks.
- [ ] Add synthetic repair data builders.
- [ ] Add quality filters for minimal patch behavior.

### Stage 4 tool-use specialization

- [ ] Define the bounded tool schema.
- [ ] Define serialized tool trajectory format.
- [ ] Add memory block formatting.
- [ ] Add tool-policy tuning path that stays separate from base model training.
- [ ] Add bounded debug-loop evaluation cases.

## Evaluation backlog

- [ ] Build completion evaluation harness.
- [ ] Add constrained completion metrics.
- [ ] Add type-aware completion metrics.
- [ ] Add `pass@k` task evaluation.
- [ ] Build repair evaluation harness.
- [ ] Add lint/type/test fix-rate metrics.
- [ ] Add minimal-diff scoring.
- [ ] Build tool-use evaluation harness.
- [ ] Add tool-call accuracy and unnecessary-tool metrics.
- [ ] Build memory summary evaluation.

## Decisions that still need to be made

- [ ] Freeze the exact permissive-license allowlist.
- [ ] Freeze the near-duplicate dedup strategy.
- [ ] Freeze the exact QK-norm, layer-scaling, and value-residual formulations if current defaults change.
- [ ] Decide whether 16k context belongs in v1 or a later extension milestone.
- [ ] Decide whether preference optimization is part of v1.
- [ ] Decide the release posture for model weights, tokenizer, and data artifacts.

## Useful project checkpoints

- [x] Repo can build plans and manifests.
- [x] Repo can describe and validate tokenizer/data/training configs.
- [x] Repo can adapt offline Stage 1 external samples.
- [x] Repo can produce packed sample artifacts.
- [x] Repo can validate packed sample lengths with a tokenizer artifact.
- [ ] Repo can run a model-forward smoke on real packed data.
- [ ] Repo can run a single real optimization step.
- [ ] Repo can run a stable Stage 1 training job.

## Suggested maintenance rule for this file

- [ ] Update this file whenever a milestone task changes state.
- [ ] Keep completed items checked instead of deleting them.
- [ ] Add dates or short notes beside major completions once real training begins.