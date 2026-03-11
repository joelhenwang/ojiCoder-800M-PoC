# PLAN

## 1. Project scope

Build a full v1 training-stack repository for an ~800M dense decoder-only code model described in [PROJECT.md](PROJECT.md).

### v1 goals
- Implement the model architecture and configuration system.
- Build tokenizer, data, training, and evaluation scaffolding.
- Support staged pretraining and later specialization for repair/tool use.
- Optimize first for single-node development on Halo Strix (`gfx1151`) with ROCm 7.2.

### Non-goals for v1
- Frontier-scale distributed training from day one.
- Open-ended SWE-bench-style autonomy.
- RL-heavy post-training before the supervised stack is stable.
- MoE, multimodality, or hybrid architectures.

## 2. Fixed assumptions

- Runtime stack: Python + PyTorch 2.10 + ROCm 7.2.
- Model framework: Hugging Face `transformers>=5.2.0` plus custom modules where needed.
- Initial hardware target: single node only.
- Baseline optimizer: `AdamW`.
- Advanced optimizer (`NorMuon + Cautious Weight Decay`) is an ablation after baseline stability.
- Default context target: 8k for v1 training, 16k only as a later extension milestone.
- Tokenizer: project-owned `BPE`, 64k vocab, byte fallback.
- Base tokenizer excludes `"<|im_start|>"`, `"<|im_end|>"`, `"<think>"`, and `"</think>"`.
- Release posture: undecided.

## 3. Token budgets

- Prototype / pipeline shakedown: 0.8B–2.5B effective tokens.
- Preferred v1 envelope: 43B–52B effective tokens.
- Stage 1 base pretraining target: 20B–25B tokens.
- Stage 2 code CPT target: 15B–20B tokens.
- Stage 3 repair/debug target: 3B–6B token-equivalent.
- Stage 4 tool-policy target: 0.3B–1B token-equivalent.

## 4. Frozen v1 architecture contract

- Decoder-only Transformer.
- ~800M parameters.
- 24 layers.
- Hidden size: 1536.
- MLP size: 4096 with `SwiGLU`.
- Attention: 16 query heads, 4 key/value heads, head dim 96.
- `RoPE` positional encoding.
- `RMSNorm` pre-norm.
- Tied token embedding and LM head.
- Causal attention with FlashAttention-compatible tensor layout.

### Required architectural additions
- `GroupedQueryAttention`.
- `QK-Norm`.
- Value residual connections.
- Layer scaling.
- Per-head gating.

### Acceptance checks
- Forward pass works for training and generation shapes.
- Parameter count lands within agreed tolerance of the 800M target.
- KV head sharing is validated by shape tests.
- RoPE application is correct at 4k and 8k sequence lengths.

## 5. Initial repository structure

```text
src/
	eight_hundred_m/
		config.py
		modeling.py
		tokenizer/
		data/
		training/
		eval/
configs/
	model/
	data/
	train/
scripts/
tests/
artifacts/
docs/
```

## 6. Milestones

### M0 — Repository bootstrap
- Status: complete.
- Create Python project metadata and dependency definitions.
- Create package skeleton and config loading utilities.
- Add initial test layout and base configuration files.
- Add this execution plan and decision log.

### M1 — Model skeleton
- Status: complete for scaffold scope.
- Implement config dataclasses for model/training/data stages.
- Implement `RMSNorm`, `SwiGLU`, RoPE helpers, and grouped-attention interfaces.
- Build a minimal model forward path that validates tensor shapes.
- Add parameter counting and model-spec validation helpers.

### M2 — Tokenizer pipeline
- Status: scaffold complete, artifact training pending.
- Implement tokenizer corpus sampling and normalization hooks.
- Define authoritative tokenizer metadata in config and artifact files.
- Validate preservation of code whitespace and punctuation structure.
- Add tokenizer runtime loading for real artifact use during downstream packing and smoke checks.
- Train and validate the project-owned `BPE` tokenizer.
- Train a 64k tokenizer with byte fallback.
- Benchmark one reputable external code tokenizer for compression and fidelity only.
- Version special tokens: `<|repo|>`, `<|path|>`, `<|file_sep|>`, `<|diff|>`, `<|tool_call|>`, `<|tool_result|>`, `<|memory|>`, `<|plan|>`, `<|patch|>`.

### M3 — Data pipeline MVP
- Status: mostly complete for Stage 1 offline flow.
- Define source manifests and extend them for offline external datasets.
- Implement file filtering for generated, minified, vendor, and invalid files.
- Implement JSONL adaptation for `bigcode/the-stack-v2-dedup` and `bigcode/commitpackft` offline slices.
- Restrict `commitpackft` Stage 1 use to `new_contents`.
- Add language bucketing and sampling reports.
- Emit train-ready packed text shards with provenance metadata.
- Implement file-level dedup and leave near-duplicate detection as a tracked subtask.
- Finish license allowlist enforcement and stronger quality filtering.

### M4 — Pretraining stage runner
- Status: planning and run initialization complete, real training loop pending.
- Implement stage-aware training configs.
- Encode explicit token budgets and sequence-length mixes in stage configs.
- Support next-token training first.
- Add optional fill-in-the-middle formatting hooks.
- Support mixed sequence lengths with packing.
- Add CLI-driven run summaries, run initialization, external packing, and smoke-batch validation.
- Add tensorization, loss computation, and model-forward smoke on packed samples.
- Add checkpointing, resume, gradient clipping, cosine decay, warmup, EMA, and eval hooks.

### M5 — Code CPT and repair specialization
- Status: not started.
- Add stage-2 code-heavy continued pretraining config.
- Add stage-3 repair datasets: lint, type, test, diff, and stack-trace repair.
- Add synthetic task builders and quality filters.
- Define success criteria for patch minimality and repair accuracy.

### M6 — Tool-use specialization
- Status: not started.
- Create bounded tool schema and serialized trajectory format.
- Add memory block formatting and summarization targets.
- Implement adapter-based tool-policy tuning.
- Keep tool-policy training separate from base assistant training.

### M7 — Evaluation harness
- Status: not started.
- Completion metrics: constrained completion, type-aware completion, `pass@k`.
- Repair metrics: lint/type/test fix rate and minimal-diff score.
- Tool metrics: tool-call accuracy, unnecessary tool rate, bounded-turn success.
- Memory metrics: summary usefulness and compression quality.

## 7. Current implementation status

- M0 is complete.
- M1 is complete for the planned scaffold and shape-validation scope.
- M2 is complete at the repository level, but still requires training a real tokenizer artifact on the target machine.
- M3 is complete enough to run Stage 1 offline ingestion and packing once the tokenizer artifact exists.
- M4 is partially complete: planning, run initialization, and smoke validation exist, but tensorization and actual optimization steps are not implemented yet.
- M5 through M7 remain future work.

## 8. Workstream order

1. Bootstrap repo and config system.
2. Freeze architecture in code.
3. Add shape tests and parameter-count checks.
4. Add tokenizer training pipeline.
5. Add data ingestion MVP.
6. Train the real tokenizer artifact and rerun packing with exact token counts.
7. Add packed-sample tensorization and a model-forward smoke path.
8. Add a minimal single-node stage-1 trainer.
9. Add stage-2/3 dataset formats.
10. Add tool-trajectory formats and adapter tuning.
11. Add full evaluation harness.

## 9. Decision log

### Already decided
- Use Hugging Face plus custom model code rather than a fully custom training stack.
- Start with single-node development.
- Use `AdamW` as the default first optimizer.
- Use a project-owned `BPE` tokenizer for v1.
- Exclude chat/thinking wrapper tokens from the base tokenizer.
- Use offline slices from `bigcode/the-stack-v2-dedup` and `bigcode/commitpackft` for Stage 1 external ingestion.
- Restrict `commitpackft` Stage 1 ingestion to `new_contents`.

### Open decisions
- Exact permissive-license allowlist.
- Near-duplicate dedup implementation.
- Exact `QK-Norm`, layer scaling, and value-residual formulation.
- RoPE extension method for 16k.
- Preference optimization algorithm.
- Release policy.

## 10. Immediate execution checklist

- [x] Create project metadata and package layout.
- [x] Add model config dataclasses.
- [x] Add architecture skeleton with shape-safe modules.
- [x] Add base model YAML/JSON config.
- [x] Add initial tests for config validation and parameter estimation.
- [x] Add tokenizer and data package placeholders.
- [x] Freeze tokenizer artifact schema and ownership rules.
- [x] Add train-stage artifact/config files.
- [x] Add config validation for token budgets and stage formatting flags.
- [x] Add tokenizer/data manifest loading and planning utilities.
- [x] Add executable tokenizer validation and data artifact smoke path.
- [x] Add offline external dataset manifest for Stage 1 sources.
- [x] Add Stage 1 external dataset adapter scaffolding for The Stack v2 dedup and CommitPackFT.
- [x] Add heuristic and tokenizer-backed Stage 1 packing.
- [x] Add tokenizer runtime loading for exact token counting.
- [x] Add CLI commands for external packing, run initialization, and smoke-batch validation.
- [ ] Train the real tokenizer artifact on the target machine.
- [ ] Run Stage 1 external packing with the real tokenizer artifact.
- [ ] Add packed-sample tensorization and model-forward smoke.
- [ ] Implement the first minimal real training loop.

## 11. Risks and fallback paths

- If ROCm-specific kernels slow development, keep a pure PyTorch reference path.
- If `NorMuon` is unstable or costly to implement, keep it out of the critical path.
- If long-context work destabilizes training, hold v1 at 8k.
- If near-duplicate dedup is too slow initially, ship file-level dedup first and track the rest.
- If tool-use data quality is weak, keep the base model and tool policy in separate adapters until better traces exist.
- If checkpoint interoperability becomes a goal later, revisit tokenizer reuse only as a deliberate compatibility decision.
