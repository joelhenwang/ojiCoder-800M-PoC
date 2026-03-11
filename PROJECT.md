Here’s the blueprint I’d use for an ~800M dense decoder-only code model optimized for reasoning, code completion for simple implementations/templating, type/lint fixes, and a compact tool-driven debug loop.

1) The model to build
A dense Transformer. At this size, recent results suggest the biggest gains come from data quality, staged training, and a handful of stabilizing transformer tweaks. SmolLM2’s results are strongly data-centric, and IMU-1 shows that small models benefit from a compact set of architectural and optimizer upgrades.

- Exact target spec:
Parameters: ~800M
Architecture: decoder-only Transformer
Layers: 24
Hidden size: 1536
Intermediate size (MLP): 4096 with SwiGLU
Attention heads: 16 query heads
Key/value heads: 4 via Grouped Query Attention
Head dim: 96
Positional encoding: RoPE
Context length: 8k initially, train for 16k only later
Norm: RMSNorm / pre-norm
Embeddings: tied input/output
Attention: causal, with FlashAttention-compatible layout


- Tweaks to include from day one:
QK-Norm
Value residual connections
LayerNorm scaling
Per-head gating
Grouped Query Attention


- What to try in the future:
Mixture-of-Experts
latent attention variants
state-space hybrids
speculative training tricks tied to giant-scale infra
giant context from the start
native multimodality


2) Tokenizer
- Use a SentencePiece Unigram or BPE tokenizer with:
vocab size: 64k
byte fallback: yes
aggressive preservation of:
indentation
whitespace
newlines
common programming punctuation
path-like strings
stack traces
shell commands

- A small set of code-structure special tokens, inspired by coding-model practice:
file separator
repository name
path separator
diff hunk markers
tool call start/end
observation start/end
memory summary marker


A practical set might be:

<|repo|>
<|path|>
<|file_sep|>
<|diff|>
<|tool_call|>
<|tool_result|>
<|memory|>
<|plan|>
<|patch|>
Do not go overboard; keep it compact.

3) Training objective (Use standard next-token prediction for base pretraining)

- Then specialize in later stages with mixtures of:
fill-in-the-middle
prefix-to-suffix completion
diff prediction
repair prediction
tool transcript prediction
trajectory summarization

- We want the model to learn:
finish a function
patch small mistakes
react to tool outputs
compress state across multiple debug steps

(The pretraining objective should broaden in later stages)

4) Data plan
Data curation and staged mixing matter more than almost anything else for small models.

- Final target data mix:
60–70% source code
15–20% code-adjacent text
10–15% synthetic coding tasks
5–10% general text
5–10% tool-use / repair / debug trajectories in late stages

- Data buckets
-- A. Source code
Prioritize:
Python
TypeScript / JavaScript
Go
Rust
Bash / shell
YAML / JSON
SQL
HTML / CSS

- Filter for:
permissive licenses only
dedup at file and near-duplicate level
parser-valid when possible
minimum quality thresholds
remove generated/minified/vendor files

-- B. Code-adjacent natural language
- Use:
README files
API docs
comments/docstrings
issue threads
commit messages
type/lint/test error explanations
StackOverflow-style Q/A if licensing allows


-- C. Synthetic code tasks
- Generate:
function completion prompts
signature-constrained implementations
refactor prompts
bug-fix prompts
test-to-code prompts
error-to-patch prompts
docstring-to-function prompts


-- D. Tool trajectories
- Create short executable trajectories around:
read_file
grep
find_symbol
run_tests
typecheck
lint
edit_file
show_diff
summarize_state


5) Training stages

- Stage 1 — broad base pretraining:
Goal: competent small language model with code bias.
Tokens = 20B–30B tokens

- Mix:
45% code
25% docs / technical text
20% general high-quality text
10% structured synthetic completion / repair

- Sequence length:
mostly 4k, some 8k

- Objective
mostly next-token
small amount of fill-in-the-middle


- Stage 2 — code-heavy continued pretraining
Goal: turn the base into a real code model.

Tokens = 15B–30B tokens

- Mix:
70% code
15% docs / API text
10% synthetic completion / repair
5% general text

- Add objectives:
fill-in-the-middle
repository-context completion
constrained completion

- Special formatting:
Start using:

<|repo|>

<|path|>

<|file_sep|>

import trees

local symbol lists


- Stage 3 — repair + debug specialization
Goal: teach the model to fix simple real-world issues.

Tokens / samples
5B–10B tokens equivalent
or several tens of millions of short specialized examples

- Mix:
failing type-check → patch
failing lint → patch
failing unit test → patch
stack trace → diagnosis + patch
diff completion
code review comments → patch

- Data construction
Use stronger models and static analyzers to score/filter samples. That follows the same general direction as recent code-data pipelines that rely on model-based scorers rather than only heuristics. (arXiv)

Stage 4 — compact tool-use / agent tuning
Goal: teach the model a bounded debug loop.

- Train on trajectories like:
inspect problem
retrieve relevant file/snippet
state hypothesis
patch
run test/lint/typecheck
read result
summarize next step
retry or stop

- Constraints:
max 3–5 turns
max 3 tools per turn
bounded memory
small action space
verifiable rewards


6) Optimizer and training recipe
- First-choice optimizer:
NorMuon + Cautious Weight Decay
IMU-1 reports clear gains from this stack over AdamW in its small-model recipe. (arXiv)

Fallback
If that becomes a stability or implementation sink:

AdamW with strong baseline settings

Other recipe choices. Use:
bf16
gradient clipping
cosine decay
warmup
EMA checkpoints late in training
checkpoint averaging for eval candidates
global batch scaled conservatively
sequence packing
fused kernels where available

IMU-1 also reports benefit from post-hoc checkpoint EMA and a staged schedule. (arXiv)

7) Context length plan
Do not start by training everything at long context.

- Use:
Stage 1: mostly 4k
Stage 2: mix of 4k + 8k
Stage 3: mostly 8k
optional short final extension to 16k

For your use case, long-horizon behavior should come more from context management than just a giant native context window. Recent work on coding agents supports that direction. Context as a Tool argues for explicit context-maintenance actions, and repository-memory work points the same way.

8) Post-training curriculum

- Phase A — supervised code assistant tuning
Train on:
completion
implementation
repair
explanation
minimal refactor
“do not over-edit” behavior
Keep responses short and structured.

- Phase B — supervised tool-use tuning
Teach a narrow schema such as:

<|tool_call|>{"name":"read_file","args":{"path":"src/foo.py"}}</|tool_call|>
<|tool_result|>...</|tool_result|>
<|plan|>Need to fix the type mismatch in bar().</|plan|>
<|patch|>...</|patch|>

- Phase C — preference optimization
Reward:
correct tool selection
minimal edits
preserving signatures
successful fixes
concise summaries
stopping when solved

Penalize:
unnecessary tool use
over-editing
hallucinated file paths
looping

- Phase D — narrow RL or RLVR
Only if your environment is stable.

Use verifiable signals:
tests pass
type-check passes
lint passes
patch compiles
fewer retries
smaller valid patch

Tool-R0 and related work support the idea that verifiable tool settings are a good fit for RL-style improvement, but for your case I would keep the environment tightly bounded. (arXiv)

9) Reasoning vs tool use: do not train them as one blob
This is one of the clearest 2026 lessons.

DART shows that reasoning and tool-use updates can interfere, and disentangling them can improve results. For your model, that means:

keep the base / reasoning assistant tuning separate from

the tool policy specialization

Practically, I would do this with:

base model + full supervised tuning for coding assistant behavior

then LoRA/adapters for tool-use policy

optionally another adapter for memory/context compression

That gives you cleaner iteration and less destructive interference. (arXiv)

10) Memory and long-horizon design
For the “compacted long horizon step-by-step debug loop,” I would train the model to explicitly maintain a small structured working memory.

Use a memory block like:

<|memory|>
Goal: make tests in tests/test_parser.py pass
Known facts:
- failure is in parse_config()
- bad type: Optional[str] returned where str expected
Tried:
- changed default handling in line 83, still failing
Next hypothesis:
- caller expects normalization before None check
</|memory|>
Teach the model when to:

update memory

compress history

discard stale observations

restate the current hypothesis

That aligns with the direction in Context as a Tool and repo-memory work: long-horizon coding agents improve when context management is explicit rather than purely append-only. (arXiv)

11) What to evaluate
Do not rely only on HumanEval-style benchmarks.

Completion / code quality
function completion

constrained completion

type-aware completion

simple bug fixing

pass@k where appropriate

Repair
lint fix success

type-check fix success

unit-test repair success

minimal-diff score

Tool use
tool-call accuracy

unnecessary tool-call rate

patch success after tool observation

number of steps to solve

Long-horizon compact loop
success within 3 turns

success within 5 turns

memory usefulness

context compression quality

Terminal-Bench 2.0 is a useful reminder that open-ended terminal autonomy is hard even for much larger agents, so your evals should stay tightly scoped. (arXiv)

12) The practical target capability
If trained well, this 800M model should aim to be good at:

templated code completion

short function implementation

preserving type signatures

fixing common type/lint/test issues

simple repo-local debugging loops

compact state summarization between steps

It should not be expected to:

autonomously solve hard long-horizon SWE tasks

outperform frontier coding agents

deeply reason across very large repositories with no retrieval help

That expectation matches the current state of coding-agent benchmarks. (arXiv)

13) My recommended minimal v1
If you want the highest chance of success, start with this narrower v1:

Model
24 layers

d_model 1536

16 Q heads / 4 KV heads

SwiGLU 4096

RoPE

QK-Norm

value residual

RMSNorm

vocab 64k

context 8k

Data
25B base tokens

20B code CPT tokens

specialized synthetic/repair/tool corpus after that

Training
AdamW first if NorMuon slows progress

bf16

FlashAttention-compatible kernels

multi-stage checkpoints and ablations

Post-training
SFT for coding assistant

SFT for narrow tool use

DPO or similar preference tuning

bounded RL only after environment is reliable

That version is modern without becoming fragile.

14) My strongest recommendation
The part most likely to make or break this project is data and curriculum, not the exact hidden size.

So if you have to prioritize engineering effort, do it in this order:

excellent code-data cleaning and dedup

good synthetic completion/repair data

narrow executable tool environment

explicit memory/context format

small architectural upgrades

optimizer experimentation