<!-- program.md -->
# DeepOMAPNet autoresearch

Autonomous optimization of the DeepOMAPNet CITE-seq model.
Mirrors the [autoresearch](https://github.com/karpathy/autoresearch) pattern —
the agent modifies `train.py`, runs 5-minute experiments, and keeps only improvements.

## Setup

To start a new run, work with the user to:

1. **Agree on a run tag** based on today's date (e.g. `mar29`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files** — these are small, read all of them:
   - `research/prepare.py` — fixed constants, synthetic data, evaluation. Do not modify.
   - `research/train.py` — the file you modify. Model, optimizer, hyperparameters.
   - `scripts/model/doNET.py` — model source (read-only context).
4. **Initialize results.tsv**: create `research/results.tsv` with just the header row.
5. **Confirm setup** and kick off experimentation.

## Experimentation

Each experiment runs for a **fixed time budget of 5 minutes** (wall-clock training time).
Launch it as:

```bash
cd DeepOMAPNet
python research/train.py > research/run.log 2>&1
```

**What you CAN do:**
- Modify `research/train.py` — this is the only file you edit.
  Everything is fair game: architecture hyperparameters, optimizer, loss weights,
  learning rate schedule, regularization.

**What you CANNOT do:**
- Modify `research/prepare.py`. It is read-only.
  It contains the fixed evaluation metric, data generation, and graph construction.
- Import packages not already available in the conda/pip environment.
- Change `NUM_CELLS`, `NUM_GENES`, `NUM_ADTS`, `TIME_BUDGET`, or `RANDOM_SEED`
  (these are fixed constants defined in `prepare.py`).
- Change the `GATWithTransformerFusion` model source in `scripts/model/doNET.py`.

**The goal: get the lowest `val_nrmse`.** Lower is better.
Since the time budget is fixed, everything is fairly comparable regardless of model size
or batch configuration.

`val_pearson` (mean per-protein Pearson r) and `val_auc` (AML AUC-ROC) are informational —
they don't affect whether you keep or discard an experiment.

**Simplicity criterion**: all else being equal, simpler is better.
A 0.002 val_nrmse improvement from deleting code? Keep it.
A 0.001 improvement that adds 30 lines of hacky workarounds? Not worth it.

**The first run**: always establish the baseline first — run `train.py` unchanged.

## Output format

```
---
val_nrmse:         0.312450
val_pearson:       0.721300
val_auc:           0.883000
training_seconds:  300.1
total_seconds:     302.4
peak_memory_mb:    0.0
num_steps:         4821
num_params_M:      0.38
hidden_channels:   64
num_layers:        2
device:            mps
```

Extract the key metric:
```bash
grep "^val_nrmse:" research/run.log
```

## Logging results

Log to `research/results.tsv` (tab-separated, NOT comma-separated).
**Do not git-commit this file.**

Header + columns:
```
commit	val_nrmse	val_pearson	val_auc	status	description
```

1. `commit` — short git hash (7 chars)
2. `val_nrmse` — e.g. `0.312450`; use `9.999999` for crashes
3. `val_pearson` — e.g. `0.721300`; use `0.000000` for crashes
4. `val_auc` — e.g. `0.883000`; use `0.000000` for crashes
5. `status` — `keep`, `discard`, or `crash`
6. `description` — short text, no tabs

Example:
```
commit	val_nrmse	val_pearson	val_auc	status	description
a1b2c3d	0.312450	0.721300	0.883000	keep	baseline
b2c3d4e	0.308100	0.729500	0.889000	keep	increase HIDDEN_CHANNELS 64→96
c3d4e5f	0.319200	0.710000	0.871000	discard	reduce NUM_LAYERS 2→1 (degraded)
d4e5f6g	9.999999	0.000000	0.000000	crash	USE_AMP=True on MPS (not supported)
```

## Experiment loop

Run on `autoresearch/<tag>` branch. LOOP FOREVER:

1. Check current git state (branch, last commit).
2. Edit `research/train.py` with one experimental idea.
3. `git add research/train.py && git commit -m "<description>"`
4. Run: `python research/train.py > research/run.log 2>&1`
5. Read results: `grep "^val_nrmse:\|^val_pearson:\|^val_auc:" research/run.log`
6. If grep is empty → crashed. Run `tail -50 research/run.log` and attempt a fix.
   Give up after 2-3 failed attempts; log as `crash`.
7. Log results to `research/results.tsv`.
8. If `val_nrmse` improved (lower) → **keep** the commit, advance the branch.
9. If equal or worse → `git reset --soft HEAD~1 && git restore research/train.py` → **discard**.

**Timeout**: if a run exceeds 10 minutes, kill it (`Ctrl-C`) and treat as failure.

**NEVER STOP**: once the loop begins, do NOT pause to ask the human whether to continue.
Keep going until manually interrupted.

## Ideas to try (starting points)

- Tune `HIDDEN_CHANNELS` (32 / 64 / 96 / 128)
- Tune `NUM_LAYERS` (1 / 2 / 3 / 4)
- Tune `LEARNING_RATE` (5e-4 / 1e-3 / 3e-3)
- Tune `DROPOUT` (0.1 / 0.2 / 0.3 / 0.4 / 0.5)
- Try `USE_ADAPTERS = False` (simpler model)
- Tune `REDUCTION_FACTOR` (2 / 4 / 8)
- Tune `ADT_LOSS_WEIGHT` vs `AML_LOSS_WEIGHT` (e.g. 2.0 / 0.3)
- Switch `USE_SPARSE_ATTN = False` for dense attention
- Adjust `NEIGHBORHOOD_SIZE` (10 / 20 / 30 / 50)
- Tune `WEIGHT_DECAY` (0 / 1e-5 / 1e-4 / 1e-3)
- Try cosine annealing with restarts instead of single cosine decay
- Try AdamW instead of Adam
- Tune `GRAD_CLIP` (0.5 / 1.0 / 2.0 / off)
- Try larger warmup ratio (10% → 20%)
- Try `NUM_HEADS = 2` vs `8` for both GAT and transformer

After exhausting individual changes, try combining best improvements found so far.
