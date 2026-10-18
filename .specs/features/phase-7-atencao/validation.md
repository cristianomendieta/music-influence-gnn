# Phase 7 — Arquitetura com atenção Validation (iteration 4, final; extra iteration approved by the user)

**Date**: 2026-10-18
**Spec**: `.specs/features/phase-7-atencao/spec.md`
**Pré-compromisso**: `docs/adr/0006-precompromisso-fase-atencao.md`, with both amendments of 2026-10-18 (100 épocas / `patience = 10`; `hidden = 112` for the attention encoder, 432.209 parameters)
**Diff range**: `aa25149..f1ddcdc` (feature). Fix commit verified in this iteration: `f1ddcdc` (test: `n_seeds` counts only the seeds that had the cell)
**Verifier**: independent sub-agent (author ≠ verifier). Read-only on the real tree; mutations ran only in a temporary `git worktree` under the scratchpad.

## Validation: PASS ✅

The only open gap of iteration 3 (surviving mutant N4, `n_seeds` in `sensitivity_by_seed`) is closed: N4 is now killed by `tests/test_veredito_atencao.py:248`. Three fresh mutations, one per layer (`attention_stats.py`, `veredito_atencao.py`, torch-side `edge_features.py`), are all killed. Every gate is green. What remains is the declared, accepted set: ATT-06 AC3 (stratum file name guessed), the notebooks were never executed, and one cosmetic nit in 15a.

---

## History: iterations 1 to 3

| Iter | Gap | Resolution | Status |
| ---- | --- | ---------- | ------ |
| 1 | Blocker: gatv2 `hidden = 128` = 561.537, outside [407.808, 498.433] | ADR-0006 amendment `hidden = 112` (432.209, −4,6%); `tests/test_phase7_budget.py`; 8 grid configs re-probed on Colab: all 432.209 | ✅ Resolved (iter 2) |
| 1 | Surviving mutant L13 (no renormalisation) | `tests/test_attention_stats.py:44-52` | ✅ Resolved (iter 2) |
| 1 | `patience = 10` never asserted | `tests/test_phase7_train.py:101-105` | ✅ Resolved (iter 2) |
| 1 | Spec-precision: encoder vs whole-model count | `spec.md:67` says **modelo inteiro** | ✅ Resolved (iter 2) |
| 2 | P2/P3: CI boundary at exactly 0 untested | `41ff727`: `tests/test_veredito_atencao.py:109-110` | ✅ Resolved (iter 3) |
| 2 | P1: percentage in the budget block not asserted | `41ff727`: `tests/test_veredito_atencao.py:287` (`:278` before `f1ddcdc`) | ✅ Resolved (iter 3) |
| 2 | Spec-precision: tie rule vacuous when the counts are equal | `d98182d`: `spec.md:144` lowest `val_mse` fallback, logged; consistent with 15a cell 18 (verifier simulation, iter 3) | ✅ Resolved (iter 3) |
| 3 | Surviving mutant N4: `n_seeds` only tested where it equals the total | `f1ddcdc`: `tests/test_veredito_atencao.py:243-249` | ✅ Resolved (iter 4, N4 killed at `:248`) |
| 1-4 | ATT-06 AC3: Phase 6b stratum file not specified (MAP-08 deferred) | 15e cell 19 declares the guessed name and schema; cell 20 skips with a message | ⚠️ Accepted, declared gap |

---

## Task Completion

All tasks T1-T14 of `tasks.md` are done (unchanged since iteration 3; `validate_tasks.py` 0 errors). T9-T12 and T14 are notebook tasks with `Tests: none`, as the coverage matrix declares.

---

## Spec-Anchored Acceptance Criteria

**Spot-checked against the current files in this iteration (7 ACs, different from the 8 of iteration 3):** ATT-01 AC3, ATT-01 AC4, ATT-01 AC7, ATT-04 AC4 (training checkpoint), ATT-05 AC1, ATT-05 AC4, ATT-05 AC5. They are marked 🔎 with the lines re-read. The other rows carry over from iteration 3, where they were re-read; the test files they cite have not changed since, except `tests/test_veredito_atencao.py`, which gained one 9-line test after line 241; lines from 242 on moved by 9, and every row below uses the current numbers.

### ATT-01 / ATT-02 — P1 Encoder de atenção causal

| Criterion | Spec-defined outcome | `file:line` + assertion | Result |
| --------- | -------------------- | ----------------------- | ------ |
| AC1 `encoder="gatv2"` offered, `sage` default unchanged | sage `state_dict` identical; gatv2 runs | `tests/test_phase7_model.py:137` `assert shapes == SAGE_STATE_SHAPES`; `:145` output in [0, 0,5]; `:193` unknown-encoder message | ✅ PASS |
| AC2 only `chart` and `log1p(w − first_seen_week)` | `(E, 2)` equal to a hand table | `tests/test_phase7_edge_features.py:56` `assert_close(feats, _EXPECTED_WEEK5, atol=1e-6)`; `tests/test_phase7_encoder.py:134` | ✅ PASS |
| AC3 perturbing `days_together` / `avg_position_distance` → identical embeddings 🔎 | bit-identical | `tests/test_phase7_encoder.py:124` `assert torch.equal(_encode(enc, g), _encode(enc, perturbed))`; `test_phase7_edge_features.py:66`; `test_phase7_model.py:155` | ✅ PASS |
| AC4 `strip_all_edges` graph = `HeteroSpatialEncoder` with the same root weights 🔎 | equal embeddings | `tests/test_phase7_encoder.py:159` `assert_close(z_att, z_sage, atol=1e-6, rtol=0)`, after copying summed SAGE `lin_r`/`lin_l.bias` into the roots and randomising the GATv2 biases (`:150-155`), with `:158` `z_sage.abs().sum() > 0` so the check is not vacuous | ✅ PASS |
| AC5 attention sums to 1 per destination with ≥1 in-edge | 1, tol `1e-5` | `test_phase7_encoder.py:183-190` (`index_add_` sums, `has_edge.any()` asserted at `:187`) | ✅ PASS |
| AC6 whole-model count; abort outside ±10% of 453.121 | `[407.808, 498.433]`; real config inside | `tests/test_phase7_budget.py:82`, `:88-89` (`== 432_209`), `:96-98`; `tests/test_phase7_train.py:110-120` boundaries | ✅ PASS |
| AC7 relation without edges omitted, no error 🔎 | root-only embedding | `tests/test_phase7_encoder.py:113` `assert_close(z, expected, atol=0, rtol=0)` with `expected = roots[0]["music"](x).relu()` and every GATv2 bias filled with 1 (`:109`), so a leaked bias would show; `:98` non-zero output for each single empty relation | ✅ PASS |
| Edge case: parallel edges stay separate | 2 entries, distinct weights | `test_phase7_encoder.py:201`, `:204` | ✅ PASS |

### ATT-03 / ATT-04 — P1 Treino e avaliação da matriz

| Criterion | Spec-defined outcome | Evidence | Result |
| --------- | -------------------- | -------- | ------ |
| AC1 tune 8 configs on `current`/seed 42, record `val_mse` + params before any test eval | 8 rows inside the budget | 15a cells 9, 14, 16, 18; iter-2 Colab probe: 8/8 = 432.209 | ✅ (notebook not executed) |
| AC2 chosen config unchanged after the test | hash check | 15a cell 18 sha256; 15b cell 7 recompute/abort | Read only |
| AC3 resume without retraining | skip existing | 15a cell 14, 15b cell 15, 15c cell 15, 15e cell 13 | Read only |
| AC4 checkpoint records split, seed, variante, config, graph version 🔎 | those fields; model rebuildable | `tests/test_phase7_train.py:140-145` dict of `encoder, heads, attn_dropout, variante, split_regime, seed, graph_seed, n_params` equal to the expected values; `:146` `graph_version == "hetero_full_pre_pandemia.pt"`; `:156` `torch.equal(y, y_ref)` after `torch.save`/`load` and `model_from_checkpoint` | ✅ PASS |
| AC5 Mode 2: k ∈ {1,2,4}, both charts, both cuts, both edge budgets | | 15c cells 7, 15, 17 | Read only |
| AC6 `max_cotraj_edges = 30_000`, `max_epochs = 100`, `patience = 10` | the three values | `tests/test_phase7_train.py:104-105` | ✅ PASS |
| Edge case: tie within 0,000003 → fewer params; equal counts → lowest `val_mse`, logged | rule applied and logged | `test_veredito_atencao.py:276` `TIE_VAL_MSE == 0.000003`; 15a cell 18 (simulated in iter 3) | ✅ Consistent (notebook not executed) |
| Edge cases: NaN retry; GPU memory batches | | 15b cell 15; 15c cell 15 | Read only |

### ATT-05 — P1 Veredito do ADR-0006

| Criterion | Spec-defined outcome | `file:line` + assertion | Result |
| --------- | -------------------- | ----------------------- | ------ |
| AC1 H1/H2/H3 per split and on-chart cell, CI 95% + Wilcoxon on per-song squared error 🔎 | values equal `paired_by_song` on squared errors; on-chart only | `tests/test_veredito_atencao.py:92` `row[key] == pytest.approx(ref[key])` for `mean_diff, ci_lo, ci_hi, p_value, win_rate`, with `ref` built from `(y_true − y_pred) ** 2` (`:87-89`; kills Q2); `:93` `n_songs`; `:129` `assert_frame_equal(t_on, t_all)` with far-worse off-chart rows added; `:133-136` missing model raises naming it | ✅ PASS |
| AC2 Holm over the 18 of each split | 18 rows; Holm of the 18 together | `:70`, `:75-76` | ✅ PASS |
| "Vence"/"perde" strict at 0 and at 0,05 | | `:103-110` | ✅ PASS |
| AC3 O1–O4, majority 4/6, O4 precedence, H3 inert | exact codes | `:165-175`, `:178-181`, `:185`, `:189`, `:193-195` | ✅ PASS |
| AC4 splits disagree → discordance reported, main claim restricted 🔎 | `discordancia`, `"restrita ao que se replica"` | `tests/test_veredito_atencao.py:214-218`: `desfecho` O1 vs O3, `out.discordancia.all()`, `(out.afirmacao_principal == "restrita ao que se replica").all()`, `"replica" in nota`; `:205-208` agreeing splits have no discordance | ✅ PASS |
| AC5 per-seed sensitivity, concordance with the mean 🔎 | 3/3, 2/3; seed outcome next to the mean's; `n_seeds` = seeds that had the cell | `:236-240` (3/3, `n_seeds == 3`), **new** `:248` `(cells[pre_pandemia].n_seeds == 2).all()` and `:249` `(cells[current].n_seeds == 3).all()` with seed 44 lacking `pre_pandemia` (kills N4); `:257-258` 2/3 and seed 44 `perde`; `:266-270` outcomes O1/O1/O1/O4, `seeds_concordam == 2` | ✅ PASS |
| AC6 tuning budget (8 vs 24), param counts, −4,6%, `hidden` limitation | | `:280-283`, `:287` `"-4.6%" in budget_block(432_209)`; `hidden` sentence in 15d cell 17 (read only) | ✅ PASS; `hidden` sentence notebook-only |

### ATT-06 — P2 Pesos de atenção

| Criterion | Spec-defined outcome | `file:line` + assertion | Result |
| --------- | -------------------- | ----------------------- | ------ |
| AC1 normalised entropy by in-degree | uniform 1, single 0, degree 1 → 0, degree 0 NaN, scale-invariant; bins | `tests/test_attention_stats.py:28`, `:34`, `:41`, `:51-52`, `:60-62`, `:106-108` (`n_nodes == [1, 2, 1]` for bins `(1, 2, 4)`; kills Q1), `:117-118` | ✅ PASS |
| AC2 legend on every figure | | 15e cells 7, 17 | Read only |
| AC3 cross with degree tercile if Phase 6b exists | skip otherwise | 15e cells 19-20 | ⚠️ Accepted, declared gap |

**Status**: ✅ All code-level ACs covered with spec-anchored assertions; notebook-only ACs read; one accepted declared gap (ATT-06 AC3).

---

## Discrimination Sensor (iteration 4)

Scratch: `git worktree add --detach <scratchpad>/wt4 HEAD` (at `f1ddcdc`). Each mutant is an exact single-string replacement (`<scratchpad>/mutate.py`, asserts one occurrence; specs in `<scratchpad>/mutants_iter4.json`), reverted with `git -C wt4 checkout -- src` and an empty worktree status checked after each revert. Pandas mutants: local venv, `PYTHONPATH=wt4/src pytest tests/test_veredito_atencao.py tests/test_attention_stats.py` from inside the worktree (import resolves to `wt4/src`, checked; 46 passed unmutated). Torch mutant: Colab session `tests` via `REPO=wt4 colab_pytest.sh tests/test_phase7_edge_features.py tests/test_phase7_encoder.py tests/test_phase7_model.py` (29 passed unmutated, `PYTEST_EXIT=0`). The session had expired and was recreated (`colab new -s tests` + `colab install`).

| # | File:line | Mutation | Killed by | Result |
| - | --------- | -------- | --------- | ------ |
| N4 (iter-3 survivor) | `evaluation/veredito_atencao.py:169` | `n_seeds = cells[label_cols].notna().sum(axis=1)` → `len(label_cols)` | `test_n_seeds_counts_only_the_seeds_that_had_the_cell` (`tests/test_veredito_atencao.py:248`: pre_pandemia cells report 3 instead of 2) | ✅ Killed |
| Q1 (fresh) | `evaluation/attention_stats.py:93` | degree-bin upper edges `b - 1` → `b` (adjacent bins overlap) | `test_entropy_by_degree_has_one_row_per_bin_and_counts_nodes_with_edges` (`tests/test_attention_stats.py:107`) | ✅ Killed |
| Q2 (fresh) | `evaluation/veredito_atencao.py:86` | squared error → absolute error in the paired comparison | `test_difference_matches_paired_by_song_called_directly` (`tests/test_veredito_atencao.py:92`) | ✅ Killed |
| Q3 (fresh, torch) | `graph/edge_features.py:33` | unmasked-snapshot guard `first_seen > week` → `>=` (edges first seen at the snapshot week rejected) | 18 tests, first `test_two_columns_match_hand_table` (`tests/test_phase7_edge_features.py:53`, week 5 with edges first seen at 5), also encoder and model tests; `PYTEST_EXIT=1` | ✅ Killed |

None of Q1-Q3 repeats an earlier mutant (checked against L1-L16, T1-T15, the iteration-2 set and N1-N4).

**Sensor depth**: lightweight plus the re-run (4 mutations: 1 re-run, 3 fresh over the three layers).
**Result**: 4/4 killed. Sensor PASS.
**Isolation**: real-tree `git status --porcelain` before (`<scratchpad>/porcelain_before4.txt`) = after (`porcelain_after4.txt`) = the 3 untracked verifier files (`.specs/LESSONS.md`, `.specs/features/phase-7-atencao/validation.md`, `.specs/lessons.json`). `git stash list` empty before and after. Worktree removed (`git worktree remove --force` + `prune`); `git worktree list` shows only the main tree.

Cumulative over iterations 1-4: every surviving mutant (L13, P1, P2, P3, N4) has been killed by a later fix.

---

## Code Quality

| Principle | Status |
| --------- | ------ |
| Minimum code (`f1ddcdc` adds one 7-line test, no source change) | ✅ |
| Surgical changes | ✅ |
| No scope creep | ✅ |
| Matches patterns (reuses `_seeds`, `compare_models`, `B`) | ✅ |
| Spec-anchored outcome check | ✅ |
| Per-layer Coverage Expectation met | ✅ |
| Every test maps to a spec requirement (new test → ATT-05 AC5) | ✅ |
| Documented guidelines followed: `CLAUDE.md`, ruff config in `pyproject.toml` | ✅ |

---

## Edge Cases

- [x] Parallel edges stay separate (`test_phase7_encoder.py:201-204`)
- [x] Relation without edges omitted (`test_phase7_encoder.py:98`, `:113`)
- [x] Tuning tie, including equal parameter counts (spec `:144`, 15a cell 18, iter-3 simulation)
- [x] Seed missing a split: `n_seeds` reports the seeds present (`test_veredito_atencao.py:248-249`)
- [ ] NaN retry and GPU memory batches: notebook-only (15b/15c cell 15), read, not executed

---

## Gate Check

- **Gate command (real tree, Colab CPU "tests")**: `colab_pytest.sh tests -q -m "not slow" -rfE` → **264 passed**, 3 failed, 4 errors, 1 skipped, 9 deselected (84 s). The FAILED/ERROR id set is **identical** to `<scratchpad>/baseline_failures.txt` (the 7 data-missing ids, pre-existing; `diff` empty).
- **Test count**: 263 (iter 3) → 264. Delta +1, the new `test_n_seeds_counts_only_the_seeds_that_had_the_cell`. No deletions.
- **Local pandas tests**: `tests/test_veredito_atencao.py tests/test_attention_stats.py` → 46 passed.
- **Ruff** `src/music_diffusion_gnn tests`: 60 errors, all in pre-existing files (`trainer.py` 7 and `diffusion_gnn.py` 1 as at `aa25149`, the rest in files the feature never touched). None in feature files; `veredito_atencao.py`, `attention_stats.py`, `test_veredito_atencao.py`: "All checks passed!".
- **Notebooks** `scripts/check_notebook.py <nb> --forbid`: 0 errors on 15a, 15b, 15c, 15d, 15e.
- **`validate_tasks.py phase-7-atencao`**: 0 errors, 5 warnings (T9-T12, T14 `Tests: none`, as in the matrix).
- **`validate_spec.py phase-7-atencao`**: 0 errors, 0 warnings.
- **Skipped**: 1 pre-existing skip outside the feature; 9 `slow` deselected by the gate command.

---

## Notebooks: not executed

The five notebooks (`notebooks/item15{a,b,c,d,e}_*.ipynb`) were **never run** by the implementer or by any verifier iteration: they need the real graph, a GPU and Drive. Their behaviour is verified only by the syntax gate, by reading, and by verbatim copies of the decisive cells run on synthetic data (15a cells 9/14 in iteration 2; 15a cell 18 in iteration 3). Declared limitation, accepted.

---

## Accepted items and nits

- **ATT-06 AC3** (accepted, declared): the Phase 6b stratum file is not specified (MAP-08 deferred); 15e cell 19 guesses the name and schema, cell 20 skips with a message.
- **Notebooks not executed** (accepted, declared, see above).
- **Nit (cosmetic, not a gap)**: the `"criterio"` string that 15a cell 18 writes into `config_escolhida.json` does not mention the `val_mse` fallback for equal parameter counts. The markdown of cell 17 and `spec.md:144` do, and the logic is right.
- No open spec-precision gap.

---

## Requirement Traceability Update

| Requirement | Previous Status | New Status |
| ----------- | --------------- | ---------- |
| ATT-01 | ✅ Verified | ✅ Verified |
| ATT-02 | ✅ Verified | ✅ Verified |
| ATT-03 | ✅ Unblocked | ✅ Verified at code level; notebook not executed |
| ATT-04 | ✅ | ✅ AC4 and AC6 verified; AC3/AC5 notebook-only |
| ATT-05 | ⚠️ Needs one minor test fix | ✅ Verified (N4 killed); 15d not executed |
| ATT-06 | ✅ AC1 verified | ✅ AC1 verified; AC2 notebook-only; AC3 accepted gap |
| ATT-07 | — | Out of scope (P3, depends on outcome) |

---

## Lessons

- L-001…L-005: resolved in earlier iterations.
- **L-006: resolved.** N4 is killed by `tests/test_veredito_atencao.py:248` (a seed lacks a split, so the count differs from the total, as the lesson says).
- No new lesson: this iteration has no new grounded failure (no survivor, no failed AC, no spec-precision gap).

---

## Summary

**Overall**: ✅ Ready (code level), with the declared accepted items

**Spec-anchored check**: all code-level ACs traced to assertions on the spec-defined outcome; 7 spot-checked against the current code this iteration (ATT-01 AC3/AC4/AC7, ATT-04 AC4, ATT-05 AC1/AC4/AC5). 0 open spec-precision gaps.
**Sensor**: 4 mutations (N4 re-run + 3 fresh across pandas, verdict and torch layers), 4 killed, 0 survived.
**Gate**: 264 passed; failure set identical to the 7-id data-missing baseline; ruff clean in feature files; notebooks pass the syntax gate; `validate_tasks` and `validate_spec` 0 errors.

**What works**: causal GATv2 encoder and its leakage gates, the parameter budget at `hidden = 112`, checkpoint metadata, the ADR-0006 verdict (Holm, O1-O4, discordance, per-seed sensitivity including partial seeds, budget text), attention entropy by degree.

**Next steps**: run notebooks 15a → 15e on Colab GPU with the real graph. Optionally add the `val_mse` fallback to the `"criterio"` string of 15a.

## validate_state

`python3 ~/.claude/skills/tlc-spec-driven/scripts/validate_state.py phase-7-atencao` → **exit 0**: "validate_state: 0 error(s) across [phase-7-atencao]".
