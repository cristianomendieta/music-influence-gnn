# LESSONS - auto-maintained by scripts/lessons.py

> Machine-owned. Do NOT hand-edit. Changes are overwritten on the next `lessons.py` write.
> Canonical state lives in `.specs/lessons.json`. Edit lessons only via the script.
> promote_threshold=2 distinct features · window_days=45 · quarantine_threshold=2

## Confirmed (load these at Specify/Design)

Corroborated across multiple features. Safe to apply as guidance.

_none_

## Candidates (under observation - do NOT load as guidance yet)

Seen once or not yet corroborated. Tracked, not trusted.

### L-001 - When a spec fixes a numeric budget for a configuration, test the real configuration at production dimensions against that budget, not only the guard function on literal values
- signal: `ac_gap` · recurrence: 1 feature(s) · scope: `models` · harmful: 0
- features: phase-7-atencao
- evidence: ATT-02 AC6 / ATT-03 AC1 (validation.md Gap 1, probe 561537 vs 453121±10%) (models)
- last seen: 2026-10-18T15:40:23Z

### L-002 - When a function promises to normalise its input, test it with unnormalised input
- signal: `surviving_mutant` · recurrence: 1 feature(s) · scope: `evaluation` · harmful: 0
- features: phase-7-atencao
- evidence: L13 src/music_diffusion_gnn/evaluation/attention_stats.py:245 (evaluation)
- last seen: 2026-10-18T15:40:23Z

### L-003 - State in the AC exactly which object a count or budget refers to and which component enforces it
- signal: `spec_precision_gap` · recurrence: 1 feature(s) · scope: `specs` · harmful: 0
- features: phase-7-atencao
- evidence: ATT-02 AC6 (encoder vs whole-model parameter count) (specs)
- last seen: 2026-10-18T15:40:23Z

### L-004 - When a rule uses a strict inequality against a threshold, test the value exactly at the threshold
- signal: `surviving_mutant` · recurrence: 1 feature(s) · scope: `evaluation` · harmful: 0
- features: phase-7-atencao
- evidence: P2/P3 src/music_diffusion_gnn/evaluation/veredito_atencao.py:59,61 (evaluation)
- last seen: 2026-10-18T16:52:34Z

### L-005 - When a report text carries a derived number the spec states, assert that number, not only the inputs it is derived from
- signal: `surviving_mutant` · recurrence: 1 feature(s) · scope: `evaluation` · harmful: 0
- features: phase-7-atencao
- evidence: P1 src/music_diffusion_gnn/evaluation/veredito_atencao.py:181 (evaluation)
- last seen: 2026-10-18T16:52:40Z

### L-006 - When a function reports a count of what was present, test it on an input where something is missing, so the count differs from the total
- signal: `surviving_mutant` · recurrence: 1 feature(s) · scope: `evaluation` · harmful: 0
- features: phase-7-atencao
- evidence: N4 src/music_diffusion_gnn/evaluation/veredito_atencao.py:169 (evaluation)
- last seen: 2026-10-18T18:18:52Z

## Quarantined (failed when applied - ignore)

A confirmed lesson that recurred alongside failure. Kept for the maintainer to review.

_none_
