# Reproducibility

## Reference scale

- Agents: 180
- Episodes: 150
- Reference replications: 30
- Task-mechanism replications: 20
- Environmental factorial replications: 20 per condition
- Reliability-shock replications: 30 per condition
- Sensitivity replications: 20 per setting

## Full pipeline

```bash
python run_analysis.py
```

The pipeline runs:

1. AI-enabled reference and Human-only benchmark;
2. task-route mechanism analysis;
3. 36-condition environmental factorial;
4. AI reliability shock/recovery analysis;
5. ±20% mechanism sensitivity;
6. figure generation.

## Confidence intervals

The result summaries use two-sided 95% Student-t confidence intervals across independent replications.

## Seeds and paired comparisons

Replication seeds are deterministic. Paired reference comparisons, shock-recovery contrasts, and sensitivity effects use common replication indices/seeds so that paired differences are not dominated by unrelated random variation.

## Large event-level data

`task_mode_agent_events.csv` is not committed because it exceeds GitHub's normal per-file limit. Running the full task-mechanism experiment regenerates it. Compact aggregate and replication-level outputs are included in `results/`.

## Figure regeneration

```bash
python regenerate_figures.py
```

The script reads the saved result files and recreates the figures under `figures/`.
