# Generative AI as an Interaction Intermediary

This repository contains the agent-based model, experiment design, result summaries, and publication figures for the study **“Generative AI as an Interaction Intermediary: Modeling How AI-Mediated Interaction Changes Collaboration, Trust, and Social Learning.”**

Repository URL used in the conference paper:

`https://github.com/yakupturgut/generative-ai-interaction-abm`

## Research question

The model examines how repeated problem-solving choices among **Peer-first**, **AI-first**, and **Hybrid** support affect:

- task quality and success,
- knowledge accumulation,
- trust in AI and peers,
- social-tie reinforcement and activity,
- longer-run patterns of AI use and human interaction.

System-level interaction patterns arise from heterogeneous agents, task requirements, AI capability and access, peer expertise, social ties, and accumulated experience under a common behavioral and update architecture.

## Main model design

Reference configuration:

- **180 agents**
- **150 repeated task episodes**
- connected Watts-Strogatz social network
- three-dimensional agent skill profiles
- structured, contextual, and integrative task requirement profiles
- common probabilistic choice equation for Peer-first / AI-first / Hybrid
- task-specific experience learning
- AI and partner-specific peer trust updates
- knowledge learning from practice and novel support information
- contribution-based tie reinforcement and slow unused-tie decay

The model uses a common route-choice architecture:

`utility = expected task quality + source trust + local social signal - route cost`

and converts feasible route utilities to probabilities with a softmax rule.

## Experiments

The full analysis consists of five blocks:

1. **Reference AI-enabled system vs. Human-only benchmark**  
   30 paired replications.

2. **Task-route mechanism analysis**  
   20 replications with detailed task-level records.

3. **Environmental factorial**  
   3 AI-access levels × 3 AI-reliability levels × 2 verification-support levels × 2 peer-response levels = **36 conditions**, with 20 replications per condition.

4. **AI reliability shock and recovery**  
   Stable reliability, temporary drop + recovery, and persistent drop; 30 replications per condition.

5. **Mechanism sensitivity**  
   16 behavioral/update parameters varied by ±20%, with 20 paired replications per setting.

All primary uncertainty intervals are two-sided **95% Student-t confidence intervals** across independent replications.

## Repository structure

```text
.
├── model.py                 # Agent, task, choice, outcome, learning and network mechanisms
├── experiments.py           # Reference, factorial, shock and sensitivity experiments
├── run_analysis.py          # Complete experiment pipeline
├── precheck.py              # Small model/mechanism diagnostic
├── figures.py               # Analysis/output figures
├── regenerate_figures.py    # Recreate figures from saved CSV results
├── requirements.txt
├── docs/
│   ├── ODD_model_description.md
│   ├── ODD_model_description.tex
│   ├── ODD_model_description.pdf
│   ├── MODEL_PARAMETERS.md
│   └── REPRODUCIBILITY.md
├── results/                 # Compact replication-level and summary CSV outputs
└── figures/                 # Publication/analysis figures in PNG/PDF/SVG
```

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Quick diagnostic

```bash
python precheck.py
python run_analysis.py --quick
```

The quick run checks the full pipeline on a small design. It is not intended for scientific interpretation.

## Full analysis

```bash
python run_analysis.py
```

The experiment functions are checkpoint-aware. If a long run is interrupted, rerunning the command reuses completed replications already present in the result CSV files.

## Recreate figures only

```bash
python regenerate_figures.py
```

## Main outcome definitions

- **Human interaction rate** = Peer-first share + Hybrid share.
- **AI use rate** = AI-first share + Hybrid share.

These measures overlap because Hybrid uses both AI and another person; they are not complements and do not sum to one.

## Large event-level file

The full task-level experiment generates `task_mode_agent_events.csv`, which is approximately 179 MB in the reference study and exceeds GitHub's standard per-file size limit. It is intentionally not included in the repository package. The file is regenerated automatically by the task-mechanism experiment when the full analysis is run.

All compact summaries and replication-level files required to inspect the reported findings are included under `results/`.

## Reproducibility notes

- Independent replications use deterministic seed construction.
- Paired comparisons reuse common replication seeds where appropriate.
- The Human-only benchmark removes AI availability while preserving the same task, peer, learning, and network mechanisms.
- Task labels never directly award a route-specific utility or performance bonus.
- AI-first use does not directly penalize social ties; unused ties decay because they receive fewer opportunities for reinforcement.

See `docs/ODD_model_description.md` and `docs/REPRODUCIBILITY.md` for details.
