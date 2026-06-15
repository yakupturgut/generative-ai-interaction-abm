# AI Interaction ABM v5

This repository contains a compact agent-based model (ABM) for exploring how generative AI may act as an interaction intermediary in collaborative task settings. The model compares peer-first, AI-first and hybrid interaction modes and records how these modes affect task quality, knowledge accumulation, trust, human-interaction retention and performance inequality.

## Epistemological position

The model is a **generative and exploratory social simulation**, not a calibrated predictive model of a particular empirical population. Variables such as knowledge, trust, AI literacy and sociability are operational constructs on normalised scales. In particular, the model does not claim that human knowledge is intrinsically scalar; `knowledge` represents task-relevant capability for the abstract collaborative tasks studied here.

The choice equations use mode-specific behavioural propensity scores. These are described as utility functions in the paper, but they should be interpreted as bounded-rational propensities rather than welfare utilities. Their purpose is to encode explicit directional assumptions and examine the aggregate consequences of those assumptions.

## What is the task?

Each simulation step represents one repeated **collaborative problem-solving episode**. The task is intentionally abstract because the model is designed to compare interaction regimes rather than model a specific workplace, school or platform. Task difficulty is drawn randomly at each step and affects all interaction modes.

## Files

- `model.py`: core ABM implementation and behavioural mechanisms.
- `experiments.py`: scenario definitions, base settings and sensitivity analysis.
- `run_simulation.py`: full command-line pipeline for regenerating results and figures.
- `plots.py`: figure-generation functions.
- `results/`: CSV outputs generated from the included run.
- `figures/`: PNG and PDF figures generated from the included run.
- `ODD_model_description.pdf`: full ODD-style model documentation for review and replication.
- `MODEL_DOCUMENTATION.md`: compact documentation of assumptions, variables and reproducibility steps.
- `requirements.txt`: required Python packages.

## Reproducing the experiment

From this directory, create an environment and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run_simulation.py
```

This will regenerate:

- `results/replication_histories.csv`
- `results/final_outcomes_by_replication.csv`
- `results/aggregated_results.csv`
- `results/sensitivity_results.csv`
- `results/summary_means.csv`
- all PNG and PDF figures in `figures/`

## Experimental settings

Main experiment:

- 120 agents
- 60 task episodes
- 4 scenarios
- 20 replications per scenario
- connected Watts-Strogatz small-world network
- base network density: 0.06
- fixed seed structure starting from 2026

Sensitivity analysis:

- one-at-a-time sensitivity around the AI-as-complement scenario
- varied factors: AI reliability, AI convenience, verification strength and network density
- 6 replications per factor level
- fixed seed structure starting from 9026

## Notes on network size

The default population size is 120 agents. This is within the broad 110--150 range often discussed in relation to Dunbar-type social network arguments. However, agents do not interact with all other agents; the small-world network gives each agent a much smaller local neighbourhood while preserving clustered interaction and some long-range reach.

## No notebooks required

The experiment is fully scripted and does not require Jupyter notebooks. This is intentional: reviewers can recreate the outputs using a single command-line script.
