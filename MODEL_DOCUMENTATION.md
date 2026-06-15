# Model documentation note

## Purpose

The model explores how generative AI can change collaboration, trust and social learning when it becomes an interaction intermediary. The goal is not prediction but mechanism exploration: what aggregate patterns emerge when agents repeatedly choose between peer-first, AI-first and hybrid interaction?

## Entities and state variables

The model contains human agents connected by a social network. Each agent has:

- `knowledge`: task-relevant capability, normalised to [0, 1]
- `confidence`: self-confidence, normalised to [0, 1]
- `trust_ai`: trust in AI-mediated interaction, normalised to [0, 1]
- `trust_peer`: trust in peer interaction, normalised to [0, 1]
- `ai_literacy`: ability to benefit from AI, normalised to [0, 1]
- `sociability`: orientation toward peer interaction, normalised to [0, 1]
- `verification`: tendency to check or combine information, normalised to [0, 1]
- `recent_success`: exponentially smoothed memory of recent task success
- `performance`: cumulative task quality

The network contains edge weights that represent tie strength and are reinforced or weakened through use.

## Scheduling

At each task episode:

1. Task difficulty is drawn uniformly from [0.30, 0.90].
2. Each agent chooses peer-first, AI-first or hybrid interaction using softmax over behavioural propensities.
3. Peer-first and hybrid agents select one neighbour.
4. Task quality is computed using the selected mode.
5. Knowledge, trust, recent success, tie weights and cumulative performance are updated.
6. Aggregate outputs are recorded.

## Behavioural choice

The paper uses the term utility, but in the code these should be understood as behavioural propensity scores. They are not estimated welfare utilities. The current weights are intentionally exposed in `UTILITY_WEIGHTS` in `model.py`.

## Task quality

The task is an abstract collaborative problem-solving episode. Quality depends on the selected interaction mode, task difficulty, own capability, peer capability where relevant, AI reliability, trust and verification. The current weights are exposed in `QUALITY_WEIGHTS` in `model.py`.

## Network

The default network is a connected Watts-Strogatz small-world network. This captures the idea that collaborative environments often have clustered local interaction with some bridging ties. The base setting is 120 agents and network density 0.06, giving an average degree of approximately 8.

## Scenarios

The four scenarios are defined in `experiments.py`:

1. `baseline_human_only`: AI is effectively unavailable.
2. `ai_as_substitute`: AI is convenient and frequently used instead of peer interaction.
3. `ai_as_complement`: AI is used with verification and peer interaction.
4. `overreliance_fragility`: AI is highly convenient and trusted despite lower reliability.

## Reproducibility

Run:

```bash
pip install -r requirements.txt
python run_simulation.py
```

The pipeline regenerates all CSV outputs and figures. Random seeds are fixed in `experiments.py`.

## Limitations

- The model is not empirically calibrated.
- Knowledge, trust and sociability are operational constructs, not direct psychological measurements.
- The task is abstract and should be interpreted as a repeated collaborative problem-solving episode.
- The network is stylised; other network forms can be explored in future work.
- Utility/propensity weights encode transparent theoretical assumptions and should be tested in future empirical work.
