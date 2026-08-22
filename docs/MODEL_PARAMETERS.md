# Reference Model Parameters

The values below reproduce the reference configuration used for the reported analysis. The authoritative machine-readable copy is `results/analysis_configuration.json`, and the Python defaults are defined in `ModelConfig` in `model.py`.

## Scale and AI environment

| Parameter | Value | Meaning |
|---|---:|---|
| `n_agents` | 180 | Number of agents |
| `episodes` | 150 | Repeated task episodes per run |
| `ai_enabled` | `True` | AI support available in the reference system |
| `ai_reliability` | 0.78 | Scales realized AI capability |
| `ai_access_mean` | 0.60 | Mean individual AI access/ease |
| `ai_access_sd` | 0.16 | Heterogeneity in AI access |
| `verification_support` | 0.55 | Environmental support for checking/integrating AI output |
| `ai_availability_floor` | 0.70 | Lower bound of access-to-availability mapping |
| `ai_availability_ceiling` | 0.97 | Upper bound of access-to-availability mapping |
| `ai_access_saturation` | 2.0 | Curvature of access mapping |
| `ai_capability_profile` | (0.94, 0.64, 0.76) | AI capability across the three requirement dimensions |
| `ai_capability_noise_sd` | 0.085 | AI capability noise SD |
| `ai_output_concentration` | 22.0 | AI-output dispersion configuration value |
| `ai_ambiguity_penalty` | 0.16 | Ambiguity-related reduction in effective AI capability |

## Peer and network environment

| Parameter | Value | Meaning |
|---|---:|---|
| `network_density` | 0.055 | Initial connected small-world network density target |
| `rewiring_probability` | 0.10 | Watts-Strogatz rewiring probability |
| `peer_response_probability` | 0.80 | Probability an existing neighbor is reachable |
| `external_peer_search_probability` | 0.12 | Base probability of searching beyond current neighbors |
| `peer_expertise_visibility` | 0.30 | Initial visibility of partner expertise |
| `social_observation_rate` | 0.45 | Probability of observing a neighbor's recent result |
| `peer_communication_noise_sd` | 0.095 | Communication noise SD |
| `peer_ambiguity_penalty` | 0.10 | Ambiguity-related reduction in peer transfer efficiency |
| `peer_min_transfer_efficiency` | 0.28 | Minimum peer transfer efficiency |
| `peer_max_transfer_efficiency` | 0.80 | Maximum peer transfer efficiency |

## Tasks and heterogeneous agents

| Parameter | Value | Meaning |
|---|---:|---|
| `task_mix` | (1/3, 1/3, 1/3) | Structured / contextual / integrative task-family probabilities |
| `difficulty_mean` | 0.58 | Mean task difficulty |
| `difficulty_concentration` | 10.0 | Beta-distribution concentration for difficulty |
| `task_similarity_temperature` | 0.090 | Softness of task-to-anchor similarity weights |
| `knowledge_mean` | 0.52 | Mean initial general capability |
| `ai_literacy_mean` | 0.50 | Mean AI literacy |
| `sociability_mean` | 0.55 | Mean sociability |
| `verification_mean` | 0.52 | Mean verification tendency |
| `confidence_mean` | 0.50 | Mean confidence |
| `trait_sd` | 0.14 | SD of heterogeneous bounded traits |

Task prototypes used to define interpretable regions of the continuous requirement space are:

| Task family | Requirement prototype | Ambiguity center |
|---|---|---:|
| Structured | (0.66, 0.22, 0.12) | 0.24 |
| Contextual | (0.22, 0.63, 0.15) | 0.66 |
| Integrative | (0.34, 0.29, 0.37) | 0.48 |

## Common route-choice rule

| Parameter | Value | Meaning |
|---|---:|---|
| `expected_quality_weight` | 3.60 | Weight on learned task-specific route quality |
| `trust_weight` | 0.12 | Weight on source trust |
| `local_social_learning_weight` | 0.55 | Weight on local observational signal |
| `choice_temperature` | 0.31 | Softmax choice temperature |
| `exploration_floor` | 0.025 | Exploration mass per feasible route |
| `peer_base_cost` | 0.22 | Base peer coordination cost |
| `ai_base_cost` | 0.20 | Base AI access cost |
| `ai_ambiguity_verification_cost` | 0.14 | Extra checking burden under ambiguous AI use |
| `hybrid_extra_cost` | 0.12 | Hybrid integration/coordination cost |
| `weak_network_search_cost` | 0.10 | Cost of weak/external peer search |

## Learning and adaptation

| Parameter | Value | Meaning |
|---|---:|---|
| `mode_value_learning_rate` | 0.32 | Task-specific route expectation learning |
| `trust_learning_rate` | 0.12 | AI and peer trust learning |
| `local_memory_learning_rate` | 0.13 | Local observational-memory learning |
| `confidence_learning_rate` | 0.030 | Confidence adaptation |
| `practice_learning_rate` | 0.008 | Learning from practice |
| `knowledge_learning_rate` | 0.095 | Learning from novel support information |

## Network co-evolution

| Parameter | Value | Meaning |
|---|---:|---|
| `tie_reinforcement_rate` | 0.085 | Positive tie update from useful peer contribution |
| `tie_negative_update_rate` | 0.040 | Negative tie update from harmful peer contribution |
| `tie_decay` | 0.0020 | Multiplicative decay of unused ties per episode |
| `tie_removal_threshold` | 0.050 | Tie deletion threshold |
| `initial_tie_low` | 0.42 | Lower bound of initial active tie strength |
| `initial_tie_high` | 0.68 | Upper bound of initial active tie strength |
| `new_tie_initial_strength` | 0.34 | Initial strength of a newly formed tie |
| `new_tie_formation_threshold` | 0.010 | Minimum positive peer marginal contribution for new tie formation |

## Outcome mechanism

| Parameter | Value | Meaning |
|---|---:|---|
| `support_uptake_scale` | 0.72 | Scale for beneficial source uptake |
| `harmful_advice_uptake_scale` | 0.40 | Scale for uptake of harmful information |
| `hybrid_coordination_burden` | 0.055 | Hybrid coordination burden |
| `hybrid_conflict_scale` | 0.20 | Penalty for AI-peer disagreement |
| `quality_noise_sd` | 0.024 | Final task-quality noise SD |
| `seed` | 20260822 | Reference configuration seed stored with the result set |

## Experiment settings

| Experiment | Settings |
|---|---|
| Reference AI-enabled system | 30 replications |
| Human-only benchmark | 30 paired replications |
| Task-route mechanism analysis | 20 replications with task-level records |
| Environmental factorial | 36 conditions × 20 replications |
| AI access levels | 0.35, 0.60, 0.85 |
| AI reliability levels | 0.55, 0.74, 0.90 |
| Verification-support levels | 0.30, 0.70 |
| Peer-response levels | 0.60, 0.90 |
| Reliability-shock experiment | 3 conditions × 30 replications |
| Shock start / recovery | 0.86 → 0.48 at mid-horizon; recovery to 0.86 at three-quarters for temporary-shock condition |
| Mechanism sensitivity | 16 parameters at −20% / reference / +20%, 20 paired replications per setting |
| Confidence intervals | Two-sided 95% Student-t intervals across independent replications |

See `ODD_model_description.md` / `.pdf` for equations, scheduling, initialization, and submodel details.
