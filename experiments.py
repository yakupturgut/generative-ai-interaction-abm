"""Experiment definitions for the AI-mediated interaction ABM.

This file separates scenario design from the core model implementation. It is
intended to make the experiment reproducible: all main scenario parameters,
replication counts, seeds and output files can be inspected here.

Scenarios are not calibrated empirical cases. They are stylised regimes that
encode different ways generative AI can be embedded in collaboration:

1. baseline_human_only: AI is practically unavailable.
2. ai_as_substitute: AI is convenient and frequently substitutes peer exchange.
3. ai_as_complement: AI is useful but combined with verification and peer input.
4. overreliance_fragility: AI is very convenient and trusted despite lower reliability.
"""

from __future__ import annotations

import os
import pandas as pd
from model import InteractionABM

# Scenario parameters vary the accessibility/reliability of AI and the learning
# consequences of different interaction modes. These values are transparent
# modelling assumptions; sensitivity analysis varies the most important ones.
SCENARIOS = {
    "baseline_human_only": {
        "ai_reliability": 0.0,
        "ai_convenience": -2.0,
        "verification_strength": 0.60,
        "peer_learning_rate": 0.042,
        "ai_learning_rate": 0.0,
        "hybrid_learning_rate": 0.0,
        "trust_gain": 0.035,
        "trust_loss": 0.050,
    },
    "ai_as_substitute": {
        "ai_reliability": 0.80,
        "ai_convenience": 0.95,
        "verification_strength": 0.25,
        "peer_learning_rate": 0.038,
        "ai_learning_rate": 0.022,
        "hybrid_learning_rate": 0.028,
        "trust_gain": 0.038,
        "trust_loss": 0.042,
    },
    "ai_as_complement": {
        "ai_reliability": 0.82,
        "ai_convenience": 0.72,
        "verification_strength": 0.80,
        "peer_learning_rate": 0.042,
        "ai_learning_rate": 0.018,
        "hybrid_learning_rate": 0.040,
        "trust_gain": 0.035,
        "trust_loss": 0.050,
    },
    "overreliance_fragility": {
        "ai_reliability": 0.65,
        "ai_convenience": 1.10,
        "verification_strength": 0.12,
        "peer_learning_rate": 0.034,
        "ai_learning_rate": 0.024,
        "hybrid_learning_rate": 0.022,
        "trust_gain": 0.045,
        "trust_loss": 0.030,
    },
}

# The default population size is 120 agents. This is inside the broad 110--150
# range often associated with Dunbar-type social-network arguments, but each
# agent only interacts with a smaller local neighbourhood in the small-world graph.
BASE_SETTINGS = {
    "n_agents": 120,
    "steps": 60,
    "network_density": 0.06,
}


def summarize_final_outcomes(df: pd.DataFrame, scenario_name: str) -> dict:
    """Extract final-step outputs from one replication."""
    final = df.iloc[-1]
    return {
        "scenario": scenario_name,
        "final_mean_quality": float(final["mean_quality"]),
        "final_mean_knowledge": float(final["mean_knowledge"]),
        "final_mean_trust_ai": float(final["mean_trust_ai"]),
        "final_mean_trust_peer": float(final["mean_trust_peer"]),
        "final_HIRI": float(final["HIRI"]),
        "final_AMR": float(final["AMR"]),
        "final_avg_tie_strength": float(final["avg_tie_strength"]),
        "final_gini_performance": float(final["gini_performance"]),
    }


def run_experiments(output_dir: str, n_replications: int = 20, seed_start: int = 2026):
    """Run all main scenarios and save replication-level outputs."""
    os.makedirs(output_dir, exist_ok=True)
    replication_frames = []
    final_rows = []

    for scenario_idx, (scenario_name, params) in enumerate(SCENARIOS.items()):
        for rep in range(n_replications):
            seed = seed_start + scenario_idx * 1000 + rep
            model = InteractionABM(seed=seed, **BASE_SETTINGS, **params)
            model.run()
            df = pd.DataFrame(model.history)
            df["scenario"] = scenario_name
            df["replication"] = rep
            replication_frames.append(df)
            final_rows.append(summarize_final_outcomes(df, scenario_name))

    full = pd.concat(replication_frames, ignore_index=True)
    finals = pd.DataFrame(final_rows)
    full.to_csv(os.path.join(output_dir, "replication_histories.csv"), index=False)
    finals.to_csv(os.path.join(output_dir, "final_outcomes_by_replication.csv"), index=False)
    return full, finals


def run_sensitivity_analysis(output_dir: str, n_replications: int = 6, seed_start: int = 9026):
    """Run one-at-a-time sensitivity analysis around the complement scenario."""
    os.makedirs(output_dir, exist_ok=True)
    base = dict(SCENARIOS["ai_as_complement"])
    factors = {
        "ai_reliability": [0.65, 0.82, 0.92],
        "ai_convenience": [0.45, 0.72, 0.95],
        "verification_strength": [0.30, 0.80, 1.00],
        "network_density": [0.03, 0.06, 0.10],
    }
    labels = ["low", "base", "high"]
    rows = []

    factor_idx = 0
    for factor, values in factors.items():
        for level, value in zip(labels, values):
            for rep in range(n_replications):
                seed = seed_start + factor_idx * 100 + rep
                settings = dict(BASE_SETTINGS)
                params = dict(base)
                if factor == "network_density":
                    settings["network_density"] = value
                else:
                    params[factor] = value

                model = InteractionABM(seed=seed, **settings, **params)
                model.run()
                final = pd.DataFrame(model.history).iloc[-1]
                rows.append({
                    "factor": factor,
                    "level": level,
                    "value": value,
                    "replication": rep,
                    "final_mean_quality": float(final["mean_quality"]),
                    "final_HIRI": float(final["HIRI"]),
                    "final_mean_trust_peer": float(final["mean_trust_peer"]),
                    "final_gini_performance": float(final["gini_performance"]),
                })
            factor_idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "sensitivity_results.csv"), index=False)
    return df
