"""Run the full AI Interaction ABM pipeline.

Usage
-----
From this directory, run:

    python run_simulation.py

The script performs three tasks:
1. Run the main four-scenario experiment with 20 replications per scenario.
2. Run one-at-a-time sensitivity analysis with 6 replications per parameter level.
3. Regenerate all CSV result files and all PNG/PDF figures.

No Jupyter notebooks are required. The complete experiment is scripted so that a
reviewer can recreate the results from the command line.
"""

from __future__ import annotations

import os
import pandas as pd
from experiments import run_experiments, run_sensitivity_analysis
from plots import (
    configure_matplotlib,
    add_confidence_intervals,
    plot_knowledge,
    plot_trust,
    plot_interaction_mix,
    plot_hiri_and_quality,
    plot_final_summary,
    plot_sensitivity,
)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(root, "results")
    figures_dir = os.path.join(root, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    n_replications = 20

    full, finals = run_experiments(output_dir=results_dir, n_replications=n_replications)
    sensitivity = run_sensitivity_analysis(output_dir=results_dir, n_replications=6)

    agg = (
        full.groupby(["scenario", "step"], as_index=False)
        .agg(
            mean_quality=("mean_quality", "mean"),
            sd_quality=("mean_quality", "std"),
            mean_knowledge=("mean_knowledge", "mean"),
            sd_knowledge=("mean_knowledge", "std"),
            mean_trust_ai=("mean_trust_ai", "mean"),
            sd_trust_ai=("mean_trust_ai", "std"),
            mean_trust_peer=("mean_trust_peer", "mean"),
            sd_trust_peer=("mean_trust_peer", "std"),
            HIRI=("HIRI", "mean"),
            sd_HIRI=("HIRI", "std"),
            AMR=("AMR", "mean"),
            sd_AMR=("AMR", "std"),
            avg_tie_strength=("avg_tie_strength", "mean"),
            sd_avg_tie_strength=("avg_tie_strength", "std"),
        )
    )
    for col in agg.columns:
        if col.startswith("sd_"):
            agg[col] = agg[col].fillna(0.0)
    agg = add_confidence_intervals(agg, n_replications=n_replications)

    agg.to_csv(os.path.join(results_dir, "aggregated_results.csv"), index=False)
    finals.groupby("scenario", as_index=False).mean(numeric_only=True).to_csv(
        os.path.join(results_dir, "summary_means.csv"), index=False
    )

    configure_matplotlib()
    plot_knowledge(agg, figures_dir)
    plot_trust(agg, figures_dir)
    plot_interaction_mix(full, figures_dir)
    plot_hiri_and_quality(agg, figures_dir)
    plot_final_summary(finals, figures_dir)
    plot_sensitivity(sensitivity, figures_dir)

    print("Simulation finished.")
    print(f"Results directory: {results_dir}")
    print(f"Figures directory: {figures_dir}")


if __name__ == "__main__":
    main()
