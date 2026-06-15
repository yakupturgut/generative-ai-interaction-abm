"""Plotting utilities for the AI-mediated interaction ABM.

The functions in this module regenerate the six figures reported in the paper
from the CSV outputs produced by run_simulation.py. Both PNG and PDF versions
are written to the figures directory.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 17,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def _save(fig, outpath_base):
    fig.savefig(outpath_base + ".png", dpi=450, bbox_inches="tight")
    fig.savefig(outpath_base + ".pdf", bbox_inches="tight")
    plt.close(fig)

def add_confidence_intervals(df: pd.DataFrame, n_replications: int) -> pd.DataFrame:
    out = df.copy()
    z = 1.96
    pairs = [
        ("mean_quality", "sd_quality"),
        ("mean_knowledge", "sd_knowledge"),
        ("mean_trust_ai", "sd_trust_ai"),
        ("mean_trust_peer", "sd_trust_peer"),
        ("HIRI", "sd_HIRI"),
        ("AMR", "sd_AMR"),
        ("avg_tie_strength", "sd_avg_tie_strength"),
    ]
    for mean_col, sd_col in pairs:
        out[f"{mean_col}_low"] = out[mean_col] - z * out[sd_col] / np.sqrt(n_replications)
        out[f"{mean_col}_high"] = out[mean_col] + z * out[sd_col] / np.sqrt(n_replications)
    return out

def plot_knowledge(agg_df: pd.DataFrame, figures_dir: str):
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    for scenario, sdf in agg_df.groupby("scenario"):
        ax.plot(sdf["step"], sdf["mean_knowledge"], label=scenario.replace("_", " ").title())
        ax.fill_between(sdf["step"], sdf["mean_knowledge_low"], sdf["mean_knowledge_high"], alpha=0.14)
    ax.set_title("Knowledge Growth Across Scenarios")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Mean Knowledge")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, title="Scenario")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    _save(fig, os.path.join(figures_dir, "figure_1_knowledge_growth"))

def plot_trust(agg_df: pd.DataFrame, figures_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), sharex=True)
    for scenario, sdf in agg_df.groupby("scenario"):
        label = scenario.replace("_", " ").title()
        axes[0].plot(sdf["step"], sdf["mean_trust_ai"], label=label)
        axes[0].fill_between(sdf["step"], sdf["mean_trust_ai_low"], sdf["mean_trust_ai_high"], alpha=0.10)
        axes[1].plot(sdf["step"], sdf["mean_trust_peer"], label=label)
        axes[1].fill_between(sdf["step"], sdf["mean_trust_peer_low"], sdf["mean_trust_peer_high"], alpha=0.10)
    axes[0].set_title("Trust in AI")
    axes[1].set_title("Trust in Peers")
    for ax in axes:
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Mean Trust")
        ax.set_ylim(0, 1.02)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.86, 0.5), frameon=True, title="Scenario")
    fig.suptitle("Trust Trajectories Under AI-Mediated Interaction", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.84, 0.97])
    _save(fig, os.path.join(figures_dir, "figure_2_trust_trajectories"))

def plot_interaction_mix(full_df: pd.DataFrame, figures_dir: str):
    grouped = (
        full_df.groupby(["scenario", "step"], as_index=False)
        .agg(peer_events=("peer_events", "mean"),
             ai_events=("ai_events", "mean"),
             hybrid_events=("hybrid_events", "mean"))
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, scenario in zip(axes, grouped["scenario"].drop_duplicates().tolist()):
        sdf = grouped[grouped["scenario"] == scenario]
        ax.plot(sdf["step"], sdf["peer_events"], label="Peer-first")
        ax.plot(sdf["step"], sdf["ai_events"], label="AI-first")
        ax.plot(sdf["step"], sdf["hybrid_events"], label="Hybrid")
        ax.set_title(scenario.replace("_", " ").title())
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Average Number of Events")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), frameon=True)
    fig.suptitle("Interaction Mode Composition by Scenario", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    _save(fig, os.path.join(figures_dir, "figure_3_interaction_mix"))

def plot_hiri_and_quality(agg_df: pd.DataFrame, figures_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.1), sharex=True)
    for scenario, sdf in agg_df.groupby("scenario"):
        label = scenario.replace("_", " ").title()
        axes[0].plot(sdf["step"], sdf["HIRI"], label=label)
        axes[0].fill_between(sdf["step"], sdf["HIRI_low"], sdf["HIRI_high"], alpha=0.10)
        axes[1].plot(sdf["step"], sdf["mean_quality"], label=label)
        axes[1].fill_between(sdf["step"], sdf["mean_quality_low"], sdf["mean_quality_high"], alpha=0.10)
    axes[0].set_title("Human Interaction Retention Index (HIRI)")
    axes[0].set_xlabel("Simulation Step")
    axes[0].set_ylabel("HIRI")
    axes[1].set_title("Mean Task Quality")
    axes[1].set_xlabel("Simulation Step")
    axes[1].set_ylabel("Mean Quality")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.86, 0.5), frameon=True, title="Scenario")
    fig.suptitle("Interaction Retention and Collective Performance", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.84, 0.97])
    _save(fig, os.path.join(figures_dir, "figure_4_hiri_quality"))

def plot_final_summary(finals_df: pd.DataFrame, figures_dir: str):
    summary = finals_df.groupby("scenario", as_index=False).mean(numeric_only=True)
    scenario_labels = [s.replace("_", " ").title() for s in summary["scenario"]]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    metrics = [
        ("final_mean_knowledge", "Final Mean Knowledge"),
        ("final_HIRI", "Final HIRI"),
        ("final_mean_trust_peer", "Final Mean Peer Trust"),
        ("final_gini_performance", "Performance Inequality (Gini)"),
    ]
    for ax, (col, title) in zip(axes.flatten(), metrics):
        ax.bar(scenario_labels, summary[col])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Final Outcome Comparison Across Scenarios", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, os.path.join(figures_dir, "figure_5_final_summary"))

def plot_sensitivity(sensitivity_df: pd.DataFrame, figures_dir: str):
    metrics = [
        ("final_mean_quality", "Final Mean Quality"),
        ("final_HIRI", "Final HIRI"),
        ("final_mean_trust_peer", "Final Mean Peer Trust"),
        ("final_gini_performance", "Performance Inequality (Gini)"),
    ]
    summary = sensitivity_df.groupby(["factor", "level", "value"], as_index=False).mean(numeric_only=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.6))
    axes = axes.flatten()
    factor_order = ["ai_reliability", "ai_convenience", "verification_strength", "network_density"]
    factor_titles = {
        "ai_reliability": "AI Reliability",
        "ai_convenience": "AI Convenience",
        "verification_strength": "Verification Strength",
        "network_density": "Network Density",
    }

    for ax, (metric_col, metric_title) in zip(axes, metrics):
        for factor in factor_order:
            sdf = summary[summary["factor"] == factor].sort_values("value")
            ax.plot(sdf["value"], sdf[metric_col], marker="o", label=factor_titles[factor])
        ax.set_title(metric_title)
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel(metric_title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.86, 0.5), frameon=True, title="Sensitivity Factor")
    fig.suptitle("One-at-a-Time Sensitivity Analysis", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.84, 0.97])
    _save(fig, os.path.join(figures_dir, "figure_6_sensitivity_analysis"))
