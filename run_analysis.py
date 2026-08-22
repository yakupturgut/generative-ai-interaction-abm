"""Run the complete AI–peer interaction agent-based modeling study.

Usage
-----
Full study (reproduces repository result design):
    python run_analysis.py

Small pipeline check:
    python run_analysis.py --quick

The experiment functions are checkpoint-aware. Re-running the script skips
replications already present in the output CSV files.
"""
from pathlib import Path
import argparse
import time

from model import ModelConfig
from experiments import (
    run_reference,
    run_task_mode_mechanism,
    run_environment_factorial,
    run_ai_shock_experiment,
    run_mechanism_sensitivity,
    save_config,
)
from figures import make_all_figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run a small diagnostic pipeline instead of the full study.')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation.')
    args = parser.parse_args()

    quick = bool(args.quick)
    root = Path(__file__).resolve().parent
    results = root / ('results_quick_test' if quick else 'results')
    figures_dir = root / ('figures_quick_test' if quick else 'figures')
    results.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    if quick:
        n_agents, episodes = 60, 36
        reference_reps, task_reps, factorial_reps, shock_reps, sensitivity_reps = 3, 3, 2, 3, 2
    else:
        n_agents, episodes = 180, 150
        reference_reps, task_reps, factorial_reps, shock_reps, sensitivity_reps = 30, 20, 20, 30, 20

    cfg = ModelConfig(
        n_agents=n_agents,
        episodes=episodes,
        ai_enabled=True,
        ai_reliability=0.78,
        ai_access_mean=0.60,
        ai_access_sd=0.16,
        verification_support=0.55,
        network_density=0.055,
        rewiring_probability=0.10,
        peer_response_probability=0.80,
        external_peer_search_probability=0.12,
        peer_expertise_visibility=0.30,
        social_observation_rate=0.45,
        task_mix=(1/3, 1/3, 1/3),
        seed=20260822,
    )

    run_settings = {
        'quick_test': quick,
        'n_agents': n_agents,
        'episodes': episodes,
        'reference_replications': reference_reps,
        'task_mechanism_replications': task_reps,
        'factorial_replications_per_condition': factorial_reps,
        'shock_replications_per_condition': shock_reps,
        'sensitivity_replications_per_setting': sensitivity_reps,
        'environment_factorial': '3 AI-access x 3 AI-reliability x 2 verification-support x 2 peer-response = 36 conditions',
        'confidence_intervals': '95% Student-t confidence intervals across independent replications',
    }
    save_config(cfg, results, run_settings)

    print('=' * 80)
    print('GENERATIVE AI AS AN INTERACTION INTERMEDIARY - ABM ANALYSIS')
    print('=' * 80)
    print(f"Mode: {'quick diagnostic' if quick else 'full study'}")
    print(f'Agents: {n_agents}; episodes: {episodes}')
    print(f'Results: {results}')
    print(f'Figures: {figures_dir}')

    start = time.time()
    run_reference(cfg, results, n_reps=reference_reps, tail=max(5, episodes // 6))
    run_task_mode_mechanism(cfg, results, n_reps=task_reps)
    run_environment_factorial(cfg, results, n_reps=factorial_reps, quick=quick)
    run_ai_shock_experiment(cfg, results, n_reps=shock_reps)
    run_mechanism_sensitivity(cfg, results, n_reps=sensitivity_reps, quick=quick)
    if not args.no_figures:
        make_all_figures(results, figures_dir)

    print(f'Completed in {(time.time()-start)/60:.2f} minutes.')


if __name__ == '__main__':
    main()
