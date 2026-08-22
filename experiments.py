from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from model import AIPeerInteractionModel, ModelConfig


FINAL_METRICS = [
    "final_peer_share",
    "final_ai_share",
    "final_hybrid_share",
    "final_human_interaction_rate",
    "final_ai_use_rate",
    "final_mean_quality",
    "final_success_rate",
    "final_mean_knowledge",
    "final_knowledge_gini",
    "final_knowledge_rank_corr_with_initial",
    "final_performance_gini",
    "final_mean_ai_trust",
    "final_mean_peer_trust",
    "final_mean_tie_strength",
    "final_network_density",
    "final_mean_weighted_degree",
    "final_recent_active_tie_fraction",
    "final_mean_unique_peer_partners",
    "final_mean_peer_marginal",
    "final_mean_ai_availability_probability",
    "edge_retention_ratio",
    "edge_count_ratio",
    "new_edge_ratio",
    "strong_tie_fraction",
]


def ci95(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(x.mean())
    if n == 1:
        return mean, np.nan, np.nan, 1
    se = float(x.std(ddof=1) / np.sqrt(n))
    crit = float(student_t.ppf(0.975, df=n - 1))
    half = crit * se
    return mean, mean - half, mean + half, n


def summarize_with_ci(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), df)]
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = {col: val for col, val in zip(group_cols, key)}
        for metric in metric_cols:
            if metric not in g:
                continue
            mean, lo, hi, n = ci95(g[metric].to_numpy())
            rows.append({**base, "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "n": n})
    return pd.DataFrame(rows)


def _append_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def _completed_keys(path: Path, cols: List[str]) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=cols)
    return set(tuple(row) for row in df[cols].itertuples(index=False, name=None))


def run_model(config: ModelConfig, tail=20, keep_history=False, keep_task_details=False, keep_agent_summary=False):
    cfg = replace(config, record_task_details=bool(keep_task_details))
    model = AIPeerInteractionModel(cfg)
    model.run()
    model.validate_state()
    summary = model.final_summary(tail=tail)
    history = pd.DataFrame(model.history) if keep_history else None
    details = pd.DataFrame(model.task_mode_records) if keep_task_details else None
    agents = pd.DataFrame(model.agent_summary()) if keep_agent_summary else None
    return summary, history, details, agents


def _base_seed(rep: int) -> int:
    # Same replication seed is reused across experimental conditions where paired
    # comparisons are meaningful (common random numbers).
    return 20260821 + 1009 * int(rep)


def run_reference(
    base_cfg: ModelConfig,
    results_dir: Path,
    n_reps: int = 30,
    tail: int = 20,
):
    print("\n[1/5] Reference AI-enabled system + human-only benchmark")
    final_path = results_dir / "reference_replications.csv"
    hist_path = results_dir / "reference_histories.csv"
    completed = _completed_keys(final_path, ["system", "replication"])

    for system in ("AI-enabled", "Human-only benchmark"):
        for rep in range(n_reps):
            if (system, rep) in completed:
                continue
            cfg = replace(
                base_cfg,
                ai_enabled=(system == "AI-enabled"),
                seed=_base_seed(rep),
            )
            t0 = time.time()
            summary, history, _, _ = run_model(cfg, tail=tail, keep_history=True)
            row = {
                "system": system,
                "replication": rep,
                "seed": cfg.seed,
                **summary,
                "runtime_seconds": time.time() - t0,
            }
            _append_csv(final_path, [row])
            h = history.copy()
            h.insert(0, "seed", cfg.seed)
            h.insert(0, "replication", rep)
            h.insert(0, "system", system)
            _append_csv(hist_path, h.to_dict("records"))
            print(f"  {system:22s} rep {rep+1:02d}/{n_reps} done ({row['runtime_seconds']:.2f}s)")

    final = pd.read_csv(final_path)
    hist = pd.read_csv(hist_path)

    summary = summarize_with_ci(final, ["system"], FINAL_METRICS)
    summary.to_csv(results_dir / "reference_summary_95ci.csv", index=False)

    hist_metrics = [
        "peer_share",
        "ai_share",
        "hybrid_share",
        "human_interaction_rate",
        "ai_use_rate",
        "mean_quality",
        "success_rate",
        "mean_knowledge",
        "knowledge_rank_corr_with_initial",
        "mean_ai_trust",
        "mean_peer_trust",
        "mean_tie_strength",
        "network_density",
        "mean_weighted_degree",
        "recent_active_tie_fraction",
        "mean_unique_peer_partners",
        "mean_peer_marginal",
        "ai_reliability",
    ]
    hist_summary = summarize_with_ci(hist, ["system", "episode"], hist_metrics)
    hist_summary.to_csv(results_dir / "reference_history_summary_95ci.csv", index=False)

    # Paired AI-enabled minus human-only differences using common seeds.
    paired_rows = []
    a = final[final.system == "AI-enabled"].set_index("replication")
    h = final[final.system == "Human-only benchmark"].set_index("replication")
    common = a.index.intersection(h.index)
    for metric in FINAL_METRICS:
        diff = a.loc[common, metric].to_numpy() - h.loc[common, metric].to_numpy()
        mean, lo, hi, n = ci95(diff)
        paired_rows.append({"metric": metric, "mean_difference": mean, "ci_low": lo, "ci_high": hi, "n": n})
    pd.DataFrame(paired_rows).to_csv(results_dir / "reference_paired_differences_95ci.csv", index=False)

    # Monte Carlo precision diagnostic: CI half-width as replication count grows.
    conv_rows = []
    for system in ("AI-enabled", "Human-only benchmark"):
        d = final[final.system == system].sort_values("replication")
        candidate_ns = [5, 10, 15, 20, 25, 30]
        if len(d) < 5:
            candidate_ns = list(range(2, len(d) + 1))
        for n in candidate_ns:
            if len(d) < n:
                continue
            for metric in [
                "final_human_interaction_rate",
                "final_mean_quality",
                "final_mean_knowledge",
                "final_mean_peer_trust",
                "final_mean_tie_strength",
                "final_recent_active_tie_fraction",
            ]:
                mean, lo, hi, _ = ci95(d.iloc[:n][metric])
                conv_rows.append(
                    {
                        "system": system,
                        "n_replications": n,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "ci_half_width": (hi - lo) / 2 if np.isfinite(lo) else np.nan,
                    }
                )
    pd.DataFrame(conv_rows).to_csv(results_dir / "replication_convergence.csv", index=False)


def run_task_mode_mechanism(
    base_cfg: ModelConfig,
    results_dir: Path,
    n_reps: int = 20,
):
    print("\n[2/5] Task-type × chosen-mode mechanism check")
    out_path = results_dir / "task_mode_agent_events.csv"
    agent_path = results_dir / "agent_behavior_replications.csv"
    done_path = results_dir / "task_mode_completed.csv"
    completed = _completed_keys(done_path, ["replication"])

    for rep in range(n_reps):
        if (rep,) in completed:
            continue
        cfg = replace(base_cfg, ai_enabled=True, seed=_base_seed(rep))
        t0 = time.time()
        _, _, details, agents = run_model(cfg, tail=20, keep_task_details=True, keep_agent_summary=True)
        details.insert(0, "seed", cfg.seed)
        details.insert(0, "replication", rep)
        _append_csv(out_path, details.to_dict("records"))
        agents.insert(0, "seed", cfg.seed)
        agents.insert(0, "replication", rep)
        _append_csv(agent_path, agents.to_dict("records"))
        _append_csv(done_path, [{"replication": rep, "seed": cfg.seed, "runtime_seconds": time.time() - t0}])
        print(f"  task-mode rep {rep+1:02d}/{n_reps} done")

    d = pd.read_csv(out_path)
    d["support_gain"] = d["quality"] - d["solo_quality"]
    d["dominant_requirement"] = np.select(
        [
            (d["req_dim1"] >= d["req_dim2"]) & (d["req_dim1"] >= d["req_dim3"]),
            (d["req_dim2"] >= d["req_dim1"]) & (d["req_dim2"] >= d["req_dim3"]),
        ],
        ["Requirement 1 dominant", "Requirement 2 dominant"],
        default="Requirement 3 dominant",
    )
    # Replication-level means first, then CI across independent replications.
    rep_means = (
        d.groupby(["replication", "task_type", "mode"], as_index=False)
        .agg(
            mean_quality=("quality", "mean"),
            mean_support_gain=("support_gain", "mean"),
            success_rate=("success", "mean"),
            mean_difficulty=("difficulty", "mean"),
            mean_solo_quality=("solo_quality", "mean"),
            n_events=("quality", "size"),
        )
    )
    rep_means.to_csv(results_dir / "task_mode_replication_means.csv", index=False)
    smry = summarize_with_ci(rep_means, ["task_type", "mode"], ["mean_quality", "mean_support_gain", "success_rate", "n_events"])
    smry.to_csv(results_dir / "task_mode_summary_95ci.csv", index=False)

    # Choice shares by task type are often more informative than raw outcome means.
    counts = d.groupby(["replication", "task_type", "mode"]).size().rename("n").reset_index()
    total = counts.groupby(["replication", "task_type"])["n"].transform("sum")
    counts["share"] = counts["n"] / total
    choice_summary = summarize_with_ci(counts, ["task_type", "mode"], ["share"])
    choice_summary.to_csv(results_dir / "task_mode_choice_shares_95ci.csv", index=False)

    # Learning diagnostic: do task-specific choices separate as experience accumulates?
    # Compare the first and last third of the run within each independent replication.
    max_ep = int(d["episode"].max())
    early_end = max(1, int(round(max_ep / 3)))
    late_start = max(early_end + 1, int(round(2 * max_ep / 3)) + 1)
    phase = np.where(d["episode"] <= early_end, "Early", np.where(d["episode"] >= late_start, "Late", "Middle"))
    dl = d.assign(phase=phase)
    dl = dl[dl.phase.isin(["Early", "Late"])]
    plc = dl.groupby(["replication", "phase", "task_type", "mode"]).size().rename("n").reset_index()
    plc["share"] = plc["n"] / plc.groupby(["replication", "phase", "task_type"])["n"].transform("sum")
    summarize_with_ci(plc, ["phase", "task_type", "mode"], ["share"]).to_csv(
        results_dir / "task_mode_choice_early_late_95ci.csv", index=False
    )

    # A compact adaptation index: mean absolute difference in the three-mode choice
    # vector between task contexts. Zero means task-invariant choice; larger values
    # indicate that learned choices have become context-sensitive.
    adapt_rows = []
    for (rep, ph), gph in plc.groupby(["replication", "phase"]):
        piv = gph.pivot_table(index="task_type", columns="mode", values="share", fill_value=0.0)
        tasks = list(piv.index)
        diffs = []
        for a in range(len(tasks)):
            for b in range(a + 1, len(tasks)):
                va = piv.loc[tasks[a]].reindex(["Peer-first", "AI-first", "Hybrid"], fill_value=0).to_numpy(float)
                vb = piv.loc[tasks[b]].reindex(["Peer-first", "AI-first", "Hybrid"], fill_value=0).to_numpy(float)
                diffs.append(0.5 * np.abs(va - vb).sum())
        adapt_rows.append({"replication": rep, "phase": ph, "task_choice_divergence": float(np.mean(diffs)) if diffs else np.nan})
    adapt_df = pd.DataFrame(adapt_rows)
    adapt_df.to_csv(results_dir / "task_choice_adaptation_by_replication.csv", index=False)
    summarize_with_ci(adapt_df, ["phase"], ["task_choice_divergence"]).to_csv(
        results_dir / "task_choice_adaptation_95ci.csv", index=False
    )

    # A second, non-categorical view: which support modes are chosen and how much
    # value they add when different requirement dimensions dominate. This guards
    # against results being an artefact of the three presentation labels.
    req_rep = (
        d.groupby(["replication", "dominant_requirement", "mode"], as_index=False)
        .agg(
            mean_quality=("quality", "mean"),
            mean_support_gain=("support_gain", "mean"),
            n_events=("quality", "size"),
        )
    )
    req_counts = d.groupby(["replication", "dominant_requirement", "mode"]).size().rename("n").reset_index()
    req_counts["share"] = req_counts["n"] / req_counts.groupby(["replication", "dominant_requirement"])["n"].transform("sum")
    summarize_with_ci(req_rep, ["dominant_requirement", "mode"], ["mean_quality", "mean_support_gain"]).to_csv(
        results_dir / "requirement_mode_outcomes_95ci.csv", index=False
    )
    summarize_with_ci(req_counts, ["dominant_requirement", "mode"], ["share"]).to_csv(
        results_dir / "requirement_mode_choice_shares_95ci.csv", index=False
    )

    # Feasible-choice diagnostic: when AI is actually available, how do choices
    # differ across task contexts? This separates preference learning from simple
    # source unavailability.
    da = d[d["ai_available"] == 1].copy()
    ac = da.groupby(["replication", "task_type", "mode"]).size().rename("n").reset_index()
    ac["share"] = ac["n"] / ac.groupby(["replication", "task_type"])["n"].transform("sum")
    summarize_with_ci(ac, ["task_type", "mode"], ["share"]).to_csv(
        results_dir / "task_mode_choice_given_ai_available_95ci.csv", index=False
    )

    # Agent attributes: compute within-replication correlations first, then CI across
    # replications so agents from one run do not masquerade as independent runs.
    agents_all = pd.read_csv(agent_path)
    trait_cols = ["initial_knowledge", "ai_literacy", "sociability", "verification", "initial_confidence", "ai_access"]
    outcome_cols = ["peer_share", "ai_share", "hybrid_share", "mean_quality", "final_knowledge", "final_mean_peer_trust", "final_weighted_degree", "knowledge_gain"]
    corr_rows = []
    for rep, g in agents_all.groupby("replication"):
        for trait in trait_cols:
            for outcome in outcome_cols:
                x = g[trait].to_numpy(dtype=float)
                y = g[outcome].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 5 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
                    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
                else:
                    r = np.nan
                corr_rows.append({"replication": rep, "trait": trait, "outcome": outcome, "pearson_r": r})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(results_dir / "agent_trait_behavior_correlations_by_replication.csv", index=False)
    summarize_with_ci(corr_df, ["trait", "outcome"], ["pearson_r"]).to_csv(
        results_dir / "agent_trait_behavior_associations_95ci.csv", index=False
    )

    # Direct micro-feedback diagnostic: after a route performs better or worse than
    # expected, does the agent change the probability of using that same route the
    # next time it faces the same task family? This does not alter the simulation;
    # it only analyses the recorded decision sequence.
    tr = d.sort_values(["replication", "agent", "task_type", "episode"]).copy()
    grp = tr.groupby(["replication", "agent", "task_type"], sort=False)
    tr["next_mode"] = grp["mode"].shift(-1)
    tr["next_p_peer"] = grp["p_peer"].shift(-1)
    tr["next_p_ai"] = grp["p_ai"].shift(-1)
    tr["next_p_hybrid"] = grp["p_hybrid"].shift(-1)
    tr["next_probability_same_route"] = np.select(
        [tr["mode"] == "Peer-first", tr["mode"] == "AI-first", tr["mode"] == "Hybrid"],
        [tr["next_p_peer"], tr["next_p_ai"], tr["next_p_hybrid"]],
        default=np.nan,
    )
    tr["repeat_same_route"] = (tr["next_mode"] == tr["mode"]).astype(float)
    # A small deadband avoids classifying numerical noise as meaningful surprise.
    tr["experience_signal"] = np.where(
        tr["prediction_error"] >= 0.015, "Better than expected",
        np.where(tr["prediction_error"] <= -0.015, "Worse than expected", "Near expectation")
    )
    tr = tr[np.isfinite(tr["next_probability_same_route"])].copy()
    tr_rep = tr.groupby(["replication", "mode", "experience_signal"], as_index=False).agg(
        next_probability_same_route=("next_probability_same_route", "mean"),
        repeat_same_route=("repeat_same_route", "mean"),
        n_events=("repeat_same_route", "size"),
    )
    tr_rep.to_csv(results_dir / "experience_reinforcement_by_replication.csv", index=False)
    summarize_with_ci(
        tr_rep, ["mode", "experience_signal"],
        ["next_probability_same_route", "repeat_same_route", "n_events"]
    ).to_csv(results_dir / "experience_reinforcement_95ci.csv", index=False)


def run_environment_factorial(
    base_cfg: ModelConfig,
    results_dir: Path,
    n_reps: int = 20,
    quick: bool = False,
):
    print("\n[3/5] Environmental factorial: when does AI crowd out peer interaction?")
    out_path = results_dir / "environment_factorial_replications.csv"
    key_cols = ["ai_access_mean", "ai_reliability", "verification_support", "peer_response_probability", "replication"]
    completed = _completed_keys(out_path, key_cols)

    if quick:
        access_values = [0.45, 0.75]
        reliability_values = [0.60, 0.82]
        verification_values = [0.35]
        peer_response_values = [0.75]
    else:
        access_values = [0.35, 0.60, 0.85]
        reliability_values = [0.55, 0.74, 0.90]
        verification_values = [0.30, 0.70]
        peer_response_values = [0.60, 0.90]

    total_conditions = len(access_values) * len(reliability_values) * len(verification_values) * len(peer_response_values)
    condition_idx = 0
    for access in access_values:
        for reliability in reliability_values:
            for verification in verification_values:
                for peer_response in peer_response_values:
                    condition_idx += 1
                    label = f"A{access:.2f}_R{reliability:.2f}_V{verification:.2f}_P{peer_response:.2f}"
                    for rep in range(n_reps):
                        key = (access, reliability, verification, peer_response, rep)
                        # CSV round-trip floats can have tiny differences; use a rounded-key check.
                        rounded_completed = {
                            (round(float(a), 4), round(float(r), 4), round(float(v), 4), round(float(p), 4), int(q))
                            for a, r, v, p, q in completed
                        }
                        rkey = (round(access, 4), round(reliability, 4), round(verification, 4), round(peer_response, 4), rep)
                        if rkey in rounded_completed:
                            continue
                        cfg = replace(
                            base_cfg,
                            ai_enabled=True,
                            ai_access_mean=access,
                            ai_reliability=reliability,
                            verification_support=verification,
                            peer_response_probability=peer_response,
                            seed=_base_seed(rep),
                        )
                        t0 = time.time()
                        summary, _, _, _ = run_model(cfg, tail=max(20, base_cfg.episodes // 6))
                        row = {
                            "condition": label,
                            "ai_access_mean": access,
                            "ai_reliability": reliability,
                            "verification_support": verification,
                            "peer_response_probability": peer_response,
                            "replication": rep,
                            "seed": cfg.seed,
                            **summary,
                            "runtime_seconds": time.time() - t0,
                        }
                        _append_csv(out_path, [row])
                    print(f"  condition {condition_idx:02d}/{total_conditions}: {label}")

    df = pd.read_csv(out_path)
    group_cols = ["condition", "ai_access_mean", "ai_reliability", "verification_support", "peer_response_probability"]
    summary = summarize_with_ci(df, group_cols, FINAL_METRICS)
    summary.to_csv(results_dir / "environment_factorial_summary_95ci.csv", index=False)

    # Simple outcome-level associations are descriptive; causal interpretation comes
    # from the factorial manipulations, not these correlations.
    corr_rows = []
    for x in ["ai_access_mean", "ai_reliability", "verification_support", "peer_response_probability"]:
        for y in [
            "final_human_interaction_rate",
            "final_ai_use_rate",
            "final_mean_quality",
            "final_mean_knowledge",
            "final_mean_peer_trust",
            "final_mean_tie_strength",
            "final_recent_active_tie_fraction",
            "final_mean_unique_peer_partners",
            "edge_retention_ratio",
        ]:
            corr_rows.append({"factor": x, "outcome": y, "pearson_r": df[[x, y]].corr().iloc[0, 1]})
    pd.DataFrame(corr_rows).to_csv(results_dir / "environment_factorial_correlations.csv", index=False)


def run_ai_shock_experiment(
    base_cfg: ModelConfig,
    results_dir: Path,
    n_reps: int = 30,
):
    print("\n[4/5] AI reliability shock and social recovery")
    final_path = results_dir / "shock_replications.csv"
    hist_path = results_dir / "shock_histories.csv"
    completed = _completed_keys(final_path, ["shock_condition", "replication"])

    shock_start = int(round(0.50 * base_cfg.episodes)) + 1
    recovery_start = int(round(0.75 * base_cfg.episodes)) + 1
    conditions = {
        "Stable reliability": (0.86, None),
        "Temporary drop + recovery": (0.86, {shock_start: 0.48, recovery_start: 0.86}),
        "Persistent drop": (0.86, {shock_start: 0.48}),
    }
    for label, (start_rel, schedule) in conditions.items():
        for rep in range(n_reps):
            if (label, rep) in completed:
                continue
            cfg = replace(
                base_cfg,
                ai_enabled=True,
                ai_reliability=start_rel,
                reliability_schedule=schedule,
                ai_access_mean=0.70,
                seed=_base_seed(rep),
            )
            t0 = time.time()
            summary, hist, _, _ = run_model(cfg, tail=max(20, base_cfg.episodes // 6), keep_history=True)
            row = {
                "shock_condition": label,
                "replication": rep,
                "seed": cfg.seed,
                **summary,
                "runtime_seconds": time.time() - t0,
            }
            _append_csv(final_path, [row])
            hist.insert(0, "seed", cfg.seed)
            hist.insert(0, "replication", rep)
            hist.insert(0, "shock_condition", label)
            _append_csv(hist_path, hist.to_dict("records"))
            print(f"  {label:27s} rep {rep+1:02d}/{n_reps} done")

    final = pd.read_csv(final_path)
    hist = pd.read_csv(hist_path)
    summarize_with_ci(final, ["shock_condition"], FINAL_METRICS).to_csv(
        results_dir / "shock_final_summary_95ci.csv", index=False
    )
    hist_metrics = [
        "human_interaction_rate",
        "ai_use_rate",
        "mean_quality",
        "mean_knowledge",
        "knowledge_rank_corr_with_initial",
        "mean_ai_trust",
        "mean_peer_trust",
        "mean_tie_strength",
        "recent_active_tie_fraction",
        "mean_unique_peer_partners",
        "network_density",
        "ai_reliability",
    ]
    summarize_with_ci(hist, ["shock_condition", "episode"], hist_metrics).to_csv(
        results_dir / "shock_history_summary_95ci.csv", index=False
    )

    # Recovery / hysteresis diagnostics: compare late post-recovery window to the
    # corresponding stable-reliability condition using paired seeds.
    late_start = max(recovery_start + 1, int(round(0.88 * base_cfg.episodes)))
    late = hist[hist.episode >= late_start].groupby(["shock_condition", "replication"], as_index=False).mean(numeric_only=True)
    stable = late[late.shock_condition == "Stable reliability"].set_index("replication")
    temp = late[late.shock_condition == "Temporary drop + recovery"].set_index("replication")
    common = stable.index.intersection(temp.index)
    rows = []
    for metric in ["human_interaction_rate", "ai_use_rate", "mean_peer_trust", "mean_tie_strength", "recent_active_tie_fraction", "mean_unique_peer_partners", "mean_knowledge", "mean_quality"]:
        diff = temp.loc[common, metric].to_numpy() - stable.loc[common, metric].to_numpy()
        mean, lo, hi, n = ci95(diff)
        rows.append({"metric": metric, "post_recovery_difference_vs_stable": mean, "ci_low": lo, "ci_high": hi, "n": n})
    pd.DataFrame(rows).to_csv(results_dir / "shock_recovery_paired_differences_95ci.csv", index=False)


def run_mechanism_sensitivity(
    base_cfg: ModelConfig,
    results_dir: Path,
    n_reps: int = 20,
    quick: bool = False,
):
    print("\n[5/5] Mechanism robustness: uncertain behavioural/update parameters")
    out_path = results_dir / "mechanism_sensitivity_replications.csv"
    completed = _completed_keys(out_path, ["factor", "level", "replication"])

    factors = [
        "expected_quality_weight",
        "local_social_learning_weight",
        "choice_temperature",
        "mode_value_learning_rate",
        "trust_learning_rate",
        "knowledge_learning_rate",
        "tie_reinforcement_rate",
        "tie_decay",
        "external_peer_search_probability",
        "hybrid_coordination_burden",
        "task_similarity_temperature",
        "peer_expertise_visibility",
        "peer_communication_noise_sd",
        "peer_max_transfer_efficiency",
        "ai_ambiguity_verification_cost",
        "ai_access_saturation",
    ]
    if quick:
        factors = factors[:3]

    settings = [("reference", "reference", 1.0)]
    for f in factors:
        settings.append((f, "-20%", 0.80))
        settings.append((f, "+20%", 1.20))

    for idx, (factor, level, multiplier) in enumerate(settings, 1):
        for rep in range(n_reps):
            if (factor, level, rep) in completed:
                continue
            cfg = replace(base_cfg, ai_enabled=True, seed=_base_seed(rep))
            if factor != "reference":
                old = float(getattr(cfg, factor))
                new = old * multiplier
                # Respect probability bounds for probability-like parameters.
                if factor in {"external_peer_search_probability", "peer_expertise_visibility", "peer_max_transfer_efficiency"}:
                    new = float(np.clip(new, 0.001, 0.95))
                cfg = replace(cfg, **{factor: new})
            t0 = time.time()
            summary, _, _, _ = run_model(cfg, tail=max(20, base_cfg.episodes // 6))
            _append_csv(
                out_path,
                [
                    {
                        "factor": factor,
                        "level": level,
                        "multiplier": multiplier,
                        "replication": rep,
                        "seed": cfg.seed,
                        **summary,
                        "runtime_seconds": time.time() - t0,
                    }
                ],
            )
        print(f"  sensitivity setting {idx:02d}/{len(settings)}: {factor} {level}")

    df = pd.read_csv(out_path)
    summarize_with_ci(df, ["factor", "level", "multiplier"], FINAL_METRICS).to_csv(
        results_dir / "mechanism_sensitivity_summary_95ci.csv", index=False
    )

    # Paired effects versus the reference setting with common seeds.
    ref = df[(df.factor == "reference") & (df.level == "reference")].set_index("replication")
    effect_rows = []
    for (factor, level), g in df[df.factor != "reference"].groupby(["factor", "level"]):
        g = g.set_index("replication")
        common = ref.index.intersection(g.index)
        for metric in [
            "final_human_interaction_rate",
            "final_ai_use_rate",
            "final_mean_quality",
            "final_mean_knowledge",
            "final_mean_peer_trust",
            "final_mean_tie_strength",
            "final_recent_active_tie_fraction",
            "final_mean_unique_peer_partners",
            "edge_retention_ratio",
        ]:
            diff = g.loc[common, metric].to_numpy() - ref.loc[common, metric].to_numpy()
            mean, lo, hi, n = ci95(diff)
            effect_rows.append(
                {
                    "factor": factor,
                    "level": level,
                    "metric": metric,
                    "mean_difference": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": n,
                }
            )
    pd.DataFrame(effect_rows).to_csv(results_dir / "mechanism_sensitivity_paired_effects_95ci.csv", index=False)


def save_config(base_cfg: ModelConfig, results_dir: Path, run_settings: Dict):
    payload = {
        "model_config": asdict(base_cfg),
        "run_settings": run_settings,
        "notes": {
            "confidence_intervals": "Two-sided 95% Student-t intervals across independent replications.",
            "main_interpretation": "AI-enabled runs use one common behavioural mechanism in which task-resource matching, heterogeneous agents and accumulated experience shape route choice and outcomes.",
            "human_interaction_rate": "Peer-first share + Hybrid share (any task episode involving a peer).",
            "ai_use_rate": "AI-first share + Hybrid share (any task episode involving AI).",
        },
    }
    with open(results_dir / "analysis_configuration.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
