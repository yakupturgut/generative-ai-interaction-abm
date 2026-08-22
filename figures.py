from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PEER = "#2F7DBA"
AI = "#F26B4A"
HYBRID = "#00A6A6"
INK = "#071A2F"
SLATE = "#58697B"
GREEN = "#2D9B65"
GOLD = "#D89B22"


def _style():
    # Deliberately large typography for conference projection.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 19,
        "axes.titlesize": 25,
        "axes.labelsize": 21,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "figure.titlesize": 26,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "lines.linewidth": 3.4,
    })


def save_all(fig, figures_dir: Path, stem: str):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_interaction_trajectories(results_dir: Path, figures_dir: Path):
    d = pd.read_csv(results_dir / "reference_history_summary_95ci.csv")
    d = d[d.system == "AI-enabled"]
    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    for metric, label, color in [
        ("human_interaction_rate", "Human interaction", PEER),
        ("ai_use_rate", "AI use", AI),
    ]:
        s = d[d.metric == metric].sort_values("episode")
        ax.plot(s.episode, s["mean"], color=color, label=label)
        ax.fill_between(s.episode, s.ci_low, s.ci_high, color=color, alpha=0.14)
    ax.set_title("Experience shifts the balance between AI and human interaction")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Share of task episodes")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=False, loc="best")
    ax.text(0.01, -0.16, "Shaded bands: 95% confidence intervals across independent runs.",
            transform=ax.transAxes, color=SLATE, fontsize=15)
    fig.tight_layout()
    save_all(fig, figures_dir, "01_ai_use_vs_human_interaction")


def plot_task_choice(results_dir: Path, figures_dir: Path):
    """Show whether task-specific mode choices separate with experience."""
    p = results_dir / "task_mode_choice_early_late_95ci.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.metric == "share"]
    tasks = ["Structured", "Contextual", "Integrative"]
    modes = ["Peer-first", "AI-first", "Hybrid"]
    colors = [PEER, AI, HYBRID]
    x = np.arange(len(tasks))
    width = 0.23
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.8), sharey=True)
    for ax, phase in zip(axes, ["Early", "Late"]):
        dp = d[d.phase == phase]
        for k, (mode, color) in enumerate(zip(modes, colors)):
            sub = dp[dp["mode"] == mode].set_index("task_type")
            means=[]; los=[]; his=[]
            for task in tasks:
                r = sub.loc[task]
                means.append(float(r["mean"])); los.append(float(r["ci_low"])); his.append(float(r["ci_high"]))
            means=np.asarray(means); los=np.asarray(los); his=np.asarray(his)
            xx=x+(k-1)*width
            ax.bar(xx,means,width=width,color=color,alpha=.94,label=mode)
            ax.errorbar(xx,means,yerr=[means-los,his-means],fmt="none",ecolor=INK,capsize=4,linewidth=1.4)
        ax.set_xticks(x); ax.set_xticklabels(tasks)
        ax.set_ylim(0,0.60); ax.grid(axis="y",alpha=.17)
        ax.set_title(f"{phase} episodes")
    axes[0].set_ylabel("Choice share")
    axes[1].legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,-.12))
    fig.suptitle("Do agents learn different support strategies for different tasks?")
    fig.tight_layout(rect=[0,0.05,1,.94])
    save_all(fig,figures_dir,"02_task_context_and_mode_choice")

def plot_task_support_gain(results_dir: Path, figures_dir: Path):
    d = pd.read_csv(results_dir / "task_mode_summary_95ci.csv")
    d = d[d.metric == "mean_support_gain"]
    tasks = ["Structured", "Contextual", "Integrative"]
    modes = ["Peer-first", "AI-first", "Hybrid"]
    colors = {"Peer-first": PEER, "AI-first": AI, "Hybrid": HYBRID}
    offsets = {"Peer-first": -0.18, "AI-first": 0.0, "Hybrid": 0.18}
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    for mode in modes:
        sub = d[d["mode"] == mode].set_index("task_type")
        means=[]; los=[]; his=[]
        for task in tasks:
            r=sub.loc[task]
            means.append(float(r["mean"])); los.append(float(r["ci_low"])); his.append(float(r["ci_high"]))
        means=np.asarray(means)
        ax.errorbar(x+offsets[mode], means,
                    yerr=[means-np.asarray(los), np.asarray(his)-means],
                    fmt="o", markersize=12, capsize=5, color=colors[mode], label=mode)
    ax.axhline(0, color=SLATE, linewidth=1.3)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("Quality gain over solving alone")
    ax.set_title("Realized support value differs across tasks and chosen routes")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="best")
    fig.tight_layout()
    save_all(fig, figures_dir, "03_task_specific_support_value")


def _factorial_condition_means(results_dir: Path):
    d = pd.read_csv(results_dir / "environment_factorial_replications.csv")
    return d.groupby(
        ["condition", "ai_access_mean", "ai_reliability", "verification_support", "peer_response_probability"],
        as_index=False,
    ).mean(numeric_only=True)


def plot_access_crowding_out(results_dir: Path, figures_dir: Path):
    d = pd.read_csv(results_dir / "environment_factorial_replications.csv")
    # Average first within replication-level experimental cells, then across the
    # other manipulated factors. This makes the plotted contrast easy to read.
    g = d.groupby(["ai_access_mean", "replication"], as_index=False).agg(
        human=("final_human_interaction_rate", "mean"),
        ai_use=("final_ai_use_rate", "mean"),
    )
    rows=[]
    for access, s in g.groupby("ai_access_mean"):
        for metric in ["human", "ai_use"]:
            x=s[metric].to_numpy(float); mean=x.mean()
            if len(x)>1:
                se=x.std(ddof=1)/np.sqrt(len(x)); import scipy.stats as st; h=st.t.ppf(.975,len(x)-1)*se
            else: h=np.nan
            rows.append((access,metric,mean,mean-h,mean+h))
    sm=pd.DataFrame(rows,columns=["access","metric","mean","lo","hi"])
    fig, ax=plt.subplots(figsize=(12.4,7.0))
    for metric,label,color in [("human","Human interaction",PEER),("ai_use","AI use",AI)]:
        s=sm[sm.metric==metric].sort_values("access")
        ax.errorbar(s.access,s["mean"],yerr=[s["mean"]-s.lo,s.hi-s["mean"]],
                    marker="o",markersize=10,capsize=5,color=color,label=label)
    ax.set_xlabel("Mean AI access / ease")
    ax.set_ylabel("Final interaction rate")
    ax.set_ylim(0,1)
    ax.set_title("Does easier AI access crowd out human interaction?")
    ax.grid(axis="y",alpha=.18); ax.legend(frameon=False)
    fig.tight_layout(); save_all(fig,figures_dir,"04_ai_access_and_crowding_out")


def plot_reliability_performance(results_dir: Path, figures_dir: Path):
    d=pd.read_csv(results_dir/"environment_factorial_replications.csv")
    g=d.groupby(["ai_reliability","replication"],as_index=False).agg(
        quality=("final_mean_quality","mean"), knowledge=("final_mean_knowledge","mean")
    )
    fig,ax=plt.subplots(figsize=(12.4,7.0))
    for metric,label,color in [("quality","Task quality",AI),("knowledge","Knowledge",GREEN)]:
        rows=[]
        for rel,s in g.groupby("ai_reliability"):
            x=s[metric].to_numpy(float); mean=x.mean()
            if len(x)>1:
                from scipy.stats import t
                h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
            else:h=np.nan
            rows.append((rel,mean,mean-h,mean+h))
        s=pd.DataFrame(rows,columns=["rel","mean","lo","hi"]).sort_values("rel")
        ax.errorbar(s.rel,s["mean"],yerr=[s["mean"]-s.lo,s.hi-s["mean"]],marker="o",markersize=10,capsize=5,color=color,label=label)
    ax.set_xlabel("AI reliability")
    ax.set_ylabel("Final outcome")
    ax.set_title("Does better AI change performance and learning?")
    ax.grid(axis="y",alpha=.18); ax.legend(frameon=False)
    fig.tight_layout(); save_all(fig,figures_dir,"05_ai_reliability_and_outcomes")


def plot_tradeoff(results_dir: Path, figures_dir: Path):
    g=_factorial_condition_means(results_dir)
    fig,ax=plt.subplots(figsize=(11.2,7.4))
    sc=ax.scatter(g.final_human_interaction_rate,g.final_mean_quality,
                  s=120,c=g.ai_access_mean,cmap="viridis",edgecolors="white",linewidths=1.0,alpha=.90)
    cb=fig.colorbar(sc,ax=ax,pad=.03); cb.set_label("Mean AI access",fontsize=18); cb.ax.tick_params(labelsize=14)
    ax.set_xlabel("Final human-interaction rate")
    ax.set_ylabel("Final task quality")
    ax.set_title("Performance gains can come with different social interaction levels")
    ax.grid(alpha=.16)
    fig.tight_layout(); save_all(fig,figures_dir,"06_performance_social_tradeoff")


def plot_shock_behavior(results_dir: Path, figures_dir: Path):
    d=pd.read_csv(results_dir/"shock_history_summary_95ci.csv")
    conds=["Stable reliability","Temporary drop + recovery"]
    colors={"Stable reliability":SLATE,"Temporary drop + recovery":AI}
    fig,ax=plt.subplots(figsize=(12.6,7.0))
    for cond in conds:
        s=d[(d.shock_condition==cond)&(d.metric=="human_interaction_rate")].sort_values("episode")
        ax.plot(s.episode,s["mean"],color=colors[cond],label=cond)
        ax.fill_between(s.episode,s.ci_low,s.ci_high,color=colors[cond],alpha=.12)
    rel=d[(d.shock_condition=="Temporary drop + recovery")&(d.metric=="ai_reliability")].sort_values("episode")
    changes=np.where(np.abs(np.diff(rel["mean"].to_numpy()))>1e-8)[0]+1
    eps=rel.episode.to_numpy()
    for k,idx in enumerate(changes[:2]):
        ax.axvline(eps[idx],color=INK,linestyle="--" if k==0 else ":",linewidth=1.6)
    ax.set_xlabel("Episode"); ax.set_ylabel("Human-interaction rate"); ax.set_ylim(0,1)
    ax.set_title("How quickly does human interaction respond to an AI reliability shock?")
    ax.grid(axis="y",alpha=.18); ax.legend(frameon=False)
    fig.tight_layout(); save_all(fig,figures_dir,"07_reliability_shock_behavioral_response")


def plot_shock_trust_learning(results_dir: Path, figures_dir: Path):
    d=pd.read_csv(results_dir/"shock_history_summary_95ci.csv")
    cond="Temporary drop + recovery"
    fig,ax=plt.subplots(figsize=(12.6,7.0))
    # Rescale neither metric; both are [0,1].
    for metric,label,color in [("mean_ai_trust","AI trust",AI),("mean_knowledge","Knowledge",GREEN)]:
        s=d[(d.shock_condition==cond)&(d.metric==metric)].sort_values("episode")
        ax.plot(s.episode,s["mean"],color=color,label=label)
        ax.fill_between(s.episode,s.ci_low,s.ci_high,color=color,alpha=.12)
    rel=d[(d.shock_condition==cond)&(d.metric=="ai_reliability")].sort_values("episode")
    changes=np.where(np.abs(np.diff(rel["mean"].to_numpy()))>1e-8)[0]+1; eps=rel.episode.to_numpy()
    for k,idx in enumerate(changes[:2]): ax.axvline(eps[idx],color=INK,linestyle="--" if k==0 else ":",linewidth=1.6)
    ax.set_xlabel("Episode"); ax.set_ylabel("Mean state"); ax.set_ylim(0,1)
    ax.set_title("Trust and accumulated learning may recover at different speeds")
    ax.grid(axis="y",alpha=.18); ax.legend(frameon=False)
    fig.tight_layout(); save_all(fig,figures_dir,"08_reliability_shock_trust_and_learning")


def _nice_factor(x):
    m={
        "expected_quality_weight":"Expected-performance weight",
        "local_social_learning_weight":"Local social learning",
        "choice_temperature":"Choice noise / temperature",
        "mode_value_learning_rate":"Experience learning",
        "trust_learning_rate":"Trust learning",
        "knowledge_learning_rate":"Knowledge learning",
        "tie_reinforcement_rate":"Tie reinforcement",
        "tie_decay":"Tie decay",
        "external_peer_search_probability":"External peer search",
        "hybrid_coordination_burden":"Hybrid coordination burden",
        "task_similarity_temperature":"Task-memory generalization",
        "peer_expertise_visibility":"Peer expertise visibility",
        "peer_communication_noise_sd":"Peer communication noise",
        "peer_max_transfer_efficiency":"Peer transfer ceiling",
        "ai_ambiguity_verification_cost":"AI ambiguity/checking cost",
        "ai_access_saturation":"AI-access saturation",
    }
    return m.get(x,x.replace("_"," ").title())


def plot_sensitivity(results_dir: Path, figures_dir: Path):
    d=pd.read_csv(results_dir/"mechanism_sensitivity_paired_effects_95ci.csv")
    sub=d[d.metric=="final_human_interaction_rate"].copy()
    # Plot the effect magnitude of ±20% settings, sorted by maximum absolute effect.
    maxeff=sub.groupby("factor").mean_difference.apply(lambda x: np.max(np.abs(x))).sort_values(ascending=True)
    factors=list(maxeff.index)
    y=np.arange(len(factors))
    fig,ax=plt.subplots(figsize=(12.8,max(7.2,0.50*len(factors)+2.0)))
    for level,marker,color,shift in [("-20%","o",PEER,-.10),("+20%","s",AI,.10)]:
        means=[];lo=[];hi=[]
        for f in factors:
            r=sub[(sub.factor==f)&(sub.level==level)].iloc[0]
            means.append(r.mean_difference);lo.append(r.ci_low);hi.append(r.ci_high)
        means=np.asarray(means)
        ax.errorbar(means,y+shift,xerr=[means-np.asarray(lo),np.asarray(hi)-means],fmt=marker,markersize=8,capsize=3,color=color,label=level,linestyle="none")
    ax.axvline(0,color=SLATE,linewidth=1.3)
    ax.set_yticks(y);ax.set_yticklabels([_nice_factor(f) for f in factors])
    ax.set_xlabel("Paired change in final human-interaction rate")
    ax.set_title("Which modelling assumptions most affect the social result?")
    ax.grid(axis="x",alpha=.18);ax.legend(frameon=False)
    fig.tight_layout();save_all(fig,figures_dir,"09_mechanism_sensitivity_human_interaction")


def plot_agent_traits(results_dir: Path, figures_dir: Path):
    d=pd.read_csv(results_dir/"agent_trait_behavior_associations_95ci.csv")
    d=d[d.metric=="pearson_r"]
    traits=["ai_access","ai_literacy","sociability","verification","initial_knowledge","initial_confidence"]
    labels={"ai_access":"AI access","ai_literacy":"AI literacy","sociability":"Sociability","verification":"Verification","initial_knowledge":"Initial knowledge","initial_confidence":"Confidence"}
    sub=d[d.outcome=="ai_share"].set_index("trait")
    means=np.array([float(sub.loc[t,"mean"]) for t in traits]);lo=np.array([float(sub.loc[t,"ci_low"]) for t in traits]);hi=np.array([float(sub.loc[t,"ci_high"]) for t in traits])
    order=np.argsort(np.abs(means));traits=[traits[i] for i in order];means=means[order];lo=lo[order];hi=hi[order];y=np.arange(len(traits))
    fig,ax=plt.subplots(figsize=(11.8,7.0))
    ax.errorbar(means,y,xerr=[means-lo,hi-means],fmt="o",markersize=11,capsize=5,color=AI)
    ax.axvline(0,color=SLATE,linewidth=1.3)
    ax.set_yticks(y);ax.set_yticklabels([labels[t] for t in traits])
    ax.set_xlabel("Within-run association with AI-first share")
    ax.set_title("Which individual attributes shape repeated AI-first use?")
    ax.grid(axis="x",alpha=.18)
    fig.tight_layout();save_all(fig,figures_dir,"10_agent_attributes_and_ai_use")


def plot_replication_precision(results_dir: Path, figures_dir: Path):
    p=results_dir/"replication_convergence.csv"
    if not p.exists():return
    d=pd.read_csv(p);d=d[d.system=="AI-enabled"]
    fig,ax=plt.subplots(figsize=(11.5,6.8))
    for metric,label,color in [("final_human_interaction_rate","Human interaction",PEER),("final_mean_quality","Task quality",AI),("final_mean_knowledge","Knowledge",GREEN)]:
        s=d[d.metric==metric].sort_values("n_replications")
        ax.plot(s.n_replications,s.ci_half_width,marker="o",markersize=8,label=label,color=color)
    ax.set_xlabel("Independent replications");ax.set_ylabel("95% CI half-width")
    ax.set_title("Monte Carlo precision across replications")
    ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.tight_layout();save_all(fig,figures_dir,"11_replication_precision")



def plot_experience_reinforcement(results_dir: Path, figures_dir: Path):
    p = results_dir / "experience_reinforcement_95ci.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.metric == "next_probability_same_route"]
    signals = ["Worse than expected", "Near expectation", "Better than expected"]
    modes = [("Peer-first", PEER), ("AI-first", AI), ("Hybrid", HYBRID)]
    x = np.arange(len(signals))
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    for mode, color in modes:
        sub = d[d["mode"] == mode].set_index("experience_signal")
        means=[]; los=[]; his=[]
        for sig in signals:
            if sig not in sub.index:
                means.append(np.nan); los.append(np.nan); his.append(np.nan)
            else:
                r=sub.loc[sig]; means.append(float(r["mean"])); los.append(float(r["ci_low"])); his.append(float(r["ci_high"]))
        means=np.asarray(means);los=np.asarray(los);his=np.asarray(his)
        ax.errorbar(x,means,yerr=[means-los,his-means],marker="o",markersize=11,capsize=5,color=color,label=mode)
    ax.set_xticks(x);ax.set_xticklabels(["Worse\nthan expected","Near\nexpectation","Better\nthan expected"])
    ax.set_ylabel("Probability of using the same route\non the next similar task")
    ax.set_title("Experience feeds back into the next support choice")
    ax.set_ylim(0,1);ax.grid(axis="y",alpha=.18);ax.legend(frameon=False,ncol=3,loc="best")
    fig.tight_layout();save_all(fig,figures_dir,"12_experience_feedback_next_choice")

def make_all_figures(results_dir: Path, figures_dir: Path):
    _style()
    funcs=[
        plot_interaction_trajectories,
        plot_task_choice,
        plot_task_support_gain,
        plot_access_crowding_out,
        plot_reliability_performance,
        plot_tradeoff,
        plot_shock_behavior,
        plot_shock_trust_learning,
        plot_sensitivity,
        plot_agent_traits,
        plot_replication_precision,
        plot_experience_reinforcement,
    ]
    for f in funcs:
        try:
            f(results_dir,figures_dir)
            print(f"  figure OK: {f.__name__}")
        except Exception as exc:
            print(f"  figure skipped/failed: {f.__name__}: {exc}")
