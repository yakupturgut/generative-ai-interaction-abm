# ODD Protocol Description for the Agent-Based Model
## Generative AI as an Interaction Intermediary: Collaboration, Trust, and Social Learning

**Author:** Yakup Turgut, Kırklareli University, Kırklareli, Turkey

This document provides the detailed Overview--Design concepts--Details (ODD) specification of the agent-based model used in the study. Heterogeneous agents repeatedly solve multidimensional tasks through Peer-first, AI-first, or Hybrid support. Their choices depend on learned task-specific expectations, source trust, local social information, and access or coordination costs. Realized outcomes update knowledge, confidence, source trust, route expectations, and weighted social ties. The document reports entities, state variables, scheduling, equations, stochastic processes, the complete reference parameterization, experiment settings, output measures, and reproducibility procedures.

# Purpose of this model document

This document is the repository-level specification of the simulation
model. Its purpose is to provide enough detail for an independent reader
to understand, inspect, reproduce, and modify the model beyond the
compressed description possible in the proceedings paper. It has four
functions:

1.  describe the model using the ODD structure;

2.  make state variables, behavioral rules, parameter values, and
    scheduling explicit;

3.  explain how abstract constructs such as knowledge, trust, AI access,
    peer expertise, and tie strength are operationalized;

4.  connect the conceptual description to the Python implementation,
    experiment scripts, saved result files, and figures.

The implementation and all repository documentation are available at
<https://github.com/yakupturgut/generative-ai-interaction-abm>.

# Model scope and interpretation

The model is an exploratory, mechanism-based ABM rather than a
calibrated predictor of a particular workplace, classroom, platform, or
population. It formalizes transparent assumptions about repeated problem
solving when both generative AI and human peers can be used as support
resources. System-level patterns are generated from local interactions
and adaptive feedback, consistent with the generative logic of social
simulation .

The constructs are operational. *Knowledge* denotes task-relevant
capability in a three-dimensional simulated requirement space. *AI
literacy* denotes the ability to extract productive value from AI
output. *Sociability* affects the accessibility and transfer efficiency
of peer support. *Verification* affects the ability to scrutinize AI
output and integrate multiple sources. *Trust* is a learned
source-specific belief that moves toward experienced source quality.
*Tie strength* is a weighted representation of an active social
relationship whose value changes through use and non-use.

The route-choice scores are bounded-rational behavioral propensities.
They are not empirically estimated welfare utilities. Probabilistic
softmax choice provides stochastic adaptation under heterogeneous
expectations and imperfect information, drawing on random-utility ideas
and bounded rationality .

# Overview

## Purpose and patterns

The model examines how repeated problem-solving choices among
**Peer-first**, **AI-first**, and **Hybrid** support shape:

- the composition of AI use and human interaction;

- task quality and task success;

- domain-specific knowledge accumulation;

- trust in AI and partner-specific peer trust;

- social-tie reinforcement, decay, retention, and formation;

- performance and knowledge inequality;

- recovery after temporary or persistent degradation in AI reliability.

The central mechanism is a repeated feedback loop. A task arrives, the
agent forms expectations for feasible support routes, a route is
selected probabilistically, the task outcome is realized, and the
resulting experience updates future expectations, trust, knowledge,
confidence, and social relations.

## Entities, state variables, and scales

The model contains human agents, tasks, a weighted undirected social
network, and an AI support environment. There is no physical geography.
One episode is an abstract collaborative problem-solving opportunity
rather than a fixed unit of calendar time.

<div id="tab:agentstates">

| Variable                              | Interpretation                                                       | Scale / role                                                                    |
|:--------------------------------------|:---------------------------------------------------------------------|:--------------------------------------------------------------------------------|
| $\mathbf{k}_i=(k_{i1},k_{i2},k_{i3})$ | Three-dimensional task-relevant skill vector of agent $i$.           | $[0,1]^3$; changes through practice and novel information.                      |
| $a_i$                                 | AI literacy; ability to productively use AI output.                  | $[0,1]$; affects positive AI uptake and hybrid integration.                     |
| $s_i$                                 | Sociability; ability/orientation to engage effectively with peers.   | $[0,1]$; affects peer cost, search, transfer, and learning.                     |
| $v_i$                                 | Verification tendency.                                               | $[0,1]$; affects AI checking, hybrid integration, and harmful-output filtering. |
| $c_i$                                 | Confidence.                                                          | $[0,1]$; contributes to realized task quality and adapts slowly.                |
| $A_i$                                 | Individual AI access/ease.                                           | $[0,1]$; shapes AI availability and access friction.                            |
| $t_i^{AI}$                            | Trust in AI.                                                         | $[0,1]$; moves toward experienced task-relevant AI source quality.              |
| $t_{ij}^{P}$                          | Agent $i$’s trust in peer $j$.                                       | $[0,1]$; partner-specific and experience-based.                                 |
| $V_{iam}$                             | Expected realized quality of mode $m$ for task anchor $a$.           | $[0.05,0.95]$; similarity-weighted experience memory.                           |
| $E_{ija}$                             | Agent $i$’s expectation of peer $j$’s expertise for task anchor $a$. | $[0.01,0.99]$; partner- and task-specific.                                      |
| $L_{iam}$                             | Local observational signal for mode $m$ and task anchor $a$.         | $[-0.18,0.24]$; updated from observed neighbors’ marginal support gains.        |
| $w_{ij}$                              | Social-tie strength.                                                 | $[0,1]$; reinforced by useful peer contribution, decays when unused.            |

Principal agent-level state variables.

</div>

Each task contains a three-dimensional requirement vector $\mathbf{r}$,
ambiguity $u$, difficulty $d$, success threshold $h$, and a task-family
label used for interpretation. The reference model contains $N=180$
agents and $T=150$ episodes.

## Process overview and scheduling

At each episode, the following sequence is executed:

1.  apply any scheduled change in AI reliability;

2.  freeze the current skills and ties for within-episode reference;

3.  generate one task for each agent;

4.  draw AI availability for each agent and identify reachable peers;

5.  compute route costs, expected values, trust terms, and local social
    signals;

6.  convert feasible route propensities to probabilities and sample
    Peer-first, AI-first, or Hybrid;

7.  if a peer is used, choose the partner probabilistically from
    perceived expertise, trust, tie strength, and search cost;

8.  realize task quality through multidimensional task–resource
    matching;

9.  classify task success using the task-specific threshold;

10. update route expectations, AI trust, peer trust, partner expertise
    expectations, knowledge, and confidence;

11. update used social ties from the peer’s marginal contribution;

12. decay unused ties and remove ties below the removal threshold;

13. update local observational-learning memories from visible neighbors;

14. record episode-level and, when requested, task- and agent-level
    outputs.

Agents are processed in a random permutation within each episode. Task
generation is simultaneous, while state updates occur as agents complete
their decisions.

# Design concepts

## Basic principles

The model combines five principles: heterogeneous capabilities, bounded
probabilistic choice, multidimensional task–resource matching,
experience-based learning, and co-evolving social relationships. The
same route-choice architecture is used for Peer-first, AI-first, and
Hybrid support. Differences in realized outcomes arise from source
availability, task requirements, source capability, agent attributes,
partner characteristics, and accumulated experience.

## Emergence

Aggregate patterns such as AI-use rate, human-interaction rate, trust
trajectories, knowledge inequality, and social-tie activity are
system-level outcomes of repeated micro decisions. They are not direct
state variables chosen by agents. In particular, AI-first use changes
social structure through opportunity displacement: episodes spent
without a peer provide no peer interaction with which to reinforce a
social tie, while unused ties continue to decay slowly.

## Adaptation

Agents adapt after realized outcomes. Route expectations are updated
toward the experienced quality of the selected route for similar tasks.
AI trust is updated toward experienced AI source quality. Peer trust and
peer-expertise expectations are updated toward experienced partner
quality. Confidence adapts toward task quality. Knowledge grows through
practice and novel information. Social ties adapt through the realized
marginal contribution of peer interaction.

## Objectives and decision-making

Agents seek support for the current task rather than optimizing a
long-run social objective. The route propensity for agent $i$ and route
$m\in\{P,A,H\}$ is
$$U_{im}=\beta_Q\widehat Q_{im}+\beta_T T_{im}+\beta_L L_{im}-C_{im},
\label{eq:oddutility}$$ where $\widehat Q_{im}$ is learned expected
realized quality, $T_{im}$ is source trust, $L_{im}$ is local
observational information, and $C_{im}$ is route cost. Feasible-route
probabilities are
$$\Pr_i(m)=\frac{\exp(U_{im}/\tau)}{\sum_{r\in\mathcal F_i}\exp(U_{ir}/\tau)},$$
with an exploration floor applied over feasible routes.

## Learning

Three learning channels operate simultaneously:

1.  **Experiential route learning**: selected-route quality updates
    task-similarity-weighted expected quality.

2.  **Source learning**: source quality updates AI or partner-specific
    peer trust, and peer interaction updates perceived partner
    expertise.

3.  **Capability learning**: task practice and novel support information
    increase the agent’s three-dimensional skill vector.

## Prediction

Agents use task-anchor memories to predict route quality for the current
task. Current task requirements are mapped to the three anchors through
smooth cosine-similarity weights. Peer expertise is perceived
imperfectly. The model therefore separates latent source capability from
the agent’s current expectation of that capability.

## Sensing

Each agent has access to its own traits, current task requirements,
personal route memories, AI trust and access, available peers,
partner-specific peer expectations and trust, and current tie strengths.
A stochastic subset of neighbors’ recent support outcomes is observed
through the local social-learning mechanism. Agents observe only the
outcome of the route they actually use.

## Interaction

Peer-first and Hybrid routes involve another human agent. Peer
availability depends on current network neighbors, peer-response
probability, and occasional external friend-of-friend search. A selected
peer transmits task-relevant capability imperfectly. Social interaction
can reinforce an existing tie or create a new tie when a newly
discovered peer provides sufficiently positive marginal value.

## Stochasticity

Stochasticity enters through:

- heterogeneous initial traits and skill dimensions;

- initial social-network topology and tie weights;

- task family, task requirement vector, ambiguity, and difficulty;

- AI capability noise;

- peer communication noise;

- AI availability and peer response;

- external peer discovery;

- probabilistic route choice and partner choice;

- observation of neighbors;

- final task-quality noise.

Independent replications use deterministic seed construction so that
paired experimental contrasts can use common random numbers.

## Collectives

The social network constitutes a dynamic collective structure.
Network-level quantities include density, weighted degree, active-tie
fraction, tie retention, new-edge ratio, strong-tie prevalence, and
unique peer partners. Collective interaction composition is measured
through Peer-first, AI-first, and Hybrid shares.

## Observation

Two overlapping interaction indicators are used: $$\begin{aligned}
\text{Human interaction rate} &= \text{Peer-first share}+\text{Hybrid share},\\
\text{AI-use rate} &= \text{AI-first share}+\text{Hybrid share}.
\end{aligned}$$ Because Hybrid includes both AI and another person,
these indicators are not complements and need not sum to one.

# Details

## Initialization

The initial skill profile combines a general capability component with
domain-specific deviations. General capability is drawn from a bounded
normal distribution centered near the reference knowledge mean; three
independent domain deviations generate $\mathbf{k}_i$. AI literacy,
sociability, verification, confidence, and AI access are bounded normal
traits. Initial task-specific route values are centered near 0.50 with
small random variation. AI trust is centered near 0.50.

The initial social graph is a connected Watts–Strogatz small-world
network . The nearest-neighbor parameter is selected to approximate the
target density. Initial tie weights are uniformly drawn from
$[0.42,0.68]$. Partner-specific peer trust is centered near 0.50.
Initial peer-expertise expectations combine a neutral prior with noisy
information about the connected partner’s skill profile.

## Reference parameterization

Tables <a href="#tab:param_ai" data-reference-type="ref"
data-reference="tab:param_ai">2</a>–<a href="#tab:param_update" data-reference-type="ref"
data-reference="tab:param_update">5</a> report the full reference
parameterization stored in `ModelConfig` and
`results/analysis_configuration.json`.

<div id="tab:param_ai">

| Parameter               | Value              | Interpretation                                            |
|:------------------------|:-------------------|:----------------------------------------------------------|
| Number of agents        | 180                | Population size.                                          |
| Episodes                | 150                | Repeated task opportunities per run.                      |
| AI enabled              | True               | AI support available in the reference system.             |
| AI reliability          | 0.78               | Scales the realized AI capability profile.                |
| Mean AI access          | 0.60               | Population mean of individual AI ease/access.             |
| AI access SD            | 0.16               | Heterogeneity in individual AI access.                    |
| Verification support    | 0.55               | Environmental support for checking/integrating AI output. |
| AI availability floor   | 0.70               | Minimum availability under the saturating access mapping. |
| AI availability ceiling | 0.97               | Maximum availability under the saturating access mapping. |
| AI access saturation    | 2.0                | Curvature of the access-to-availability mapping.          |
| AI capability profile   | $(0.94,0.64,0.76)$ | Baseline capability across three task dimensions.         |
| AI capability noise SD  | 0.085              | Episode-level capability noise.                           |
| AI output concentration | 22.0               | Configuration value retained for AI-output dispersion.    |
| AI ambiguity penalty    | 0.16               | Reduction in effective AI capability with task ambiguity. |

Scale and AI-environment parameters.

</div>

<div id="tab:param_peer">

| Parameter                        | Value           | Interpretation                                                       |
|:---------------------------------|:----------------|:---------------------------------------------------------------------|
| Network density                  | 0.055           | Target initial small-world density.                                  |
| Rewiring probability             | 0.10            | Watts–Strogatz rewiring probability.                                 |
| Peer-response probability        | 0.80            | Probability an existing neighbor is reachable.                       |
| External peer-search probability | 0.12            | Base probability of outside-network search.                          |
| Peer-expertise visibility        | 0.30            | Weight of noisy expertise information in the initial partner belief. |
| Social observation rate          | 0.45            | Probability of observing a neighbor’s recent outcome.                |
| Peer communication noise SD      | 0.095           | Noise in communicated peer capability.                               |
| Peer ambiguity penalty           | 0.10            | Ambiguity-related reduction in peer transfer efficiency.             |
| Minimum peer transfer efficiency | 0.28            | Lower bound on transferred peer capability.                          |
| Maximum peer transfer efficiency | 0.80            | Upper bound on transferred peer capability.                          |
| Task mix                         | $(1/3,1/3,1/3)$ | Structured, contextual, integrative family probabilities.            |
| Mean task difficulty             | 0.58            | Mean of the Beta difficulty draw.                                    |
| Difficulty concentration         | 10.0            | Concentration of the Beta difficulty distribution.                   |
| Task-similarity temperature      | 0.090           | Softness of task-to-anchor similarity weights.                       |
| Mean initial knowledge           | 0.52            | Center of general initial capability.                                |
| Mean AI literacy                 | 0.50            | Population AI-literacy center.                                       |
| Mean sociability                 | 0.55            | Population sociability center.                                       |
| Mean verification                | 0.52            | Population verification center.                                      |
| Mean confidence                  | 0.50            | Population confidence center.                                        |
| Trait SD                         | 0.14            | Standard deviation for bounded heterogeneous traits.                 |

Peer, network, task, and agent-trait parameters.

</div>

<div id="tab:param_choice">

| Parameter                              | Value | Interpretation                                        |
|:---------------------------------------|:------|:------------------------------------------------------|
| Expected-quality weight $\beta_Q$      | 3.60  | Weight on learned route-quality expectation.          |
| Trust weight $\beta_T$                 | 0.12  | Weight on source trust.                               |
| Local social-learning weight $\beta_L$ | 0.55  | Weight on observed local support value.               |
| Choice temperature $\tau$              | 0.31  | Softmax stochasticity.                                |
| Exploration floor                      | 0.025 | Minimum exploration mass per feasible route.          |
| Peer base cost                         | 0.22  | Base peer-coordination cost.                          |
| AI base cost                           | 0.20  | Base AI-access cost before access saturation.         |
| AI ambiguity verification cost         | 0.14  | Extra checking burden on ambiguous AI use.            |
| Hybrid extra cost                      | 0.12  | Additional integration/coordination cost.             |
| Weak-network search cost               | 0.10  | Search cost for low-strength or external peer access. |

Choice and route-cost parameters.

</div>

<div id="tab:param_update">

| Parameter                   | Value         | Interpretation                                            |
|:----------------------------|:--------------|:----------------------------------------------------------|
| Mode-value learning rate    | 0.32          | Adaptation of task-specific route-quality expectations.   |
| Trust learning rate         | 0.12          | Adaptation of AI and peer trust.                          |
| Local-memory learning rate  | 0.13          | Adaptation of observed neighborhood route signals.        |
| Confidence learning rate    | 0.030         | Adaptation of confidence toward realized quality.         |
| Practice learning rate      | 0.008         | Baseline learning from task practice.                     |
| Knowledge learning rate     | 0.095         | Additional learning from novel support information.       |
| Tie reinforcement rate      | 0.085         | Positive update from useful peer contribution.            |
| Negative tie-update rate    | 0.040         | Negative update from harmful peer contribution.           |
| Unused-tie decay            | 0.0020        | Per-episode multiplicative decay of unused active ties.   |
| Tie-removal threshold       | 0.050         | Tie deleted below this strength.                          |
| Initial tie range           | $[0.42,0.68]$ | Uniform initial active-edge weights.                      |
| New-tie initial strength    | 0.34          | Starting strength after successful external contact.      |
| New-tie formation threshold | 0.010         | Minimum positive peer marginal contribution for new tie.  |
| Support uptake scale        | 0.72          | Scale for beneficial source uptake.                       |
| Harmful-advice uptake scale | 0.40          | Scale for uptake of negative source information.          |
| Hybrid coordination burden  | 0.055         | Burden from combining AI and peer support.                |
| Hybrid conflict scale       | 0.20          | Penalty for disagreement between AI and peer information. |
| Quality noise SD            | 0.024         | Stochastic noise added to final task quality.             |
| Reference seed              | 20260822      | Base configuration seed recorded with results.            |

Learning, network-update, and outcome parameters.

</div>

## Task generation

The three task prototypes are $$\begin{aligned}
\mathbf{p}_{S}&=(0.66,0.22,0.12),\\
\mathbf{p}_{C}&=(0.22,0.63,0.15),\\
\mathbf{p}_{I}&=(0.34,0.29,0.37),
\end{aligned}$$ with ambiguity centers $(0.24,0.66,0.48)$. Each agent
receives a task family according to the task mix. Conditional on family
$f$, the requirement vector is drawn from a Dirichlet distribution with
concentration vector $$\boldsymbol\alpha_f=7\mathbf{p}_f+0.70,$$ then
normalized by construction. Ambiguity is a bounded normal draw around
the corresponding family center with SD 0.11. Difficulty is drawn from a
Beta distribution with mean 0.58 and concentration 10, bounded to
$[0.08,0.96]$. The task-specific success threshold is
$$h=\operatorname{clip}(0.46+0.22d+0.06u,\,0.48,\,0.76).$$

Task similarity to the three anchors is based on cosine similarity and a
softmax with temperature 0.090. These continuous weights determine how
strongly experience on the current task updates each anchor memory.

## AI availability and access cost

For individual access $A_i\in[0,1]$, the saturating access
transformation is
$$s(A_i)=\frac{1-\exp(-kA_i)}{1-\exp(-k)},\qquad k=2.0,$$ and AI
availability probability is
$$p_i^{AI}=p_{\min}+(p_{\max}-p_{\min})s(A_i),$$ with $p_{\min}=0.70$
and $p_{\max}=0.97$. AI usage friction is
$$C_i^{AI}=c_A(1-A_i)^{1.65}+c_u u(1-v_i),$$ where $c_A=0.20$ and
$c_u=0.14$.

## Peer availability, search, and partner selection

Each existing neighbor independently responds with probability 0.80.
External search occurs with probability
$$p_i^{search}=0.12(0.45+0.55s_i).$$ External candidates are weighted by
friend-of-friend connectivity when such information is available.

For a candidate peer $j$, partner-selection propensity is
$$Z_{ij}=0.50\widehat E_{ij}+0.25t_{ij}^{P}+0.25w_{ij}-C_{ij}^{search},$$
where $\widehat E_{ij}$ is task-weighted perceived expertise. A softmax
with temperature $\max(0.20,0.78\tau)$ determines the sampled partner.

## Route choice

Peer route cost is $$C_i^{P}=c_P(1-s_i)+c_W(1-w_i^{\max}),$$ where
$c_P=0.22$, $c_W=0.10$, and $w_i^{\max}$ is the strongest currently
reachable tie. Hybrid integration ability for route-cost purposes is
$$I_i^{C}=\operatorname{clip}(0.36v_i+0.24a_i+0.22s_i+0.18V,0,1),$$ with
environmental verification support $V=0.55$. Hybrid cost is
$$C_i^{H}=0.52C_i^{P}+0.52C_i^{AI}+0.12(1-I_i^{C}).$$ These costs enter
Eq. <a href="#eq:oddutility" data-reference-type="ref"
data-reference="eq:oddutility">[eq:oddutility]</a>. Trust for Hybrid is
the mean of AI trust and the prospective peer trust. Route expectations
and local social signals are blended across task anchors using current
task-similarity weights.

## AI support

The reference AI capability profile is $\mathbf{g}=(0.94,0.64,0.76)$. At
current reliability $R$, the expected capability vector moves between an
uninformative floor 0.12 and the profile:
$$\bar{\mathbf g}=0.12+R(\mathbf g-0.12).$$ Ambiguity scales the vector
by $(1-0.16u)$, after which Gaussian noise with SD 0.085 is added and
values are clipped to $[0.01,0.99]$.

AI information affects the focal agent’s temporary effective skill
through beneficial and harmful uptake. Beneficial uptake is scaled by AI
literacy and verification; harmful uptake is filtered by verification
and environmental verification support. Thus source quality and source
use are represented separately.

## Peer support

For peer $j$, communication efficiency is based on focal sociability,
peer sociability, current tie strength, and partner-specific trust:
$$r_{ij}=0.29s_i+0.29s_j+0.24w_{ij}+0.18t_{ij}^{P}.$$ This score is
mapped to the interval $[0.28,0.80]$, reduced by task ambiguity, and
used to move communicated peer capability from the focal agent’s own
capability toward the peer’s latent capability. Communication noise
increases when transfer efficiency is low. The received peer information
is therefore both relationship-dependent and imperfect.

## Hybrid support

Hybrid support integrates AI and peer information. Integration ability
is
$$I_i^{H}=\operatorname{clip}(0.18+0.28v_i+0.26a_i+0.16s_i+0.12V,0.10,0.96).$$
The integrated source vector interpolates between the average of AI and
peer information and the dimension-wise better source. Disagreement is
penalized by
$$0.20(1-I_i^{H})\lvert\mathbf g_i^{AI}-\mathbf g_{ij}^{P}\rvert.$$
Hybrid coordination burden declines when the task requirement vector is
more multidimensional, using normalized entropy of $\mathbf r$.

## Task quality and support gain

For temporary effective skill vector $\mathbf{x}$, requirement vector
$\mathbf r$, difficulty $d$, confidence $c_i$, and extra burden $b$,
latent quality is
$$Q^{*}=\sigma\left(3.35(\mathbf{x}\cdot\mathbf r-0.50)-2.20(d+b-0.50)+0.22(c_i-0.50)\right).$$
Gaussian noise with SD 0.024 is added and the final quality is clipped
to $[0,1]$. Solo quality $Q_i^{solo}$ is computed from the agent’s own
skill vector. Marginal support gain is $$G_i=Q_i-Q_i^{solo}.$$ This
quantity is used in local observational learning and, for peer
interaction, social-tie updating.

## Experience and trust updates

For the selected route, each task-anchor expectation is updated by
$$V_{iam}'=V_{iam}+\alpha_a(Q_i-V_{iam}),$$ where
$$\alpha_a=0.32(0.25+0.75\omega_a)$$ and $\omega_a$ is current task
similarity to anchor $a$. AI trust follows
$$t_i^{AI'}=\operatorname{clip}\left[t_i^{AI}+0.12(S_i^{AI}-t_i^{AI}),0.01,0.99\right],$$
where $S_i^{AI}$ is the task-relevant AI source quality.
Partner-specific peer trust uses the analogous update toward experienced
peer source quality. Partner expertise expectations are updated with the
same task-similarity-weighted route-learning rate.

## Knowledge and confidence updates

All tasks generate a small practice component:
$$\Delta\mathbf{k}_i^{practice}=0.008\,Q_i\,\mathbf r\odot(1-\mathbf k_i).$$
Novel information is the positive component of the difference between
effective supported capability and the agent’s pre-episode skill.
Additional source learning is
$$\Delta\mathbf{k}_i^{support}=0.095\,A_i^{abs}\,Q_i\,\mathbf n_i\odot\mathbf r,$$
where $A_i^{abs}$ depends on the selected route and agent attributes.
Confidence updates slowly toward realized quality at rate 0.030.

## Network updating

If peer $j$ contributes marginally to the focal outcome, an existing tie
updates according to $$\Delta w_{ij}=\begin{cases}
0.085\,G_{ij}^{P}(1-w_{ij}), & G_{ij}^{P}\ge 0,\\
0.040\,G_{ij}^{P}w_{ij}, & G_{ij}^{P}<0.
\end{cases}$$ A discovered peer with $G_{ij}^{P}>0.010$ can form a new
tie initialized at 0.34. Every unused active tie is multiplied by
$(1-0.002)$ each episode and removed below 0.050.

## Local observational learning

For each focal agent, existing neighbors are observed independently with
probability 0.45. For each route and task anchor, observed neighbors’
marginal support gains are averaged using tie strength and task
similarity as weights. The corresponding local route signal then moves
toward this observed value at rate 0.13.

## Output measures

The model records:

- Peer-first, AI-first, and Hybrid shares;

- human-interaction and AI-use rates;

- mean task quality and success rate;

- mean knowledge, knowledge Gini, performance Gini, and initial–final
  knowledge rank correlation;

- mean AI trust and mean peer trust;

- network density, mean weighted degree, mean active-tie strength,
  recent active-tie fraction, edge-retention ratio, new-edge ratio, and
  strong-tie fraction;

- mean unique peer partners;

- mean marginal peer contribution and AI-availability probability.

The final summary uses a tail window of at least 20 episodes, depending
on the experiment.

# Simulation experiments

## Reference system and human-only benchmark

The AI-enabled reference system and the human-only benchmark each use 30
paired replications with common seed construction. The human-only
benchmark sets AI availability to zero while preserving agent, task,
peer, learning, and network mechanisms. Two-sided 95% Student-$t$
confidence intervals summarize replication uncertainty.

## Task–route mechanism analysis

Twenty replications record task-level events and agent summaries.
Outputs include task-by-route quality, marginal support gain,
task-specific choice shares, early-versus-late choice adaptation,
requirement-dimension outcomes, and experience-to-next-choice
reinforcement.

## Environmental factorial

The factorial experiment contains $3\times3\times2\times2=36$ conditions
with 20 replications per condition:

<div class="center">

| Factor                    | Levels           |
|:--------------------------|:-----------------|
| Mean AI access            | 0.35, 0.60, 0.85 |
| AI reliability            | 0.55, 0.74, 0.90 |
| Verification support      | 0.30, 0.70       |
| Peer-response probability | 0.60, 0.90       |

</div>

The design estimates how opportunity conditions shape interaction
composition, task outcomes, learning, trust, and social relations.

## AI reliability shock experiment

Three conditions use 30 replications each. Reliability starts at 0.86.
The temporary-shock condition drops to 0.48 halfway through the
150-episode horizon and returns to 0.86 after three quarters of the
horizon. The persistent-shock condition drops to 0.48 at mid-horizon and
remains there. Mean AI access is 0.70 in the shock experiment. Paired
late-window contrasts compare post-recovery behavior to the stable
condition.

## Mechanism sensitivity

Sixteen behavioral and update parameters are varied by $\pm20\%$ around
their reference values using paired seeds and 20 replications per
setting:

- expected-quality weight;

- local social-learning weight;

- choice temperature;

- mode-value learning rate;

- trust learning rate;

- knowledge learning rate;

- tie reinforcement rate;

- tie decay;

- external peer-search probability;

- hybrid coordination burden;

- task-similarity temperature;

- peer-expertise visibility;

- peer communication-noise SD;

- maximum peer-transfer efficiency;

- AI ambiguity verification cost;

- AI access saturation.

# Implementation and reproducibility

The model is implemented in Python. The repository contains:

- `model.py`: entities, task generation, route choice, outcome,
  learning, trust, and network mechanisms;

- `experiments.py`: reference, task-mechanism, factorial, shock, and
  sensitivity designs;

- `run_analysis.py`: complete analysis pipeline;

- `precheck.py`: lightweight diagnostic run;

- `figures.py` and `regenerate_figures.py`: figure generation from saved
  CSV results;

- `results/`: compact replication-level and summary outputs;

- `docs/`: this ODD protocol, full parameter documentation, and
  reproducibility instructions.

Full analysis is executed with

    python run_analysis.py

and figures can be regenerated with

    python regenerate_figures.py

Experiment functions are checkpoint-aware. Replication seed $r$ is
generated as $20260821+1009r$, enabling common-random-number paired
comparisons. The large event-level file `task_mode_agent_events.csv` is
generated locally because its full reference-study size exceeds the
standard GitHub per-file limit; compact summaries used for the reported
results are included in the repository.

# Model boundaries and limitations

The model is designed for mechanism exploration. Parameter values are
transparent reference settings rather than empirical estimates. Task
dimensions represent abstract requirement components rather than named
occupational skills. AI reliability is represented as a controllable
technological property, whereas real systems may exhibit domain-specific
and temporally correlated error. Peer communication is dyadic and
abstract, and institutional constraints such as hierarchy, incentives,
workload, and formal team structure are not represented. Agents adapt
through simplified reinforcement-like rules rather than richer cognitive
models. Social ties represent repeated interaction opportunities and
relationship strength, not the full multidimensional meaning of real
social relationships.

These boundaries limit direct numerical generalization, but they make
the feedback mechanisms inspectable. Empirical extensions can calibrate
task profiles, AI performance, peer-transfer efficiency, trust learning,
and network dynamics using field or laboratory data.

# References

<div class="thebibliography">

9 Grimm, V., Berger, U., Bastiansen, F., Eliassen, S., Ginot, V., Giske,
J., Goss-Custard, J., Grand, T., Heinz, S.K., Huse, G., et al.: A
standard protocol for describing individual-based and agent-based
models. Ecological Modelling **198**(1–2), 115–126 (2006) Grimm, V.,
Berger, U., DeAngelis, D.L., Polhill, J.G., Giske, J., Railsback, S.F.:
The ODD protocol: A review and first update. Ecological Modelling
**221**(23), 2760–2768 (2010) Grimm, V., Railsback, S.F., Vincenot,
C.E., Berger, U., Gallagher, C., DeAngelis, D.L., Edmonds, B., Ge, J.,
Giske, J., Groeneveld, J., et al.: The ODD protocol for describing
agent-based and other simulation models: A second update to improve
clarity, replication, and structural realism. Journal of Artificial
Societies and Social Simulation **23**(2), 7 (2020) Müller, B., Bohn,
F., Dreßler, G., Groeneveld, J., Klassert, C., Martin, R., Schlueter,
M., Schulze, J., Weise, H., Schwarz, N.: Describing human decisions in
agent-based models – ODD+D, an extension of the ODD protocol.
Environmental Modelling & Software **48**, 37–48 (2013) Epstein, J.M.:
Generative Social Science: Studies in Agent-Based Computational
Modeling. Princeton University Press (2006) McFadden, D.: Conditional
logit analysis of qualitative choice behavior. In: Zarembka, P. (ed.)
Frontiers in Econometrics, pp. 105–142. Academic Press, New York (1974)
Simon, H.A.: A behavioral model of rational choice. Quarterly Journal of
Economics **69**(1), 99–118 (1955) Watts, D.J., Strogatz, S.H.:
Collective dynamics of small-world networks. Nature **393**, 440–442
(1998)

</div>