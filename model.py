from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import networkx as nx

MODE_PEER = 0
MODE_AI = 1
MODE_HYBRID = 2
MODE_NAMES = {MODE_PEER: "Peer-first", MODE_AI: "AI-first", MODE_HYBRID: "Hybrid"}

TASK_STRUCTURED = 0
TASK_CONTEXTUAL = 1
TASK_INTEGRATIVE = 2
TASK_NAMES = {
    TASK_STRUCTURED: "Structured",
    TASK_CONTEXTUAL: "Contextual",
    TASK_INTEGRATIVE: "Integrative",
}


@dataclass
class ModelConfig:
    # ------------------------------ scale ---------------------------------
    n_agents: int = 180
    episodes: int = 150

    # --------------------------- AI environment ----------------------------
    ai_enabled: bool = True
    ai_reliability: float = 0.78
    ai_access_mean: float = 0.60
    ai_access_sd: float = 0.16
    verification_support: float = 0.55

    # Access is intentionally NOT interpreted as a direct probability of having AI.
    # It shapes both availability and friction through bounded, saturating mappings.
    ai_availability_floor: float = 0.70
    ai_availability_ceiling: float = 0.97
    ai_access_saturation: float = 2.0

    # Multidimensional AI capability profile. These are technological capabilities,
    # not behavioural mode bonuses. Reliability scales the realised profile.
    ai_capability_profile: Tuple[float, float, float] = (0.94, 0.64, 0.76)
    ai_capability_noise_sd: float = 0.085
    ai_output_concentration: float = 22.0
    ai_ambiguity_penalty: float = 0.16

    # -------------------------- peer environment ---------------------------
    network_density: float = 0.055
    rewiring_probability: float = 0.10
    peer_response_probability: float = 0.80
    external_peer_search_probability: float = 0.12
    peer_expertise_visibility: float = 0.30
    social_observation_rate: float = 0.45

    # Peer help is useful only when advice can actually be communicated and the
    # contacted person responds. Expertise is therefore not transferred perfectly.
    peer_communication_noise_sd: float = 0.095
    peer_ambiguity_penalty: float = 0.10
    peer_min_transfer_efficiency: float = 0.28
    peer_max_transfer_efficiency: float = 0.80

    # ----------------------------- tasks ----------------------------------
    task_mix: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    difficulty_mean: float = 0.58
    difficulty_concentration: float = 10.0
    task_similarity_temperature: float = 0.090

    # ------------------------ heterogeneous agents -------------------------
    knowledge_mean: float = 0.52
    ai_literacy_mean: float = 0.50
    sociability_mean: float = 0.55
    verification_mean: float = 0.52
    confidence_mean: float = 0.50
    trait_sd: float = 0.14

    # -------------------------- common choice rule -------------------------
    expected_quality_weight: float = 3.60
    trust_weight: float = 0.12
    local_social_learning_weight: float = 0.55
    choice_temperature: float = 0.31
    exploration_floor: float = 0.025
    peer_base_cost: float = 0.22
    ai_base_cost: float = 0.20
    # Ambiguous tasks require extra checking when AI is used. The burden is lower
    # for agents with stronger verification skill/support; it is not tied to a task label.
    ai_ambiguity_verification_cost: float = 0.14
    hybrid_extra_cost: float = 0.12
    weak_network_search_cost: float = 0.10

    # ---------------------- learning and adaptation ------------------------
    mode_value_learning_rate: float = 0.32
    trust_learning_rate: float = 0.12
    local_memory_learning_rate: float = 0.13
    confidence_learning_rate: float = 0.030

    # Knowledge has a general practice component plus source-specific learning.
    practice_learning_rate: float = 0.008
    knowledge_learning_rate: float = 0.095

    # ------------------------ network co-evolution -------------------------
    tie_reinforcement_rate: float = 0.085
    tie_negative_update_rate: float = 0.040
    tie_decay: float = 0.0020
    tie_removal_threshold: float = 0.050
    initial_tie_low: float = 0.42
    initial_tie_high: float = 0.68
    new_tie_initial_strength: float = 0.34
    new_tie_formation_threshold: float = 0.010

    # --------------------------- outcome rule ------------------------------
    support_uptake_scale: float = 0.72
    harmful_advice_uptake_scale: float = 0.40
    hybrid_coordination_burden: float = 0.055
    hybrid_conflict_scale: float = 0.20
    quality_noise_sd: float = 0.024

    # Optional dynamic AI reliability schedule: {episode_1_based: new_value}
    reliability_schedule: Optional[Dict[int, float]] = None

    # Detailed task records are expensive; enable only for dedicated analyses.
    record_task_details: bool = False
    seed: int = 2026


class AIPeerInteractionModel:
    """Repeated AI/peer/hybrid problem solving with endogenous learning and ties.

    Core design principle
    ---------------------
    All AI-enabled runs use one common decision and update architecture. An agent
    sees a task, forms expected outcomes for the three support modes, chooses
    probabilistically, observes the realised outcome of the selected mode, and
    updates experience, trust, knowledge and social ties. Peer availability,
    imperfect expertise perception and communication, task-specific experience,
    and multidimensional task-resource matching jointly determine comparative
    advantage across Peer-first, AI-first and Hybrid support.
    """

    # Three overlapping task anchors. They are requirement profiles, not labels for
    # which support mode should succeed.
    TASK_PROTOTYPES = np.array(
        [
            [0.66, 0.22, 0.12],
            [0.22, 0.63, 0.15],
            [0.34, 0.29, 0.37],
        ],
        dtype=float,
    )
    TASK_AMBIGUITY_BASE = np.array([0.24, 0.66, 0.48], dtype=float)

    def __init__(self, config: ModelConfig):
        self.cfg = config
        self.rng = np.random.default_rng(int(config.seed))
        self.n = int(config.n_agents)
        self.episodes = int(config.episodes)
        self.current_ai_reliability = float(config.ai_reliability)
        self.history = []
        self.task_mode_records = []
        self._validate_config()
        self._init_agents()
        self._init_network()

    # ======================================================================
    # initialization
    # ======================================================================
    def _validate_config(self):
        c = self.cfg
        if c.n_agents < 20:
            raise ValueError("n_agents should be at least 20")
        if c.episodes < 5:
            raise ValueError("episodes should be at least 5")
        tm = np.asarray(c.task_mix, dtype=float)
        if len(tm) != 3 or np.any(tm < 0) or tm.sum() <= 0:
            raise ValueError("task_mix must contain three nonnegative values")
        if len(c.ai_capability_profile) != 3:
            raise ValueError("ai_capability_profile must contain three values")
        for name in (
            "ai_reliability", "ai_access_mean", "verification_support",
            "network_density", "rewiring_probability", "peer_response_probability",
            "external_peer_search_probability", "peer_expertise_visibility",
            "social_observation_rate", "ai_availability_floor",
            "ai_availability_ceiling",
        ):
            val = float(getattr(c, name))
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if c.ai_availability_floor > c.ai_availability_ceiling:
            raise ValueError("ai_availability_floor cannot exceed ai_availability_ceiling")

    def _bounded_normal(self, mean, sd, low=0.03, high=0.97, size=None):
        return np.clip(self.rng.normal(mean, sd, size=size), low, high)

    def _init_agents(self):
        c, n = self.cfg, self.n

        # General ability + domain-specific deviations. This yields multidimensional
        # skill profiles without assigning any behavioural orientation.
        general = self._bounded_normal(c.knowledge_mean, 0.10, 0.18, 0.88, n)
        domain_noise = self.rng.normal(0.0, 0.15, size=(n, 3))
        self.skills = np.clip(general[:, None] + domain_noise, 0.06, 0.96)

        sd = c.trait_sd
        self.ai_literacy = self._bounded_normal(c.ai_literacy_mean, sd, 0.05, 0.95, n)
        self.sociability = self._bounded_normal(c.sociability_mean, sd, 0.05, 0.95, n)
        self.verification = self._bounded_normal(c.verification_mean, sd, 0.05, 0.95, n)
        self.confidence = self._bounded_normal(c.confidence_mean, sd, 0.05, 0.95, n)
        self.ai_access = self._bounded_normal(c.ai_access_mean, c.ai_access_sd, 0.02, 0.98, n)
        if not c.ai_enabled:
            self.ai_access[:] = 0.0

        # Beliefs are indexed by three task anchors but evaluated through continuous
        # similarity weights. Priors are close and neutral.
        # Learned values are task-specific expected realised qualities for each mode.
        # Agents learn only from the mode they actually selected; unchosen
        # counterfactuals remain unknown.
        self.mode_value = np.clip(
            0.50 + self.rng.normal(0.0, 0.015, size=(n, 3, 3)), 0.35, 0.65
        )
        # Neighbour observations retain marginal support value, so social learning
        # does not merely reward generally high-ability neighbours.
        self.local_mode_signal = np.zeros((n, 3, 3), dtype=float)

        self.trust_ai = self._bounded_normal(0.50, 0.07, 0.25, 0.75, n)
        if not c.ai_enabled:
            self.trust_ai[:] = 0.0

        self.last_mode = np.full(n, MODE_PEER, dtype=int)
        self.last_quality = np.full(n, 0.50, dtype=float)
        self.last_task_weights = np.full((n, 3), 1 / 3, dtype=float)
        self.mode_counts = np.zeros((n, 3), dtype=np.int32)
        self.cumulative_quality = np.zeros(n, dtype=float)
        self.cumulative_success = np.zeros(n, dtype=float)

        self.initial_skills = self.skills.copy()
        self.initial_ai_literacy = self.ai_literacy.copy()
        self.initial_sociability = self.sociability.copy()
        self.initial_verification = self.verification.copy()
        self.initial_confidence = self.confidence.copy()
        self.initial_ai_access = self.ai_access.copy()

    def _init_network(self):
        c, n = self.cfg, self.n
        k = max(4, int(round(c.network_density * n)))
        if k % 2:
            k += 1
        max_even = n - 2 if (n - 2) % 2 == 0 else n - 3
        k = min(k, max_even)
        g = nx.connected_watts_strogatz_graph(
            n, k, c.rewiring_probability, tries=200, seed=int(c.seed)
        )

        self.tie = np.zeros((n, n), dtype=np.float32)
        self.peer_trust = np.full((n, n), 0.50, dtype=np.float32)
        self.peer_expected = np.full((n, n, 3), 0.50, dtype=np.float32)

        # Initial expertise impressions are imperfect. Existing contacts reveal some
        # information about task-relevant capability without giving agents exact skill.
        proto_expertise = self.skills @ self.TASK_PROTOTYPES.T
        visibility = float(c.peer_expertise_visibility)
        for i, j in g.edges():
            w = float(self.rng.uniform(c.initial_tie_low, c.initial_tie_high))
            self.tie[i, j] = self.tie[j, i] = w
            self.peer_trust[i, j] = float(np.clip(self.rng.normal(0.50, 0.055), 0.25, 0.75))
            self.peer_trust[j, i] = float(np.clip(self.rng.normal(0.50, 0.055), 0.25, 0.75))
            for observer, target in ((i, j), (j, i)):
                noisy = np.clip(proto_expertise[target] + self.rng.normal(0, 0.10, 3), 0.05, 0.95)
                self.peer_expected[observer, target] = (
                    (1.0 - visibility) * 0.50 + visibility * noisy
                )

        self.initial_edge_mask = (self.tie > 0).copy()
        self.initial_edge_count = int(np.count_nonzero(np.triu(self.initial_edge_mask, 1)))
        self.initial_mean_tie = self._mean_tie_strength()
        self.last_used_episode = np.zeros((n, n), dtype=np.int32)
        self.peer_interaction_count = np.zeros((n, n), dtype=np.int32)

    # ======================================================================
    # numerical helpers
    # ======================================================================
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))

    @staticmethod
    def _softmax_1d(u, temperature):
        z = np.asarray(u, dtype=float) / max(float(temperature), 1e-6)
        z -= np.max(z)
        e = np.exp(z)
        return e / e.sum() if e.sum() > 0 else np.full_like(e, 1.0 / len(e))

    @staticmethod
    def _gini(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        if np.min(x) < 0:
            x = x - np.min(x)
        total = x.sum()
        if total <= 1e-12:
            return 0.0
        x = np.sort(x)
        n = x.size
        idx = np.arange(1, n + 1)
        return float((2.0 * np.sum(idx * x) / (n * total)) - (n + 1.0) / n)

    def _mean_tie_strength(self):
        vals = self.tie[np.triu(self.tie > 0, 1)]
        return float(vals.mean()) if vals.size else 0.0

    def _network_density(self):
        m = int(np.count_nonzero(np.triu(self.tie > 0, 1)))
        possible = self.n * (self.n - 1) / 2
        return float(m / possible) if possible else 0.0

    def _weighted_degree(self):
        return self.tie.sum(axis=1).astype(float)

    def _mean_peer_trust(self):
        mask = self.tie > 0
        return float(self.peer_trust[mask].mean()) if np.any(mask) else 0.0

    def _task_similarity_weights(self, req):
        """Smooth task-to-anchor weights; no discrete if/else behaviour rule."""
        req = np.asarray(req, dtype=float)
        # cosine similarity is robust because all requirement vectors sum to one.
        p = self.TASK_PROTOTYPES
        sim = (p @ req) / (np.linalg.norm(p, axis=1) * np.linalg.norm(req) + 1e-12)
        return self._softmax_1d(sim, self.cfg.task_similarity_temperature)

    def _blended_anchor_value(self, arr3, weights):
        return float(np.dot(np.asarray(arr3, dtype=float), np.asarray(weights, dtype=float)))

    # ======================================================================
    # task generation
    # ======================================================================
    def _generate_tasks(self):
        c = self.cfg
        mix = np.asarray(c.task_mix, dtype=float)
        mix /= mix.sum()
        family = self.rng.choice(3, size=self.n, p=mix)

        req = np.zeros((self.n, 3), dtype=float)
        ambiguity = np.zeros(self.n, dtype=float)
        task_weights = np.zeros((self.n, 3), dtype=float)
        for t in range(3):
            ids = np.where(family == t)[0]
            if ids.size == 0:
                continue
            # Dirichlet concentration allows meaningful within-family heterogeneity.
            alpha = 7.0 * self.TASK_PROTOTYPES[t] + 0.70
            req[ids] = self.rng.dirichlet(alpha, size=ids.size)
            ambiguity[ids] = np.clip(
                self.rng.normal(self.TASK_AMBIGUITY_BASE[t], 0.11, size=ids.size), 0.04, 0.96
            )
            for i in ids:
                task_weights[i] = self._task_similarity_weights(req[i])

        mean = np.clip(c.difficulty_mean, 0.08, 0.92)
        conc = max(2.5, float(c.difficulty_concentration))
        difficulty = np.clip(
            self.rng.beta(mean * conc, (1 - mean) * conc, size=self.n), 0.08, 0.96
        )
        success_threshold = np.clip(0.46 + 0.22 * difficulty + 0.06 * ambiguity, 0.48, 0.76)
        return family, req, ambiguity, difficulty, success_threshold, task_weights

    # ======================================================================
    # access, peer opportunity and mode choice
    # ======================================================================
    def _ai_availability_probability(self, access):
        c = self.cfg
        if not c.ai_enabled:
            return 0.0
        x = float(np.clip(access, 0.0, 1.0))
        k = max(0.01, c.ai_access_saturation)
        sat = (1.0 - np.exp(-k * x)) / (1.0 - np.exp(-k))
        return float(c.ai_availability_floor + (c.ai_availability_ceiling - c.ai_availability_floor) * sat)

    def _ai_access_cost(self, access, ambiguity=0.0, verification=0.5):
        # Friction falls quickly from low to moderate access, then saturates. A
        # separate verification burden arises on ambiguous tasks because AI output
        # requires more checking; agents with better verification capability face less
        # of that burden.
        x = float(np.clip(access, 0.0, 1.0))
        base = self.cfg.ai_base_cost * ((1.0 - x) ** 1.65)
        verify_burden = self.cfg.ai_ambiguity_verification_cost * float(ambiguity) * (1.0 - float(verification))
        return float(base + verify_burden)

    def _available_peer_candidates(self, i):
        tie_view = getattr(self, "_tie_for_episode", self.tie)
        neighbors = np.flatnonzero(tie_view[i] > 0)
        if neighbors.size:
            reachable = neighbors[self.rng.random(neighbors.size) < self.cfg.peer_response_probability]
        else:
            reachable = np.empty(0, dtype=int)

        search_prob = self.cfg.external_peer_search_probability * (0.45 + 0.55 * self.sociability[i])
        if self.rng.random() < search_prob:
            non_neighbors = np.flatnonzero((tie_view[i] <= 0) & (np.arange(self.n) != i))
            if non_neighbors.size:
                # Efficient friend-of-friend search: count two-step connections only
                # through the focal agent's current neighbours, rather than a full
                # n-by-n matrix product at every decision.
                if neighbors.size:
                    fof_counts = np.count_nonzero(tie_view[neighbors] > 0, axis=0).astype(float)
                    scores = fof_counts[non_neighbors]
                else:
                    scores = np.zeros(non_neighbors.size, dtype=float)
                if np.all(scores <= 0):
                    external = int(self.rng.choice(non_neighbors))
                else:
                    scores = scores + 0.05
                    scores /= scores.sum()
                    external = int(self.rng.choice(non_neighbors, p=scores))
                reachable = np.unique(np.append(reachable, external)).astype(int)
        return reachable

    def _peer_expected_for_task(self, i, candidates, task_weights):
        if candidates.size == 0:
            return np.empty(0, dtype=float)
        vals = self.peer_expected[i, candidates].astype(float)
        return vals @ np.asarray(task_weights, dtype=float)

    def _peer_prospect(self, i, candidates, task_weights):
        if candidates.size == 0:
            return 0.42, 0.42, 0.0
        expected = self._peer_expected_for_task(i, candidates, task_weights)
        trust = self.peer_trust[i, candidates].astype(float)
        tie_view = getattr(self, "_tie_for_episode", self.tie)
        ties = tie_view[i, candidates].astype(float)
        score = 0.58 * expected + 0.24 * trust + 0.18 * ties
        j = int(np.argmax(score))
        return float(expected[j]), float(trust[j]), float(ties[j])

    def _choose_peer(self, i, candidates, task_weights):
        if candidates.size == 0:
            raise RuntimeError("Peer support was selected without an available peer candidate")
        expected = self._peer_expected_for_task(i, candidates, task_weights)
        trust = self.peer_trust[i, candidates].astype(float)
        tie_view = getattr(self, "_tie_for_episode", self.tie)
        ties = tie_view[i, candidates].astype(float)
        penalty = np.where(ties > 0, 0.0, self.cfg.weak_network_search_cost)
        # Agents choose from perceived expertise, trust and relationship strength;
        # expertise is deliberately imperfect, so the objectively best peer is not
        # selected mechanically.
        utility = 0.50 * expected + 0.25 * trust + 0.25 * ties - penalty
        probs = self._softmax_1d(utility, max(0.20, 0.78 * self.cfg.choice_temperature))
        partner = int(self.rng.choice(candidates, p=probs))
        return partner, bool(tie_view[i, partner] <= 0)

    def _mode_choice(self, i, candidates, task_weights, ai_available, ambiguity=0.0):
        c = self.cfg
        peer_available = bool(candidates.size > 0)
        peer_expected, peer_t, strongest_tie = self._peer_prospect(i, candidates, task_weights)

        peer_cost = c.peer_base_cost * (1.0 - self.sociability[i])
        peer_cost += c.weak_network_search_cost * (1.0 - strongest_tie)
        ai_cost = self._ai_access_cost(self.ai_access[i], ambiguity=ambiguity, verification=self.verification[i])
        integration_ability = np.clip(
            0.36 * self.verification[i] + 0.24 * self.ai_literacy[i]
            + 0.22 * self.sociability[i] + 0.18 * c.verification_support,
            0.0, 1.0,
        )
        hybrid_cost = 0.52 * peer_cost + 0.52 * ai_cost + c.hybrid_extra_cost * (1.0 - integration_ability)
        costs = np.array([peer_cost, ai_cost, hybrid_cost], dtype=float)

        # The same choice equation is used for all three modes. The task matters
        # through learned marginal gains for similar past tasks, not through a
        # task-name-specific bonus.
        expected = self.mode_value[i].T @ np.asarray(task_weights, dtype=float)
        local = self.local_mode_signal[i].T @ np.asarray(task_weights, dtype=float)
        source_trust = np.array([
            peer_t if peer_available else 0.0,
            self.trust_ai[i] if ai_available else 0.0,
            0.5 * (peer_t + self.trust_ai[i]) if (ai_available and peer_available) else 0.0,
        ])
        utility = (
            c.expected_quality_weight * expected
            + c.trust_weight * source_trust
            + c.local_social_learning_weight * local
            - costs
        )

        available = np.array([
            peer_available,
            bool(ai_available and c.ai_enabled),
            bool(peer_available and ai_available and c.ai_enabled),
        ], dtype=bool)
        if not np.any(available):
            raise RuntimeError("No support mode is available for this task")
        utility[~available] = -1e9
        probs = self._softmax_1d(utility, c.choice_temperature)

        # Exploration is applied only over currently feasible modes.
        if c.exploration_floor > 0:
            ids = np.flatnonzero(available)
            eps_total = min(float(c.exploration_floor) * len(ids), 0.12)
            uniform = np.zeros(3, dtype=float)
            uniform[ids] = 1.0 / len(ids)
            probs = (1.0 - eps_total) * probs + eps_total * uniform
            probs[~available] = 0.0
            probs /= probs.sum()

        mode = int(self.rng.choice(3, p=probs))
        return mode, probs

    # ======================================================================
    # outcome mechanism: multidimensional task-resource matching
    # ======================================================================
    def _latent_quality(self, skill_vec, req, difficulty, confidence=0.50, extra_burden=0.0):
        match = float(np.dot(np.asarray(skill_vec), np.asarray(req)))
        latent = 3.35 * (match - 0.50) - 2.20 * (difficulty + extra_burden - 0.50)
        latent += 0.22 * (confidence - 0.50)
        return float(self._sigmoid(latent))

    def _solo_quality(self, i, req, difficulty):
        skills_view = getattr(self, "_skills_for_episode", self.skills)
        return self._latent_quality(skills_view[i], req, difficulty, self.confidence[i])

    def _ai_capability_vector(self, ambiguity):
        c = self.cfg
        base = np.asarray(c.ai_capability_profile, dtype=float)
        # Reliability scales the capability vector toward an uninformative floor.
        mean_vec = 0.12 + self.current_ai_reliability * (base - 0.12)
        mean_vec *= 1.0 - c.ai_ambiguity_penalty * float(ambiguity)
        mean_vec = np.clip(mean_vec, 0.04, 0.96)
        noisy = mean_vec + self.rng.normal(0.0, c.ai_capability_noise_sd, size=3)
        return np.clip(noisy, 0.01, 0.99)

    def _peer_capability_vector(self, i, j, ambiguity):
        """Task-relevant capability actually transmitted from peer j to agent i.

        A contacted peer does not transmit their full latent skill vector perfectly.
        Communication efficiency depends on both agents and their tie; ambiguity makes
        transfer harder. This is a symmetric realism constraint, not a mode penalty.
        """
        c = self.cfg
        skills_view = getattr(self, "_skills_for_episode", self.skills)
        own = np.asarray(skills_view[i], dtype=float)
        true_peer = np.asarray(skills_view[j], dtype=float)
        tie_view = getattr(self, "_tie_for_episode", self.tie)
        tie = float(tie_view[i, j])
        raw = 0.29 * self.sociability[i] + 0.29 * self.sociability[j] + 0.24 * tie + 0.18 * self.peer_trust[i, j]
        eff = c.peer_min_transfer_efficiency + (c.peer_max_transfer_efficiency - c.peer_min_transfer_efficiency) * np.clip(raw, 0.0, 1.0)
        eff *= 1.0 - c.peer_ambiguity_penalty * float(ambiguity)
        eff = float(np.clip(eff, c.peer_min_transfer_efficiency, c.peer_max_transfer_efficiency))
        noise_sd = c.peer_communication_noise_sd * (1.15 - 0.75 * eff)
        communicated = true_peer + self.rng.normal(0.0, noise_sd, size=3)
        communicated = np.clip(communicated, 0.01, 0.99)
        delivered = own + eff * (communicated - own)
        return np.clip(delivered, 0.01, 0.99)

    def _apply_source_to_skills(self, i, source_vec, source_kind, partner=None):
        """Return a temporary effective skill vector after using a source.

        Better advice can raise effective capability; worse advice can lower it. AI
        literacy / sociability govern uptake, while verification / tie strength limit
        harm. This avoids adding a mode-specific quality bonus directly.
        """
        c = self.cfg
        own = np.asarray(getattr(self, "_skills_for_episode", self.skills)[i], dtype=float)
        delta = np.asarray(source_vec, dtype=float) - own
        if source_kind == "ai":
            positive_uptake = c.support_uptake_scale * (0.25 + 0.55 * self.ai_literacy[i] + 0.20 * self.verification[i])
            harm_filter = np.clip(0.35 + 0.45 * self.verification[i] + 0.20 * c.verification_support, 0, 1)
        else:
            tie_view = getattr(self, "_tie_for_episode", self.tie)
            tie = float(tie_view[i, partner]) if partner is not None else 0.0
            positive_uptake = c.support_uptake_scale * (
                0.34 + 0.34 * self.sociability[i] + 0.18 * tie + 0.14 * self.peer_trust[i, partner]
            )
            harm_filter = np.clip(0.30 + 0.34 * self.sociability[i] + 0.22 * tie + 0.14 * self.peer_trust[i, partner], 0, 1)
        positive_uptake = float(np.clip(positive_uptake, 0.15, 0.95))
        negative_uptake = float(c.harmful_advice_uptake_scale * (1.0 - harm_filter))
        change = np.where(delta >= 0, positive_uptake * delta, negative_uptake * delta)
        return np.clip(own + change, 0.0, 0.995)

    def _realise_outcome(self, i, mode, req, ambiguity, difficulty, partner=None):
        c = self.cfg
        own = np.asarray(getattr(self, "_skills_for_episode", self.skills)[i], dtype=float)
        solo = self._latent_quality(own, req, difficulty, self.confidence[i])

        ai_vec = None
        peer_vec = None
        q_ai_counter = np.nan
        q_peer_counter = np.nan
        ai_source_quality = np.nan
        peer_source_quality = np.nan

        if mode in (MODE_AI, MODE_HYBRID):
            ai_vec = self._ai_capability_vector(ambiguity)
            ai_supported = self._apply_source_to_skills(i, ai_vec, "ai")
            q_ai_counter = self._latent_quality(ai_supported, req, difficulty, self.confidence[i])
            ai_source_quality = float(np.dot(ai_vec, req))

        if mode in (MODE_PEER, MODE_HYBRID):
            if partner is None:
                raise RuntimeError("Peer/Hybrid requires a partner")
            peer_vec = self._peer_capability_vector(i, partner, ambiguity)
            peer_supported = self._apply_source_to_skills(i, peer_vec, "peer", partner=partner)
            q_peer_counter = self._latent_quality(peer_supported, req, difficulty, self.confidence[i])
            peer_source_quality = float(np.dot(peer_vec, req))

        if mode == MODE_AI:
            q = q_ai_counter
            effective_skill = ai_supported
            ai_marginal, peer_marginal = q - solo, np.nan
        elif mode == MODE_PEER:
            q = q_peer_counter
            effective_skill = peer_supported
            peer_marginal, ai_marginal = q - solo, np.nan
        else:
            integration = np.clip(
                0.18 + 0.28 * self.verification[i] + 0.26 * self.ai_literacy[i]
                + 0.16 * self.sociability[i] + 0.12 * c.verification_support,
                0.10, 0.96,
            )
            average_vec = 0.5 * (ai_vec + peer_vec)
            best_by_dimension = np.maximum(ai_vec, peer_vec)
            combined_source = (1.0 - integration) * average_vec + integration * best_by_dimension
            conflict = np.abs(ai_vec - peer_vec)
            conflict_penalty = c.hybrid_conflict_scale * (1.0 - integration) * conflict
            combined_source = np.clip(combined_source - conflict_penalty, 0.01, 0.99)

            # Hybrid uptake uses a balanced integration skill, not a free bonus.
            own_delta = combined_source - own
            good_uptake = c.support_uptake_scale * (0.40 + 0.45 * integration)
            bad_filter = np.clip(0.40 + 0.40 * self.verification[i] + 0.20 * c.verification_support, 0, 1)
            bad_uptake = c.harmful_advice_uptake_scale * (1.0 - bad_filter)
            effective_skill = np.clip(
                own + np.where(own_delta >= 0, good_uptake * own_delta, bad_uptake * own_delta),
                0.0, 0.995,
            )
            # Coordination burden is lower when the task genuinely spans multiple
            # requirement dimensions, because combining distinct sources can then be
            # worth the coordination effort. This uses the continuous requirement
            # vector, never a task label.
            req_arr = np.asarray(req, dtype=float)
            entropy = -float(np.sum(req_arr * np.log(np.clip(req_arr, 1e-12, 1.0)))) / np.log(3.0)
            multidimensional_relief = 1.18 - 0.42 * np.clip(entropy, 0.0, 1.0)
            burden = c.hybrid_coordination_burden * (1.0 - integration) * multidimensional_relief * (0.80 + 0.20 * (1.0 - ambiguity))
            q = self._latent_quality(effective_skill, req, difficulty, self.confidence[i], extra_burden=burden)
            peer_marginal = q - q_ai_counter
            ai_marginal = q - q_peer_counter

        if c.quality_noise_sd > 0:
            q = float(np.clip(q + self.rng.normal(0.0, c.quality_noise_sd), 0.0, 1.0))

        return {
            "quality": float(q),
            "solo_quality": float(solo),
            "effective_skill": np.asarray(effective_skill, dtype=float),
            "ai_capability_vec": None if ai_vec is None else np.asarray(ai_vec, dtype=float),
            "peer_capability_vec": None if peer_vec is None else np.asarray(peer_vec, dtype=float),
            "ai_source_quality": float(ai_source_quality) if np.isfinite(ai_source_quality) else np.nan,
            "peer_source_quality": float(peer_source_quality) if np.isfinite(peer_source_quality) else np.nan,
            "ai_marginal": float(ai_marginal) if np.isfinite(ai_marginal) else np.nan,
            "peer_marginal": float(peer_marginal) if np.isfinite(peer_marginal) else np.nan,
        }

    # ======================================================================
    # updates
    # ======================================================================
    def _update_anchor_values(self, vec3, task_weights, target, lr, low=-0.12, high=0.20):
        w = np.asarray(task_weights, dtype=float)
        for a in range(3):
            eff_lr = float(lr * (0.25 + 0.75 * w[a]))
            vec3[a] = np.clip(vec3[a] + eff_lr * (target - vec3[a]), low, high)

    def _update_experience_and_knowledge(self, i, req, task_weights, mode, outcome, success, partner=None):
        c = self.cfg
        q = float(outcome["quality"])
        support_gain = q - float(outcome["solo_quality"])

        # Mode belief tracks the realised task quality obtained with that mode on
        # similar tasks. This is exactly what the agent needs before the next choice:
        # "How well do I usually solve tasks like this when I use this support route?"
        # The agent does not observe counterfactual outcomes for unchosen modes.
        self._update_anchor_values(
            self.mode_value[i, :, mode], task_weights, q,
            c.mode_value_learning_rate, low=0.05, high=0.95
        )

        if mode in (MODE_AI, MODE_HYBRID) and np.isfinite(outcome["ai_source_quality"]):
            signal = float(outcome["ai_source_quality"])
            self.trust_ai[i] = np.clip(
                self.trust_ai[i] + c.trust_learning_rate * (signal - self.trust_ai[i]), 0.01, 0.99
            )

        if mode in (MODE_PEER, MODE_HYBRID) and partner is not None:
            source = float(outcome["peer_source_quality"])
            self.peer_trust[i, partner] = np.clip(
                self.peer_trust[i, partner] + c.trust_learning_rate * (source - self.peer_trust[i, partner]),
                0.01, 0.99,
            )
            # Partner-specific belief represents task-relevant expertise, not the
            # focal agent's final task outcome.
            self._update_anchor_values(
                self.peer_expected[i, partner], task_weights,
                float(outcome["peer_source_quality"]), c.mode_value_learning_rate,
                low=0.01, high=0.99
            )

        # Domain-specific learning: everyone learns a little from practice, but
        # support adds more when the source contains task-relevant knowledge above
        # the agent's current capability. Thus repeated interaction can alter ranks.
        own_before = np.asarray(getattr(self, "_skills_for_episode", self.skills)[i], dtype=float)
        practice = c.practice_learning_rate * q * np.asarray(req) * (1.0 - self.skills[i])

        novelty_vec = np.maximum(0.0, np.asarray(outcome["effective_skill"]) - own_before)
        if mode == MODE_AI:
            absorb = 0.35 + 0.40 * self.ai_literacy[i] + 0.25 * self.verification[i]
        elif mode == MODE_PEER:
            absorb = 0.35 + 0.42 * self.sociability[i] + 0.23 * (self.tie[i, partner] if partner is not None else 0.0)
        else:
            absorb = 0.25 + 0.28 * self.ai_literacy[i] + 0.25 * self.verification[i] + 0.22 * self.sociability[i]
        support_learning = c.knowledge_learning_rate * float(np.clip(absorb, 0.15, 0.95)) * q * novelty_vec * np.asarray(req)
        self.skills[i] = np.clip(self.skills[i] + practice + support_learning, 0.0, 0.995)

        self.confidence[i] = np.clip(
            self.confidence[i] + c.confidence_learning_rate * (q - self.confidence[i]), 0.02, 0.98
        )
        self.mode_counts[i, mode] += 1
        self.cumulative_quality[i] += q
        self.cumulative_success[i] += float(success)

    def _update_tie_from_peer_contribution(self, i, j, peer_marginal):
        c = self.cfg
        if j is None or not np.isfinite(peer_marginal):
            return
        existing = float(self.tie[i, j])
        if existing <= 0:
            if peer_marginal <= c.new_tie_formation_threshold:
                return
            existing = c.new_tie_initial_strength
            self.tie[i, j] = self.tie[j, i] = existing
            self.peer_trust[i, j] = self.peer_trust[j, i] = 0.50
        if peer_marginal >= 0:
            delta = c.tie_reinforcement_rate * peer_marginal * (1.0 - existing)
        else:
            delta = c.tie_negative_update_rate * peer_marginal * existing
        new_w = float(np.clip(existing + delta, 0.0, 1.0))
        self.tie[i, j] = self.tie[j, i] = new_w

    def _decay_unused_ties(self, used_mask):
        c = self.cfg
        active = self.tie > 0
        unused = active & (~used_mask)
        self.tie[unused] *= (1.0 - c.tie_decay)
        remove = unused & (self.tie < c.tie_removal_threshold)
        self.tie[remove] = 0.0
        np.fill_diagonal(self.tie, 0.0)

    def _update_local_observational_learning(self, modes, support_gains, task_weights):
        c = self.cfg
        for i in range(self.n):
            nbrs = np.flatnonzero(self.tie[i] > 0)
            if nbrs.size == 0:
                continue
            observed = nbrs[self.rng.random(nbrs.size) < c.social_observation_rate]
            if observed.size == 0:
                continue
            base_w = self.tie[i, observed].astype(float)
            for m in range(3):
                chosen = observed[modes[observed] == m]
                if chosen.size == 0:
                    continue
                chosen_mask = modes[observed] == m
                bw = base_w[chosen_mask]
                # An observed task updates nearby anchors in proportion to similarity.
                for a in range(3):
                    simw = bw * task_weights[chosen, a]
                    if simw.sum() <= 1e-12:
                        continue
                    signal = float(np.average(support_gains[chosen], weights=simw))
                    old = float(self.local_mode_signal[i, a, m])
                    self.local_mode_signal[i, a, m] = np.clip(
                        old + c.local_memory_learning_rate * (signal - old), -0.18, 0.24
                    )

    # ======================================================================
    # episode
    # ======================================================================
    def step(self, episode):
        c = self.cfg
        if c.reliability_schedule and episode in c.reliability_schedule:
            self.current_ai_reliability = float(np.clip(c.reliability_schedule[episode], 0.02, 0.99))

        self._tie_for_episode = self.tie.copy()
        self._skills_for_episode = self.skills.copy()

        family, req, ambiguity, difficulty, success_threshold, task_weights = self._generate_tasks()
        modes = np.zeros(self.n, dtype=int)
        qualities = np.zeros(self.n, dtype=float)
        successes = np.zeros(self.n, dtype=float)
        support_gains = np.zeros(self.n, dtype=float)
        used_mask = np.zeros((self.n, self.n), dtype=bool)
        ai_source_values = np.full(self.n, np.nan)
        peer_source_values = np.full(self.n, np.nan)
        peer_marginals = np.full(self.n, np.nan)

        if c.ai_enabled:
            avail_prob = np.array([self._ai_availability_probability(x) for x in self.ai_access])
            ai_available = self.rng.random(self.n) < avail_prob
        else:
            avail_prob = np.zeros(self.n)
            ai_available = np.zeros(self.n, dtype=bool)

        for i in self.rng.permutation(self.n):
            candidates = self._available_peer_candidates(i)
            peer_immediately_available = bool(candidates.size > 0)
            n_peer_candidates = int(candidates.size)
            if candidates.size == 0 and not bool(ai_available[i]):
                # Rare feasibility fallback: if neither source is immediately available,
                # the agent waits for one existing contact rather than consulting a
                # completely random stranger. This affects only the no-source corner.
                tie_view = getattr(self, "_tie_for_episode", self.tie)
                nbrs = np.flatnonzero(tie_view[i] > 0)
                if nbrs.size:
                    candidates = np.array([int(nbrs[np.argmax(tie_view[i, nbrs])])], dtype=int)
                else:
                    pool = np.delete(np.arange(self.n), i)
                    candidates = np.array([int(self.rng.choice(pool))], dtype=int)
            mode, probs = self._mode_choice(i, candidates, task_weights[i], bool(ai_available[i]), ambiguity=float(ambiguity[i]))
            expected_quality_before = float(self.mode_value[i, :, mode] @ np.asarray(task_weights[i], dtype=float))
            partner = None
            if mode in (MODE_PEER, MODE_HYBRID):
                partner, _ = self._choose_peer(i, candidates, task_weights[i])
                used_mask[i, partner] = used_mask[partner, i] = True
                self.last_used_episode[i, partner] = self.last_used_episode[partner, i] = int(episode)
                self.peer_interaction_count[i, partner] += 1
                self.peer_interaction_count[partner, i] += 1

            outcome = self._realise_outcome(
                i, mode, req[i], float(ambiguity[i]), float(difficulty[i]), partner=partner
            )
            q = float(outcome["quality"])
            success = bool(q >= success_threshold[i])
            modes[i], qualities[i], successes[i] = mode, q, float(success)
            support_gains[i] = q - float(outcome["solo_quality"])
            ai_source_values[i] = outcome["ai_source_quality"]
            peer_source_values[i] = outcome["peer_source_quality"]
            peer_marginals[i] = outcome["peer_marginal"]

            self._update_experience_and_knowledge(
                i, req[i], task_weights[i], mode, outcome, success, partner=partner
            )
            if partner is not None:
                self._update_tie_from_peer_contribution(i, partner, outcome["peer_marginal"])

            if c.record_task_details:
                self.task_mode_records.append({
                    "episode": int(episode),
                    "agent": int(i),
                    "task_type": TASK_NAMES[int(family[i])],
                    "mode": MODE_NAMES[int(mode)],
                    "quality": q,
                    "expected_quality_before": expected_quality_before,
                    "prediction_error": float(q - expected_quality_before),
                    "success": int(success),
                    "difficulty": float(difficulty[i]),
                    "ambiguity": float(ambiguity[i]),
                    "req_dim1": float(req[i, 0]),
                    "req_dim2": float(req[i, 1]),
                    "req_dim3": float(req[i, 2]),
                    "solo_quality": float(outcome["solo_quality"]),
                    "ai_source_quality": float(outcome["ai_source_quality"]) if np.isfinite(outcome["ai_source_quality"]) else np.nan,
                    "peer_source_quality": float(outcome["peer_source_quality"]) if np.isfinite(outcome["peer_source_quality"]) else np.nan,
                    "ai_marginal": float(outcome["ai_marginal"]) if np.isfinite(outcome["ai_marginal"]) else np.nan,
                    "peer_marginal": float(outcome["peer_marginal"]) if np.isfinite(outcome["peer_marginal"]) else np.nan,
                    "p_peer": float(probs[MODE_PEER]),
                    "p_ai": float(probs[MODE_AI]),
                    "p_hybrid": float(probs[MODE_HYBRID]),
                    "ai_available": int(ai_available[i]),
                    "peer_immediately_available": int(peer_immediately_available),
                    "n_peer_candidates": int(n_peer_candidates),
                    "ai_availability_probability": float(avail_prob[i]),
                })

        self._decay_unused_ties(used_mask)
        self._update_local_observational_learning(modes, support_gains, task_weights)
        self.last_mode = modes.copy()
        self.last_quality = qualities.copy()
        self.last_task_weights = task_weights.copy()

        peer_share = float(np.mean(modes == MODE_PEER))
        ai_share = float(np.mean(modes == MODE_AI))
        hybrid_share = float(np.mean(modes == MODE_HYBRID))
        human_interaction_rate = peer_share + hybrid_share
        ai_use_rate = ai_share + hybrid_share

        knowledge = self.skills.mean(axis=1)
        initial_knowledge = self.initial_skills.mean(axis=1)
        if np.std(initial_knowledge) > 1e-12 and np.std(knowledge) > 1e-12:
            knowledge_rank_corr = float(np.corrcoef(initial_knowledge, knowledge)[0, 1])
        else:
            knowledge_rank_corr = np.nan

        weighted_degree = self._weighted_degree()
        active_edges = self.tie > 0
        recent_cutoff = max(1, int(episode) - 19)
        recent_edges = active_edges & (self.last_used_episode >= recent_cutoff)
        active_edge_count = int(np.count_nonzero(np.triu(active_edges, 1)))
        recent_edge_count = int(np.count_nonzero(np.triu(recent_edges, 1)))
        recent_active_tie_fraction = recent_edge_count / max(1, active_edge_count)
        unique_partner_counts = np.count_nonzero(self.peer_interaction_count > 0, axis=1)

        rec = {
            "episode": int(episode),
            "peer_share": peer_share,
            "ai_share": ai_share,
            "hybrid_share": hybrid_share,
            "human_interaction_rate": human_interaction_rate,
            "ai_use_rate": ai_use_rate,
            "mean_quality": float(np.mean(qualities)),
            "success_rate": float(np.mean(successes)),
            "mean_knowledge": float(np.mean(knowledge)),
            "knowledge_gini": self._gini(knowledge),
            "knowledge_rank_corr_with_initial": knowledge_rank_corr,
            "performance_gini": self._gini(qualities),
            "mean_ai_trust": float(np.mean(self.trust_ai)) if c.ai_enabled else 0.0,
            "mean_peer_trust": self._mean_peer_trust(),
            "mean_tie_strength": self._mean_tie_strength(),
            "network_density": self._network_density(),
            "mean_weighted_degree": float(np.mean(weighted_degree)),
            "recent_active_tie_fraction": float(recent_active_tie_fraction),
            "mean_unique_peer_partners": float(np.mean(unique_partner_counts)),
            "ai_reliability": float(self.current_ai_reliability),
            "mean_ai_availability_probability": float(np.mean(avail_prob)),
            "mean_ai_source_quality": float(np.nanmean(ai_source_values)) if np.any(np.isfinite(ai_source_values)) else np.nan,
            "mean_peer_source_quality": float(np.nanmean(peer_source_values)) if np.any(np.isfinite(peer_source_values)) else np.nan,
            "mean_peer_marginal": float(np.nanmean(peer_marginals)) if np.any(np.isfinite(peer_marginals)) else np.nan,
        }
        self.history.append(rec)
        del self._tie_for_episode
        del self._skills_for_episode
        return rec

    def run(self):
        for ep in range(1, self.episodes + 1):
            self.step(ep)
        return self.history

    # ======================================================================
    # summaries
    # ======================================================================
    def final_summary(self, tail=20):
        if not self.history:
            raise RuntimeError("run() must be called before final_summary()")
        tail = max(1, min(int(tail), len(self.history)))
        recent = self.history[-tail:]
        keys = [
            "peer_share", "ai_share", "hybrid_share", "human_interaction_rate",
            "ai_use_rate", "mean_quality", "success_rate", "mean_knowledge",
            "knowledge_gini", "knowledge_rank_corr_with_initial", "performance_gini",
            "mean_ai_trust", "mean_peer_trust", "mean_tie_strength", "network_density",
            "mean_weighted_degree", "recent_active_tie_fraction",
            "mean_unique_peer_partners", "mean_peer_marginal",
            "mean_ai_availability_probability",
        ]
        out = {f"final_{k}": float(np.nanmean([r[k] for r in recent])) for k in keys}
        final_mask = self.tie > 0
        out["initial_edge_count"] = self.initial_edge_count
        out["final_edge_count"] = int(np.count_nonzero(np.triu(final_mask, 1)))
        retained = int(np.count_nonzero(np.triu(final_mask & self.initial_edge_mask, 1)))
        new_edges = int(np.count_nonzero(np.triu(final_mask & (~self.initial_edge_mask), 1)))
        out["edge_retention_ratio"] = retained / max(1, self.initial_edge_count)
        out["edge_count_ratio"] = out["final_edge_count"] / max(1, self.initial_edge_count)
        out["new_edge_ratio"] = new_edges / max(1, self.initial_edge_count)
        active_weights = self.tie[np.triu(final_mask, 1)]
        out["strong_tie_fraction"] = float(np.mean(active_weights >= 0.40)) if active_weights.size else 0.0
        out["initial_mean_tie"] = self.initial_mean_tie
        out["final_mean_skill"] = float(self.skills.mean())
        out["final_skill_gini"] = self._gini(self.skills.mean(axis=1))
        return out

    def agent_summary(self):
        rows = []
        active = self.tie > 0
        weighted_degree = self._weighted_degree()
        for i in range(self.n):
            nbrs = np.flatnonzero(active[i])
            mean_peer_trust = float(self.peer_trust[i, nbrs].mean()) if nbrs.size else np.nan
            counts = self.mode_counts[i].astype(float)
            denom = max(1.0, counts.sum())
            rows.append({
                "agent": i,
                "initial_knowledge": float(self.initial_skills[i].mean()),
                "initial_skill_dim1": float(self.initial_skills[i, 0]),
                "initial_skill_dim2": float(self.initial_skills[i, 1]),
                "initial_skill_dim3": float(self.initial_skills[i, 2]),
                "ai_literacy": float(self.initial_ai_literacy[i]),
                "sociability": float(self.initial_sociability[i]),
                "verification": float(self.initial_verification[i]),
                "initial_confidence": float(self.initial_confidence[i]),
                "ai_access": float(self.initial_ai_access[i]),
                "final_knowledge": float(self.skills[i].mean()),
                "final_skill_dim1": float(self.skills[i, 0]),
                "final_skill_dim2": float(self.skills[i, 1]),
                "final_skill_dim3": float(self.skills[i, 2]),
                "knowledge_gain": float(self.skills[i].mean() - self.initial_skills[i].mean()),
                "final_confidence": float(self.confidence[i]),
                "final_ai_trust": float(self.trust_ai[i]),
                "final_mean_peer_trust": mean_peer_trust,
                "final_weighted_degree": float(weighted_degree[i]),
                "unique_peer_partners": int(np.count_nonzero(self.peer_interaction_count[i] > 0)),
                "peer_share": float(counts[MODE_PEER] / denom),
                "ai_share": float(counts[MODE_AI] / denom),
                "hybrid_share": float(counts[MODE_HYBRID] / denom),
                "mean_quality": float(self.cumulative_quality[i] / denom),
                "success_rate": float(self.cumulative_success[i] / denom),
            })
        return rows

    def validate_state(self):
        arrays = [
            self.skills, self.ai_literacy, self.sociability, self.verification,
            self.confidence, self.mode_value, self.trust_ai, self.tie,
            self.peer_trust, self.peer_expected, self.local_mode_signal,
        ]
        for arr in arrays:
            if not np.all(np.isfinite(arr)):
                raise AssertionError("Non-finite model state detected")
        if np.any(self.tie < -1e-8) or np.any(self.tie > 1 + 1e-8):
            raise AssertionError("Tie weights outside [0,1]")
        if not np.allclose(self.tie, self.tie.T, atol=1e-6):
            raise AssertionError("Tie matrix lost symmetry")
        if not self.cfg.ai_enabled and np.any(self.mode_counts[:, MODE_AI:] > 0):
            raise AssertionError("AI/Hybrid choices occurred while AI disabled")
        return True
