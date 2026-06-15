"""AI-mediated interaction agent-based model.

This module implements a compact exploratory ABM for studying how generative AI
may alter collaboration, trust, and social learning. The model is intentionally
mechanism-oriented rather than empirically calibrated: it is designed to make a
set of behavioural assumptions explicit and then examine their aggregate
consequences under contrasting AI-use scenarios.

Epistemological position
------------------------
The model should be read as a generative, exploratory social simulation, not as a
predictive representation of a specific real-world population. Variables such as
``knowledge``, ``trust_ai`` and ``sociability`` are operational constructs on a
normalised 0--1 scale. In particular, ``knowledge`` is not a philosophical claim
that human knowledge is intrinsically scalar; it is a task-relevant performance
capacity used to compare how interaction regimes accumulate capability over
repeated task episodes.

Decision rule
-------------
The interaction choice rule uses softmax over mode-specific behavioural
propensity scores. These scores are referred to as utilities in the paper, but
should be interpreted as bounded-rational propensities rather than welfare
utilities. The numerical weights below are transparent modelling assumptions that
encode directional hypotheses from the human-AI collaboration literature. They
are exposed here so that users can inspect, modify and rerun the model.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

# -----------------------------------------------------------------------------
# Behavioural propensity weights used in the interaction-choice rule.
# These constants correspond exactly to the equations reported in the paper.
# -----------------------------------------------------------------------------
UTILITY_WEIGHTS = {
    "peer": {
        "sociability": 1.15,
        "trust_peer": 1.00,
        "neighbor_success": 0.70,
        "local_tie_strength": 0.20,
        "difficulty": -0.65,
    },
    "ai": {
        "ai_literacy": 1.15,
        "trust_ai": 0.95,
        "confidence": 0.55,
        "ai_convenience": 0.75,
        "difficulty": -0.55,
    },
    "hybrid": {
        "ai_literacy": 0.95,
        "sociability": 0.85,
        "combined_trust": 0.60,
        "verification": 0.45,
        "ai_convenience": 0.35,
        "difficulty": -0.42,
    },
}

# -----------------------------------------------------------------------------
# Mode-specific task-quality weights. The task is abstract: it represents a
# collaborative problem-solving episode in which an agent may consult a peer,
# AI, or both. Quality is clipped to [0, 1] after adding small stochastic noise.
# -----------------------------------------------------------------------------
QUALITY_WEIGHTS = {
    "peer": {
        "own_knowledge": 0.44,
        "peer_knowledge": 0.30,
        "trust_peer": 0.12,
        "verification": 0.10,
        "difficulty": -0.40,
    },
    "ai": {
        "ai_literacy": 0.34,
        "trust_ai": 0.26,
        "ai_reliability": 0.30,
        "confidence": 0.08,
        "difficulty": -0.34,
        "verification_error_penalty": -0.08,
    },
    "hybrid": {
        "ai_literacy": 0.22,
        "ai_reliability": 0.22,
        "peer_knowledge": 0.18,
        "verification": 0.14,
        "trust_peer": 0.10,
        "trust_ai": 0.08,
        "difficulty": -0.28,
    },
}


class InteractionABM:
    """Agent-based model of peer-first, AI-first, and hybrid interaction.

    Parameters are deliberately exposed in the constructor so that the model can
    be used for scenario analysis and sensitivity analysis. Unless otherwise
    stated, variables are normalised to [0, 1].

    A simulation step is one task episode, not a calendar unit. In each episode,
    every agent faces the same task difficulty draw and chooses an interaction
    mode probabilistically. The resulting task quality updates knowledge, trust,
    recent success memory, cumulative performance, and social-tie weights.
    """

    def __init__(
        self,
        n_agents=120,
        steps=60,
        network_type="small_world",
        network_density=0.06,
        ai_reliability=0.82,
        ai_convenience=0.72,
        verification_strength=0.80,
        peer_learning_rate=0.042,
        ai_learning_rate=0.018,
        hybrid_learning_rate=0.040,
        trust_gain=0.035,
        trust_loss=0.050,
        tie_gain=0.070,
        tie_loss=0.022,
        decay_interval=5,
        choice_temperature=1.0,
        seed=42,
    ):
        self.n_agents = int(n_agents)
        self.steps = int(steps)
        self.network_type = network_type
        self.network_density = float(network_density)
        self.ai_reliability = float(ai_reliability)
        self.ai_convenience = float(ai_convenience)
        self.verification_strength = float(verification_strength)
        self.peer_learning_rate = float(peer_learning_rate)
        self.ai_learning_rate = float(ai_learning_rate)
        self.hybrid_learning_rate = float(hybrid_learning_rate)
        self.trust_gain = float(trust_gain)
        self.trust_loss = float(trust_loss)
        self.tie_gain = float(tie_gain)
        self.tie_loss = float(tie_loss)
        self.decay_interval = int(decay_interval)
        self.choice_temperature = float(choice_temperature)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        if self.choice_temperature <= 0:
            raise ValueError("choice_temperature must be positive.")

        self._init_agents()
        self._init_network()
        self.history = []
        self.current_step = 0
        self.baseline_human_capacity = None

    def _bounded_normal(self, mean, sd, low=0.0, high=1.0, size=None):
        """Draw a normal variable and clip it to the model's admissible range."""
        return np.clip(self.rng.normal(mean, sd, size=size), low, high)

    def _init_agents(self):
        """Initialise heterogeneous agent attributes.

        These initial values are not calibrated to a specific population. They
        create moderate heterogeneity around central tendencies so that the model
        can examine how interaction mechanisms, rather than initial extremes,
        produce different aggregate outcomes.
        """
        n = self.n_agents
        self.knowledge = self._bounded_normal(0.52, 0.12, 0.10, 0.95, n)
        self.confidence = self._bounded_normal(0.50, 0.15, 0.05, 0.95, n)
        self.trust_ai = self._bounded_normal(0.52, 0.16, 0.05, 0.95, n)
        self.trust_peer = self._bounded_normal(0.58, 0.14, 0.05, 0.95, n)
        self.ai_literacy = self._bounded_normal(0.48, 0.18, 0.05, 0.95, n)
        self.sociability = self._bounded_normal(0.55, 0.15, 0.05, 0.95, n)
        self.verification = self._bounded_normal(0.55, 0.17, 0.05, 0.95, n)
        self.recent_success = np.full(n, 0.50, dtype=float)
        self.performance = np.zeros(n, dtype=float)

    def _init_network(self):
        """Initialise the social network.

        The default network is a connected Watts-Strogatz small-world graph. This
        is a stylised representation of clustered local relations with a limited
        number of long-range bridges. With n=120 and network_density=0.06, the
        average degree is approximately 8, so the population size is within the
        broad Dunbar-number range while the active local neighbourhood remains
        much smaller than the total population.
        """
        n = self.n_agents
        if self.network_type == "erdos_renyi":
            p = self.network_density
            g = nx.erdos_renyi_graph(n, p, seed=self.seed)
            if not nx.is_connected(g):
                k_fallback = max(4, int(round(self.network_density * n)))
                if k_fallback % 2 == 1:
                    k_fallback += 1
                g = nx.connected_watts_strogatz_graph(
                    n, k_fallback, 0.12, tries=200, seed=self.seed
                )
        else:
            k = max(4, int(round(self.network_density * n)))
            if k % 2 == 1:
                k += 1
            k = min(k, n - 1 - ((n - 1) % 2))
            g = nx.connected_watts_strogatz_graph(n, k, 0.12, tries=200, seed=self.seed)

        self.graph = g
        self.neighbors = [np.asarray(list(g.neighbors(i)), dtype=int) for i in range(n)]
        self.edge_index = {}
        edge_pairs = []
        for idx, (u, v) in enumerate(g.edges()):
            edge_pairs.append((u, v))
            self.edge_index[(u, v)] = idx
            self.edge_index[(v, u)] = idx
        self.edge_pairs = np.asarray(edge_pairs, dtype=int)
        self.edge_weights = self.rng.uniform(0.8, 1.2, size=len(edge_pairs))
        self.edge_last_used = np.zeros(len(edge_pairs), dtype=int)

    def _row_softmax(self, propensities):
        """Convert behavioural propensities into choice probabilities."""
        scaled = propensities / self.choice_temperature
        m = scaled.max(axis=1, keepdims=True)
        z = np.exp(scaled - m)
        return z / z.sum(axis=1, keepdims=True)

    def _avg_neighbor_success_and_strength(self):
        """Return local social cues used in the peer-first propensity."""
        n = self.n_agents
        avg_success = np.full(n, 0.50, dtype=float)
        avg_strength = np.full(n, 0.70, dtype=float)
        for i, nbrs in enumerate(self.neighbors):
            if nbrs.size:
                avg_success[i] = self.recent_success[nbrs].mean()
                eidxs = [self.edge_index[(i, int(j))] for j in nbrs]
                avg_strength[i] = self.edge_weights[eidxs].mean()
        return avg_success, avg_strength

    def _choose_modes(self, difficulty):
        """Choose peer-first (0), AI-first (1), or hybrid (2) interaction."""
        avg_peer_success, local_tie_strength = self._avg_neighbor_success_and_strength()
        w_peer = UTILITY_WEIGHTS["peer"]
        w_ai = UTILITY_WEIGHTS["ai"]
        w_hybrid = UTILITY_WEIGHTS["hybrid"]

        prop_peer = (
            w_peer["sociability"] * self.sociability
            + w_peer["trust_peer"] * self.trust_peer
            + w_peer["neighbor_success"] * avg_peer_success
            + w_peer["local_tie_strength"] * local_tie_strength
            + w_peer["difficulty"] * difficulty
        )
        prop_ai = (
            w_ai["ai_literacy"] * self.ai_literacy
            + w_ai["trust_ai"] * self.trust_ai
            + w_ai["confidence"] * self.confidence
            + w_ai["ai_convenience"] * self.ai_convenience
            + w_ai["difficulty"] * difficulty * (0.4 + self.verification * self.verification_strength)
        )
        prop_hybrid = (
            w_hybrid["ai_literacy"] * self.ai_literacy
            + w_hybrid["sociability"] * self.sociability
            + w_hybrid["combined_trust"] * (self.trust_ai + self.trust_peer)
            + w_hybrid["verification"] * self.verification
            + w_hybrid["ai_convenience"] * self.ai_convenience
            + w_hybrid["difficulty"] * difficulty
        )
        probs = self._row_softmax(np.column_stack([prop_peer, prop_ai, prop_hybrid]))
        draws = self.rng.random(self.n_agents)
        return (draws[:, None] > probs.cumsum(axis=1)).sum(axis=1)

    def _choose_peers(self, modes):
        """For peer-first and hybrid modes, randomly select one local neighbour."""
        peer_choices = np.full(self.n_agents, -1, dtype=int)
        active = np.where((modes == 0) | (modes == 2))[0]
        for i in active:
            nbrs = self.neighbors[i]
            if nbrs.size:
                peer_choices[i] = int(nbrs[self.rng.integers(0, nbrs.size)])
        return peer_choices

    def _task_quality(self, modes, peer_choices, difficulty):
        """Compute clipped task quality for the selected interaction mode."""
        noise = self.rng.normal(0.0, 0.045, size=self.n_agents)
        q = np.zeros(self.n_agents, dtype=float)

        peer_mask = modes == 0
        ai_mask = modes == 1
        hybrid_mask = modes == 2
        w_peer = QUALITY_WEIGHTS["peer"]
        w_ai = QUALITY_WEIGHTS["ai"]
        w_hybrid = QUALITY_WEIGHTS["hybrid"]

        if peer_mask.any():
            peers = peer_choices[peer_mask]
            peer_k = np.where(peers >= 0, self.knowledge[peers], 0.0)
            peer_t = np.where(peers >= 0, self.trust_peer[peer_mask], 0.0)
            q[peer_mask] = (
                w_peer["own_knowledge"] * self.knowledge[peer_mask]
                + w_peer["peer_knowledge"] * peer_k
                + w_peer["trust_peer"] * peer_t
                + w_peer["verification"] * self.verification[peer_mask]
                + w_peer["difficulty"] * difficulty
                + noise[peer_mask]
            )

        if ai_mask.any():
            q[ai_mask] = (
                w_ai["ai_literacy"] * self.ai_literacy[ai_mask]
                + w_ai["trust_ai"] * self.trust_ai[ai_mask]
                + w_ai["ai_reliability"] * self.ai_reliability
                + w_ai["confidence"] * self.confidence[ai_mask]
                + w_ai["difficulty"] * difficulty
                + w_ai["verification_error_penalty"] * self.verification[ai_mask] * (1.0 - self.ai_reliability)
                + noise[ai_mask]
            )

        if hybrid_mask.any():
            peers = peer_choices[hybrid_mask]
            peer_k = np.where(peers >= 0, self.knowledge[peers], 0.0)
            q[hybrid_mask] = (
                w_hybrid["ai_literacy"] * self.ai_literacy[hybrid_mask]
                + w_hybrid["ai_reliability"] * self.ai_reliability
                + w_hybrid["peer_knowledge"] * peer_k
                + w_hybrid["verification"] * self.verification[hybrid_mask]
                + w_hybrid["trust_peer"] * self.trust_peer[hybrid_mask]
                + w_hybrid["trust_ai"] * self.trust_ai[hybrid_mask]
                + w_hybrid["difficulty"] * difficulty
                + noise[hybrid_mask]
            )

        return np.clip(q, 0.0, 1.0)

    def _update_learning(self, modes, q):
        """Update task-relevant knowledge after each task episode."""
        peer_mask = modes == 0
        ai_mask = modes == 1
        hybrid_mask = modes == 2
        self.knowledge[peer_mask] += self.peer_learning_rate * q[peer_mask]
        self.knowledge[ai_mask] += self.ai_learning_rate * self.ai_literacy[ai_mask] * q[ai_mask]
        self.knowledge[hybrid_mask] += self.hybrid_learning_rate * q[hybrid_mask]
        np.clip(self.knowledge, 0.0, 1.0, out=self.knowledge)

    def _update_trust(self, modes, q):
        """Update AI and peer trust based on whether task quality clears 0.50."""
        success = (q >= 0.50).astype(float)
        delta = np.where(success == 1.0, self.trust_gain, -self.trust_loss)
        ai_mask = (modes == 1) | (modes == 2)
        peer_mask = (modes == 0) | (modes == 2)
        self.trust_ai[ai_mask] += delta[ai_mask]
        self.trust_peer[peer_mask] += delta[peer_mask]
        np.clip(self.trust_ai, 0.0, 1.0, out=self.trust_ai)
        np.clip(self.trust_peer, 0.0, 1.0, out=self.trust_peer)
        self.recent_success = 0.70 * self.recent_success + 0.30 * success
        return success

    def _update_network(self, modes, peer_choices, success):
        """Reinforce used peer ties and decay unused ties."""
        active = np.where(((modes == 0) | (modes == 2)) & (peer_choices >= 0))[0]
        for i in active:
            j = int(peer_choices[i])
            e = self.edge_index[(int(i), j)]
            self.edge_weights[e] = max(
                0.10,
                self.edge_weights[e] + (self.tie_gain if success[i] == 1.0 else -self.tie_loss),
            )
            self.edge_last_used[e] = self.current_step
        stale = (self.current_step - self.edge_last_used) >= self.decay_interval
        self.edge_weights[stale] = np.maximum(0.10, self.edge_weights[stale] - 0.010)

    @staticmethod
    def _gini(x):
        """Calculate a non-negative Gini coefficient."""
        x = np.asarray(x, dtype=float)
        if np.amin(x) < 0:
            x = x - np.amin(x)
        x = x + 1e-9
        x = np.sort(x)
        n = x.size
        idx = np.arange(1, n + 1)
        return float(np.sum((2 * idx - n - 1) * x) / (n * np.sum(x)))

    def step(self):
        """Run one task episode for all agents and record aggregate outputs."""
        self.current_step += 1
        difficulty = float(self.rng.uniform(0.30, 0.90))
        modes = self._choose_modes(difficulty)
        peer_choices = self._choose_peers(modes)
        q = self._task_quality(modes, peer_choices, difficulty)
        self._update_learning(modes, q)
        success = self._update_trust(modes, q)
        self._update_network(modes, peer_choices, success)
        self.performance += q

        peer_events = int((modes == 0).sum())
        ai_events = int((modes == 1).sum())
        hybrid_events = int((modes == 2).sum())
        active_human = peer_events + hybrid_events
        if self.baseline_human_capacity is None:
            self.baseline_human_capacity = max(active_human, 1)

        self.history.append({
            "step": self.current_step,
            "difficulty": difficulty,
            "mean_quality": float(q.mean()),
            "mean_knowledge": float(self.knowledge.mean()),
            "mean_trust_ai": float(self.trust_ai.mean()),
            "mean_trust_peer": float(self.trust_peer.mean()),
            "mean_performance": float(self.performance.mean()),
            "peer_events": peer_events,
            "ai_events": ai_events,
            "hybrid_events": hybrid_events,
            "success_rate": float(success.mean()),
            "avg_tie_strength": float(self.edge_weights.mean()),
            "HIRI": float(active_human / self.baseline_human_capacity),
            "AMR": float((ai_events + hybrid_events) / self.n_agents),
            "gini_performance": self._gini(self.performance),
        })

    def run(self):
        """Run the full simulation and return the recorded history."""
        for _ in range(self.steps):
            self.step()
        return self.history
