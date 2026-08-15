"""Online policies for the B3 validity-budget study."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Hashable

import numpy as np

from experiments.rl.budget_env import ACTIONS, ACTION_INDEX, BudgetAction


class FixedBudget:
    def __init__(self, action: BudgetAction):
        self.action = action
        self.name = f"fixed:{action.name}"

    def select(self, _feats: list[float] | None = None, _state: Hashable | None = None) -> BudgetAction:
        return self.action

    def update(self, *_a, **_k) -> None:
        return


class LinUCB:
    """Disjoint linear UCB over discrete validity-budget actions (numpy)."""

    def __init__(self, n_features: int, *, alpha: float = 1.0, l2: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha
        self.name = f"linucb:a{alpha}"
        eye = np.eye(n_features, dtype=np.float64)
        self.A = {i: l2 * eye.copy() for i in range(len(ACTIONS))}
        self.Ainv = {i: (1.0 / l2) * eye.copy() for i in range(len(ACTIONS))}
        self.b = {i: np.zeros(n_features, dtype=np.float64) for i in range(len(ACTIONS))}
        self.theta = {i: np.zeros(n_features, dtype=np.float64) for i in range(len(ACTIONS))}

    def select(self, feats: list[float], _state: Hashable | None = None) -> BudgetAction:
        x = np.asarray(feats, dtype=np.float64)
        best_i = 0
        best_ucb = -1e18
        for i in range(len(ACTIONS)):
            mean = float(self.theta[i] @ x)
            bonus = self.alpha * float(np.sqrt(max(0.0, x @ self.Ainv[i] @ x)))
            ucb = mean + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_i = i
        return ACTIONS[best_i]

    def update(self, feats: list[float], action: BudgetAction, reward: float, _state=None) -> None:
        i = ACTION_INDEX[action.name]
        x = np.asarray(feats, dtype=np.float64)
        # Sherman-Morrison rank-one update for Ainv.
        Ainv = self.Ainv[i]
        Ax = Ainv @ x
        denom = 1.0 + float(x @ Ax)
        self.Ainv[i] = Ainv - np.outer(Ax, Ax) / denom
        self.A[i] = self.A[i] + np.outer(x, x)
        self.b[i] = self.b[i] + reward * x
        self.theta[i] = self.Ainv[i] @ self.b[i]


class QLearning:
    """Tabular ε-greedy Q-learning over discrete validity-budget actions."""

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        alpha: float = 0.2,
        gamma: float = 0.0,
        seed: int = 0,
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.name = f"qlearn:e{epsilon}"
        self.Q: dict[Hashable, list[float]] = defaultdict(lambda: [0.0] * len(ACTIONS))

    def select(self, _feats: list[float] | None, state: Hashable) -> BudgetAction:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(ACTIONS))
        values = self.Q[state]
        best = max(range(len(ACTIONS)), key=lambda i: values[i])
        return ACTIONS[best]

    def update(
        self,
        _feats: list[float] | None,
        action: BudgetAction,
        reward: float,
        state: Hashable,
        next_state: Hashable | None = None,
    ) -> None:
        i = ACTION_INDEX[action.name]
        q = self.Q[state]
        nxt = 0.0
        if next_state is not None and self.gamma > 0:
            nxt = max(self.Q[next_state])
        td = reward + self.gamma * nxt - q[i]
        q[i] += self.alpha * td
