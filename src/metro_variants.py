"""Periodic symmetric proposals and fixed sparse orders, distinct from v1 tuning."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from metro_circle import CircularMetro
from toroidal import Model, wrap


class PeriodicMetro(CircularMetro):
    def __init__(self, model: Model, proposal: str = 'uniform', scale: float = 1.,
                 order: np.ndarray | None = None, max_frontier: int = 16):
        if proposal not in ('uniform', 'vonmises', 'wrapped_normal', 'antipodal_mixture'):
            raise ValueError('Unknown periodic proposal')
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Positive proposal scale required')
        super().__init__(model, tau=0., order=order, max_frontier=max(model.p, max_frontier))
        self.proposal, self.scale = proposal, scale
        # Python arbitrary-width integers are essential once p exceeds 63.
        self.neighbors = [[(int(k), mat) for k, mat in ns] for ns in self.neighbors]
        # Recompute induced sparse frontiers for the specified, fixed order.
        adj = [set() for _ in range(model.p)]
        for j, k in self.model.edges:
            adj[int(j)].add(int(k)); adj[int(k)].add(int(j))
        self.front = []
        for j in range(model.p):
            neighbors = sorted(k for k in adj[j] if k > j)
            self.front.append(neighbors)
            for k, l in combinations(neighbors, 2):
                adj[k].add(l); adj[l].add(k)
        self.width = max(map(len, self.front), default=0)
        if self.width > max_frontier:
            raise ValueError(f'Frontier {self.width} exceeds cap {max_frontier}')
        self.fmask = [sum(1 << k for k in f) for f in self.front]
        self.affected = [[k for k in range(j) if j in self.front[k]] for j in range(model.p)]

    def sample(self, rng, x, proposals=None, forced=None):
        if proposals is None:
            if self.proposal == 'uniform':
                proposals = rng.uniform(-np.pi, np.pi, x.shape)
            elif self.proposal == 'vonmises':
                proposals = wrap(x+rng.vonmises(0., self.scale, x.shape))
            elif self.proposal == 'wrapped_normal':
                proposals = wrap(x+rng.normal(0., self.scale, x.shape))
            else:
                proposals = wrap(x+rng.vonmises(0., self.scale, x.shape)
                                 +np.pi*rng.integers(0, 2, x.shape))
        # Each q_j depends only on its own x_j and is even in the increment.
        # Its density cancels; the inherited rejection-history recursion remains.
        return super().sample(rng, x, proposals=proposals, forced=forced)
