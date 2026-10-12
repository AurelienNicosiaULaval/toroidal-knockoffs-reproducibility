"""Version 2 endpoint allocations, with a principal (basis-invariant) square root.

The Perron allocation minimizes maximum node loading in the edge-admissible
family. This is not a power or reconstructability optimum. See supplement.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from toroidal import Model, vec


def psd_root(a: np.ndarray) -> np.ndarray:
    """Principal square root; reject materially negative eigenvalues."""
    values, vectors = np.linalg.eigh(a)
    tolerance = 1e-12*max(1., np.linalg.norm(a, 2))
    if values.min() < -tolerance:
        raise ValueError('Matrix is not positive semidefinite')
    # Exact analytic zeros are truncated at round-off scale on the boundary.
    values[values < tolerance] = 0.
    return (vectors*np.sqrt(values))@vectors.T


def norm_adjacency(model: Model) -> np.ndarray:
    s = np.zeros((model.p, model.p))
    for (j, k), mat in zip(model.edges, model.M):
        s[j, k] = s[k, j] = np.linalg.norm(mat, 2)
    return s


def perron_vector(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Positive Perron vector on each nontrivial connected component.

    Dense eigendecomposition costs O(sum component_size^3), once per model.
    Returned radii are one per node. Isolated vertices receive radius zero.
    """
    s = norm_adjacency(model)
    count, labels = connected_components(csr_matrix(s), directed=False)
    u = np.ones(model.p)
    radii = np.zeros(model.p)
    for label in range(count):
        ind = np.flatnonzero(labels == label)
        if len(ind) == 1:
            continue
        values, vectors = np.linalg.eigh(s[np.ix_(ind, ind)])
        component = np.abs(vectors[:, -1])
        if component.min() <= 1e-14:
            raise ArithmeticError('Perron vector numerically unresolved; rescale or split the model')
        u[ind] = component/np.exp(np.log(component).mean())
        radii[ind] = values[-1]
    return u, radii


def endpoint_allocations(model: Model, u: np.ndarray | None = None) -> np.ndarray:
    """a_jk = s_jk*u_k/u_j, a_kj = s_jk*u_j/u_k; no response access."""
    u = np.ones(model.p) if u is None else np.asarray(u, float)
    if u.shape != (model.p,) or not np.all(np.isfinite(u)) or np.any(u <= 0):
        raise ValueError('Positive finite node weights required')
    out = np.empty((len(model.edges), 2))
    for e, ((j, k), mat) in enumerate(zip(model.edges, model.M)):
        s = np.linalg.norm(mat, 2)
        out[e] = s*np.array([u[k]/u[j], u[j]/u[k]])
    return out


@dataclass
class AllocatedSampler:
    """Exactly admissible edge law; changed numerical factor is versioned v2."""
    model: Model
    allocations: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.allocations is None:
            self.allocations = endpoint_allocations(self.model)
        self.allocations = np.asarray(self.allocations, float)
        if self.allocations.shape != (len(self.model.edges), 2):
            raise ValueError('One endpoint pair is required for each edge')
        if not np.all(np.isfinite(self.allocations)) or np.any(self.allocations < 0):
            raise ValueError('Loadings must be finite and nonnegative')
        self.blocks = []
        self.loads = np.zeros(self.model.p)
        for (j, k), mat, (a, b) in zip(self.model.edges, self.model.M, self.allocations):
            s = np.linalg.norm(mat, 2)
            if a*b < s*s-1e-12*max(1., s*s):
                raise ValueError('Endpoint product is smaller than squared operator norm')
            block = np.block([[a*np.eye(2), mat], [mat.T, b*np.eye(2)]])
            root = psd_root(block)
            self.blocks.append((j, k, root[:2], root[2:]))
            self.loads[[j, k]] += [a, b]

    def fields(self, rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.model.p or not np.all(np.isfinite(x)):
            raise ValueError('Finite n by p angle array required')
        v = vec(x)
        f = np.broadcast_to(self.model.h, v.shape).copy()
        for j, k, bj, bk in self.blocks:
            z = v[:, j]@bj+v[:, k]@bk+rng.normal(size=(len(x), 4))
            f[:, j] += z@bj.T
            f[:, k] += z@bk.T
        return f

    def sample(self, rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
        f = self.fields(rng, x)
        return rng.vonmises(np.arctan2(f[:, :, 1], f[:, :, 0]), np.linalg.norm(f, axis=2))

    def aggregate(self) -> np.ndarray:
        return self.model.matrix()+np.diag(np.repeat(self.loads, 2))


def joint_mvr_score(x: np.ndarray, xt: np.ndarray, ridge: float = 1e-3) -> float:
    """Mean diagonal precision of the FULL embedded pair, lower is better.

    A regularized linear reconstruction proxy, not the nonlinear MVR optimum.
    Ridge and candidate grid must be fixed before evaluation.
    """
    f = np.concatenate([vec(x), vec(xt)], axis=1).reshape(len(x), -1)
    f -= f.mean(0)
    covariance = f.T@f/len(f)+ridge*np.eye(f.shape[1])
    return float(np.trace(np.linalg.solve(covariance, np.eye(len(covariance))))/len(covariance))
