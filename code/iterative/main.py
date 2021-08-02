from abc import ABC, abstractmethod

import numpy as np
import psutil
from joblib import Parallel, delayed
from scipy.interpolate import NearestNDInterpolator


rng = np.random.default_rng()
n_cores = psutil.cpu_count(logical=False)


class BrownianMotion():
    def __init__(self, b0, max_t, dt):
        num = np.int_(np.rint(max_t/dt))
        increments = rng.normal(0., np.sqrt(dt), size=(num-1, np.size(b0)))
        self.bts = np.cumsum(np.insert(increments, 0, b0, axis=0), axis=0)

    def get_exit_idx(self, indicator):
        bool_arr = indicator(self.bts)
        idx = np.argmin(bool_arr)
        if idx == 0:
            raise RuntimeError("exit index is out of reach")
        return idx


class Domain(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def indicator(self, pts):
        pass

    @abstractmethod
    def generate_grid(self, dx):
        pass

    @abstractmethod
    def u0(self, pts):
        pass


class OpenBall(Domain):
    def __init__(self, c, r):
        self.dim = np.size(c)
        self.c = c
        self.r = r

    def __str__(self):
        return f"{type(self).__name__} ({self.c}, {self.r})"

    def indicator(self, pts):
        c_distances = np.linalg.norm(pts-self.c, axis=1)
        return c_distances < self.r

    def generate_grid(self, dx):
        xs = np.linspace(-self.r, self.r, np.int_(np.rint(2*self.r/dx))+1)
        xxs = tuple(xs for _ in range(self.dim))
        grid = np.array(np.meshgrid(*xxs)).T.reshape(-1, self.dim) + self.c
        idx = np.nonzero(self.indicator(grid))
        return grid[idx]

    def u0(self, pts):
        c_distances = np.linalg.norm(pts-self.c, axis=1)
        return self.r - c_distances


class OpenAnnulus(Domain):
    def __init__(self, c, r1, r2):
        self.dim = np.size(c)
        self.c = c
        self.r1 = r1
        self.r2 = r2

    def __str__(self):
        return f"{type(self).__name__} ({self.c}, {self.r1}, {self.r2})"

    def indicator(self, pts):
        c_distances = np.linalg.norm(pts-self.c, axis=1)
        return (self.r1<c_distances) & (c_distances<self.r2)

    def generate_grid(self, dx):
        xs = np.linspace(-self.r2, self.r2, np.int_(np.rint(2*self.r2/dx))+1)
        xxs = tuple(xs for _ in range(self.dim))
        grid = np.array(np.meshgrid(*xxs)).T.reshape(-1, self.dim) + self.c
        idx = np.nonzero(self.indicator(grid))
        return grid[idx]

    def u0(self, pts):
        c_distances = np.linalg.norm(pts-self.c, axis=1)
        return (c_distances-self.r1) * (self.r2-c_distances)


class Simulator():
    def __init__(self, domain, p, max_t, dt, dx, n):
        self.domain = domain
        self.p = p
        self.max_t = max_t
        self.dt = dt
        self.dx = dx
        self.n = n

    def one_iteration(self, ui, grid):
        def sw_integral(pt):
            bm = BrownianMotion(pt, self.max_t, self.dt)
            exit_idx = bm.get_exit_idx(self.domain.indicator)
            interpolant = NearestNDInterpolator(grid, np.power(ui, self.p))
            return np.sum(interpolant(bm.bts[:exit_idx])) * self.dt
        def pw_update(pt):
            integrals = Parallel(n_jobs=n_cores)(
                delayed(sw_integral)(pt) for _ in np.arange(self.n))
            return np.mean(integrals)
        return np.vectorize(pw_update, signature="(d)->()")(grid)

    def run(self, epsilon):
        grid = self.domain.generate_grid(self.dx)
        ui = self.domain.u0(grid)
        delta = epsilon + 1
        num = 0
        print("deltas:")
        while delta > epsilon:
            uj = self.one_iteration(ui, grid)
            delta = np.linalg.norm(uj-ui, ord=np.inf)
            print(delta)
            ui = uj
            num += 1
        print(f"{num} iterations")
        return NearestNDInterpolator(grid, ui)
