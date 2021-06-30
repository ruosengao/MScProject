from abc import ABC, abstractmethod

import numpy as np


_rng = np.random.default_rng()
_dim = lambda pt: np.array(pt).size


class BrownianMotion():
    def __init__(self, b0, max_t, dt):
        self.dt = dt
        num_steps = np.int_(max_t/dt) - 1
        increments = _rng.normal(0., np.sqrt(dt), size=(num_steps, _dim(b0)))
        self.bts = np.cumsum(np.insert(increments, 0, b0, axis=0), axis=0)
        self.ts = np.linspace(0., max_t, np.int_(max_t/dt), False)

    def get_exit_time(self, indicator):
        bool_arr = indicator(self.bts)
        idx = np.argmin(bool_arr)
        if idx == 0:
            raise RuntimeError("exit time is out of reach")
        return self.ts[idx]

    def get_occupation_time(self, indicator, t):
        bool_arr = indicator(self.bts)[:np.int_(t/self.dt)]
        return np.sum(bool_arr) * self.dt


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


class OpenBall(Domain):
    def __init__(self, c, r):
        self.dim = _dim(c)
        self.c = c
        self.r = r

    def __str__(self):
        return f"{self.dim}D {type(self).__name__} ({self.c}, {self.r})"

    def indicator(self, pts):
        c_distances = np.linalg.norm(pts-self.c, axis=1)
        return c_distances < self.r

    def generate_grid(self, dx):
        xs = np.linspace(-self.r, self.r, np.int_(2*self.r/dx)+1)
        xxs = tuple(xs for _ in range(self.dim))
        grid = np.array(np.meshgrid(*xxs)).T.reshape(-1, self.dim) + self.c
        idx = np.nonzero(self.indicator(grid))
        return grid[idx]


class OpenAnnulus(Domain):
    def __init__(self, c, r1, r2):
        self.dim = _dim(c)
        self.c = c
        self.r1 = r1
        self.r2 = r2

    def __str__(self):
        return (f"{self.dim}D {type(self).__name__} "
                f"({self.c}, {self.r1}, {self.r2})")

    def indicator(self, pts):
        c_distances = np.linalg.norm(pts-self.c, axis=1)
        return self.r1 < c_distances < self.r2

    def generate_grid(self, dx):
        xs = np.linspace(-self.r2, self.r2, np.int_(2*self.r2/dx)+1)
        xxs = tuple(xs for _ in range(self.dim))
        grid = np.array(np.meshgrid(*xxs)).T.reshape(-1, self.dim) + self.c
        idx = np.nonzero(self.indicator(grid))
        return grid[idx]


class Simulator(ABC):
    def __init__(self, max_t, dt, dx, n):
        self.max_t = max_t
        self.dt = dt
        self.dx = dx
        self.n = n

    def generate_samples(self, b0):
        return [BrownianMotion(b0, self.max_t, self.dt) for _ in range(self.n)]

    @abstractmethod
    def run(self):
        pass


class ExitTimeSimulator(Simulator):
    def __init__(self, domain, max_t, dt, dx, n):
        super().__init__(max_t, dt, dx, n)
        self.domain = domain

    def expected_exit_time(self, b0):
        samples = self.generate_samples(b0)
        indicator = self.domain.indicator
        sw_f = lambda bm: bm.get_exit_time(indicator)
        times = np.vectorize(sw_f)(samples)
        return np.mean(times)

    def run(self):
        grid = self.domain.generate_grid(self.dx)
        times = np.vectorize(self.expected_exit_time)(grid)
        return np.max(times)

    max_expected_exit_time = run


class OccupationTimeSimulator(Simulator):
    def __init__(self, domain_d, domain_v, max_t, dt, dx, n):
        super().__init__(max_t, dt, dx, n)
        self.domain_d = domain_d
        self.domain_v = domain_v

    def expected_occupation_time(self, b0):
        samples = self.generate_samples(b0)
        indicator_d = self.domain_d.indicator
        sw_f = lambda bm: bm.get_exit_time(indicator_d)
        exit_times = np.vectorize(sw_f)(samples)
        
        indicator_v = self.domain_v.indicator
        sw_g = lambda bm, t: bm.get_occupation_time(indicator_v, t)
        occupation_times = np.vectorize(sw_g)(samples, exit_times)
        return np.mean(occupation_times)

    def run(self):
        grid = self.domain_v.generate_grid(self.dx)
        times = np.vectorize(self.expected_occupation_time)(grid)
        return np.min(times)

    min_expected_occupation_time = run
