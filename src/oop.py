import datetime
import time
from abc import ABC, abstractmethod

import numpy as np
from tabulate import tabulate


_rng = np.random.default_rng()

def _dim(pt):
    try:
        return pt.size
    except AttributeError:
        return 1


class BrownianMotion():
    def __init__(self, b0, max_t, dt):
        self.dt = dt
        n_increments = np.int_(max_t/dt) - 1
        increments = _rng.normal(0., np.sqrt(dt), size=(n_increments, _dim(b0)))
        self.bts = np.cumsum(np.insert(increments, 0, b0, axis=0), axis=0)
        self.ts = np.linspace(0., max_t, np.int_(max_t/dt), False)

    def get_exit_time(self, indicator, n=1):
        bi_arr = indicator(self.bts)
        test_arr = bi_arr - np.roll(bi_arr, 1)
        test_arr[0] = bi_arr[0] - 1
        indices = np.nonzero(test_arr==-1)
        try:
            return self.ts[indices][n-1]
        except IndexError:
            return np.nan

    def get_occupation_time(self, indicator, t):
        bi_arr = indicator(self.bts)[:np.int_(t/self.dt)]
        return np.sum(bi_arr) * self.dt

    def add_plot_to(self, t, ax):
        ...


class Domain(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def is_in(self, pt):
        pass

    def get_indicator(self):
        return lambda pt: np.int_(np.apply_along_axis(self.is_in, 1, pt))

    @abstractmethod
    def gen_grid(self, dx):
        pass


class OpenBall(Domain):
    def __init__(self, c, r):
        self.dim = _dim(c)
        self.c = c
        self.r = r

    def __str__(self):
        return f"{self.dim}D {type(self).__name__} ({self.c}, {self.r})"

    def is_in(self, pt):
        dist_to_c = np.linalg.norm(pt-self.c)
        return dist_to_c<self.r

    def gen_grid(self, dx):
        xs = np.linspace(-self.r, self.r, np.int_(2*self.r/dx)+1)
        xxs = tuple(xs for _ in range(self.dim))
        grid = np.array(np.meshgrid(*xxs)).T.reshape(-1, self.dim)
        indices = np.nonzero(self.get_indicator()(grid))
        return grid[indices]

    def add_plot_to(self, ax):
        ...


class OpenAnnulus(Domain):
    def __init__(self, c, r1, r2):
        self.dim = _dim(c)
        self.c = c
        self.r1 = r1
        self.r2 = r2

    def __str__(self):
        return (f"{self.dim}D {type(self).__name__} "
                f"({self.c}, {self.r1}, {self.r2})")

    def is_in(self, pt):
        dist_to_c = np.linalg.norm(pt-self.c)
        return dist_to_c>self.r1 and dist_to_c<self.r2

    def gen_grid(self, dx):
        xs = np.linspace(-self.r2, self.r2, np.int_(2*self.r2/dx)+1)
        xxs = tuple(xs for _ in range(self.dim))
        grid = np.array(np.meshgrid(*xxs)).T.reshape(-1, self.dim)
        indices = np.nonzero(self.get_indicator()(grid))
        return grid[indices]


class Simulator(ABC):
    def __init__(self, max_t, dt, dx, n):
        self.max_t = max_t
        self.dt = dt
        self.dx = dx
        self.n = n
        self.report = [
            [type(self).__name__],
            ["Grid", f"max_t={self.max_t}, dt={self.dt}, dx={self.dx}"],
            ["No. Samples", f"{self.n} per gridpoint"],
            ["Estimate", None],
            ["Performance", None]
        ]

    def gen_samples(self, b0):
        return [BrownianMotion(b0, self.max_t, self.dt) for _ in range(self.n)]

    @abstractmethod
    def main_simulator(self):
        pass

    def run(self):
        t0 = time.perf_counter()
        estimate = self.main_simulator()
        t1 = time.perf_counter()
        self.report[-1][1] = datetime.timedelta(seconds=t1-t0)
        self.report[-2][1] = estimate
        print(tabulate(self.report, headers="firstrow", tablefmt="fancy_grid"))


class ExitTimeSimulator(Simulator):
    def __init__(self, domain, max_t, dt, dx, n):
        super().__init__(max_t, dt, dx, n)
        self.report.insert(1, ["Domain", domain])
        self.domain = domain

    def sim_expected_exit_time(self, b0):
        samples = self.gen_samples(b0)
        indicator = self.domain.get_indicator()
        pw_f = lambda bm: bm.get_exit_time(indicator)
        times = np.vectorize(pw_f)(samples)
        return np.mean(times)

    def main_simulator(self):
        grid = self.domain.gen_grid(self.dx)
        times = np.vectorize(self.sim_expected_exit_time)(grid)
        return np.max(times)

    sim_max_expected_exit_time = main_simulator


class OccupationTimeSimulator(Simulator):
    def __init__(self, domain_d, domain_v, max_t, dt, dx, n):
        super().__init__(max_t, dt, dx, n)
        self.report.insert(1, ["Domain D, V", f"{domain_d},\n{domain_v}"])
        self.domain_d = domain_d
        self.domain_v = domain_v

    def sim_expected_occupation_time(self, b0):
        samples = self.gen_samples(b0)
        indicator_d = self.domain_d.get_indicator()
        pw_f = lambda bm: bm.get_exit_time(indicator_d)
        exit_times = np.vectorize(pw_f)(samples)
        
        indicator_v = self.domain_v.get_indicator()
        pw_g = lambda bm, t: bm.get_occupation_time(indicator_v, t)
        occup_times = np.vectorize(pw_g)(samples, exit_times)
        return np.mean(occup_times)

    def main_simulator(self):
        grid = self.domain_v.gen_grid(self.dx)
        times = np.vectorize(self.sim_expected_occupation_time)(grid)
        return np.min(times)

    sim_min_expected_occupation_time = main_simulator
