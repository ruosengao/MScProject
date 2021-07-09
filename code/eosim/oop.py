from abc import ABC, abstractmethod

import numpy as np


_rng = np.random.default_rng()


class BrownianMotion():
    def __init__(self, b0, max_t, dt):
        self.dt = dt
        num = np.int_(np.rint(max_t/dt))
        increments = _rng.normal(0., np.sqrt(dt), size=(num-1, np.size(b0)))
        self.bts = np.cumsum(np.insert(increments, 0, b0, axis=0), axis=0)

    def get_exit_time(self, indicator):
        bool_arr = indicator(self.bts)
        idx = np.argmin(bool_arr)
        if idx == 0:
            raise RuntimeError("exit time is out of reach")
        return idx * self.dt

    def get_occupation_time(self, indicator, stop_time):
        bool_arr = indicator(self.bts)
        idx = np.int_(np.rint(stop_time/self.dt))
        return np.sum(bool_arr[:idx]) * self.dt


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
        return self.r1 < c_distances < self.r2

    def generate_grid(self, dx):
        xs = np.linspace(-self.r2, self.r2, np.int_(np.rint(2*self.r2/dx))+1)
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

    @abstractmethod
    def run(self):
        pass


class ExitTimeSimulator(Simulator):
    def __init__(self, domain, max_t, dt, dx, n):
        super().__init__(max_t, dt, dx, n)
        self.domain = domain

    def expected_exit_time(self, b0):
        indicator = self.domain.indicator
        def sw_f(_):
            bm = BrownianMotion(b0, self.max_t, self.dt)
            return bm.get_exit_time(indicator)
        times = np.vectorize(sw_f)(np.arange(self.n))
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
        indicator_d = self.domain_d.indicator
        indicator_v = self.domain_v.indicator
        def sw_f(_):
            bm = BrownianMotion(b0, self.max_t, self.dt)
            exit_time = bm.get_exit_time(indicator_d)
            return bm.get_occupation_time(indicator_v, exit_time)
        times = np.vectorize(sw_f)(np.arange(self.n))
        return np.mean(times)

    def run(self):
        grid = self.domain_v.generate_grid(self.dx)
        times = np.vectorize(self.expected_occupation_time)(grid)
        return np.min(times)

    min_expected_occupation_time = run


def main(simulator, **kwargs):
    def domain_parser(domain):
        name, *para = domain
        if name == OpenBall.__name__:
            return OpenBall(*para)
        else: # name == OpenAnnulus.__name__
            return OpenAnnulus(*para)
    
    if simulator == "exit-time":
        domain = domain_parser(kwargs["domain"])
        sim = ExitTimeSimulator(domain,
            kwargs["max_t"], kwargs["dt"], kwargs["dx"], kwargs["n"])
        return sim.run()
    else: # simulator == "occup-time"
        domain_d = domain_parser(kwargs["domain_d"])
        domain_v = domain_parser(kwargs["domain_v"])
        sim = OccupationTimeSimulator(domain_d, domain_v,
            kwargs["max_t"], kwargs["dt"], kwargs["dx"], kwargs["n"])
        return sim.run()
