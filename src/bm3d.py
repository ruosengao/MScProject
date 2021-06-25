import numpy as np
import matplotlib.pyplot as plt

from .bm_interface import *


class BrownianMotion3D(BrownianMotion):
    def __init__(self, T, B0=np.zeros((3,)), dt=0.01):
        self.B0 = B0
        self.dt = dt
        self.T = T
        self._build_process()

    def _build_process(self):
        rng = np.random.default_rng()
        num_increments = np.int64(self.T/self.dt) - 1
        increments = rng.normal(0., np.sqrt(self.dt), size=(3, num_increments))
        Bts = np.cumsum(increments, axis=1)
        self.Bts = np.insert(Bts, 0, self.B0, axis=1)
        self.ts = np.arange(0, self.T, self.dt)

    def _get_index(self, t):
        return np.int64(t/self.dt)

    def get_hitting_time(self, indicator, n):
        bi_arr = indicator(self.Bts)
        test_arr = bi_arr - np.roll(bi_arr, 1)
        test_arr[0] = bi_arr[0] - 0
        indices = np.nonzero(test_arr==1)
        try:
            return self.ts[indices][n-1]
        except IndexError:
            return np.nan

    def get_occupation_time(self, indicator, t):
        bi_arr = indicator(self.Bts)[:self._get_index(t)]
        return np.sum(bi_arr) * self.dt

    def add_plot_to(self, ax, t):
        pass


class Domain3D(Domain):
    pass


class OpenBall3D(Domain3D):
    def __init__(self, c=np.zeros((3,)), r=1.0):
        self.c = c
        self.r = r

    def __str__(self):
        return f"OpenBall3D ({self.c}, {self.r})"

    def is_in(self, pt):
        dist_to_c = np.linalg.norm(pt-self.c)
        return dist_to_c < self.r

    def get_indicator(self):
        I = lambda pt: np.apply_along_axis(self.is_in, 0, pt)
        return lambda pt: np.int64(I(pt))

    def get_grid(self, dX):
        Xs = np.arange(-self.r, self.r, dX)
        grid = np.array(np.meshgrid(Xs, Xs, Xs)).T.reshape(3,-1)
        indices = np.nonzero(self.get_indicator()(grid))[0]
        return grid[:,indices]

    def add_plot_to(self, ax, t):
        pass


class OpenAnnulus3D(Domain3D):
    def __init__(self, c=np.zeros((3,)), r=0.0, R=1.0):
        self.c = c
        self.r = r
        self.R = R

    def __str__(self):
        return f"OpenAnnulus3D ({self.c}, {self.r}, {self.R})"

    def is_in(self, pt):
        dist_to_c = np.linalg.norm(pt-self.c)
        return dist_to_c > self.r and dist_to_c < self.R

    def get_indicator(self):
        I = lambda pt: np.apply_along_axis(self.is_in, 0, pt)
        return lambda pt: np.int64(I(pt))

    def get_grid(self, step=0.5):
        Xs = np.arange(-self.R, self.R, step)
        grid = np.array(np.meshgrid(Xs, Xs, Xs)).T.reshape(3,-1)
        indices = np.nonzero(self.get_indicator()(grid))[0]
        return grid[:,indices]

    def add_plot_to(self, ax, t):
        pass


class ProcessRegister3D(ProcessRegister):
    def __init__(self):
        self.processes = np.array([])
        self.domains = np.array([])

    def add_process(self, processes):
        self.processes = np.append(self.processes, processes)
        
    def add_domain(self, domains):
        self.domains = np.append(self.domains, domains)

    def clear_process(self):
        self.processes = np.array([])

    def get_dp_array(self):
        shape = (self.domains.shape[0], self.processes.shape[0], 2)
        return np.array(np.meshgrid(self.domains,self.processes)). \
               T.reshape(shape)

    def get_entry_time(self, n):
        def ew_entry_time(dp_pair):
            dmn, prc = dp_pair
            indicator = dmn.get_indicator()
            return prc.get_hitting_time(indicator, n)
        return np.apply_along_axis(ew_entry_time, 2, self.get_dp_array())

    def get_exit_time(self, n):
        def ew_exit_time(dp_pair):
            dmn, prc = dp_pair
            indicator = lambda pt: np.int64(
                        np.logical_not(dmn.get_indicator()(pt)))
            return prc.get_hitting_time(indicator, n)
        return np.apply_along_axis(ew_exit_time, 2, self.get_dp_array())

    def get_occupation_time(self, ts):
        def ew_occupation_time(dpt_arr):
            dmn, prc, t = dpt_arr
            indicator = dmn.get_indicator()
            return prc.get_occupation_time(indicator, t)
        dp_arr = self.get_dp_array()
        ts = np.expand_dims(ts, axis=2)
        dpt_arr = np.append(dp_arr, ts, axis=2)
        return np.apply_along_axis(ew_occupation_time, 2, dpt_arr)
