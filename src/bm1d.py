import numpy as np
import matplotlib.pyplot as plt

from .bm_interface import *


class BrownianMotion1D(BrownianMotion):
    def __init__(self, T, B0=0.0, dt=0.01):
        self.B0 = B0
        self.dt = dt
        self.T = T
        self._build_process()

    def _build_process(self):
        rng = np.random.default_rng()
        num_increments = np.int64(self.T/self.dt) - 1
        increments = rng.normal(0.0, np.sqrt(self.dt), num_increments)
        Bts = np.cumsum(increments)
        self.Bts = np.insert(Bts, 0, self.B0)

    def plot_up_to(self, ax, t):
        index = np.int64(t/self.dt)
        ts = np.arange(0, self.T, self.dt)
        ax.plot(ts[:index], self.Bts[:index])


class Domain1D(Domain):
    pass


class OpenInterval(Domain1D):
    def __init__(self, lower_bd=-np.inf, upper_bd=np.inf):
        self.bds = np.array([lower_bd, upper_bd])

    def is_in(self, pt):
        return (pt-self.bds[0])*(self.bds[1]-pt) > 0

    def plot_up_to(self, ax, t):
        ax.hlines(self.bds[np.isfinite(self.bds)], 0, t, colors="gray")
        fnt_bds = np.where(np.isfinite(self.bds), self.bds, ax.get_ylim())
        ax.fill_between([0,t], fnt_bds[0], fnt_bds[1], facecolor="gray",
                        alpha=0.5)


class ProcessRegister1D(ProcessRegister):
    def __init__(self):
        self.prcs = np.array([])
        self.dmns = np.array([])

    def add_process(self, processes):
        self.prcs = np.append(self.prcs, processes)
        
    def add_domain(self, domains):
        self.dmns = np.append(self.dmns, domains)

    def get_exit_time(self):
        shape = (self.dmns.shape[0], self.prcs.shape[0], 2)
        dmn_by_prc = np.array(np.meshgrid(self.dmns,self.prcs)).T.reshape(shape)

        # A helper function that computes the pair-wise exit time.
        def pw_exit_time(dp_pair):
            dmn, prc = dp_pair
            filt = dmn.is_in(prc.Bts)
            if np.all(filt):
                return np.nan
            exit_index = np.argmin(filt)
            ts = np.arange(0, prc.T, prc.dt)
            return ts[exit_index]
        
        return np.apply_along_axis(pw_exit_time, 2, dmn_by_prc)

    def plot_up_to(self, end_t):
        fig, axs = plt.subplots(self.dmns.shape[0], 1, squeeze=False)
        for i, dmn in enumerate(self.dmns):
            ax = axs[i,0]
            for j, prc in enumerate(self.prcs):
                prc.plot_up_to(ax, end_t[i,j])
            dmn.plot_up_to(ax, np.max(end_t[i]))
        fig.supxlabel("Time $t$")
        return fig

    def plot_full_process(self):
        f = np.frompyfunc((lambda prc: prc.T),1,1)
        Ts = np.tile(f(self.prcs), (self.dmns.shape[0],1))
        fig = self.plot_up_to(Ts)
        fig.suptitle("Full processes")

    def plot_up_to_exit(self):
        fig = self.plot_up_to(self.get_exit_time())
        fig.suptitle("Processes up to exit times")
