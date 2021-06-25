import numpy as np
import time, datetime
from tabulate import tabulate

from .bm3d import *


class Simulator3D(ProcessRegister3D):
    def __init__(self, domains, T, dt, dX, n):
        super().__init__()
        self.add_domain(domains)
        self.T = T
        self.dt = dt
        self.dX = dX
        self.n = n

    def generate_samples(self, B0):
        samples = [BrownianMotion3D(self.T, B0, self.dt) for i in range(self.n)]
        self.add_process(samples)

    def reset_simulator(self):
        self.clear_process()


class ExitTimeSimulator3D(Simulator3D):
    def __init__(self, domain, T, dt=0.01, dX=0.1, n=1000):
        super().__init__(domain, T, dt, dX, n)

    def sim_expected_exit_time(self, B0):
        self.generate_samples(B0)
        exit_times = self.get_exit_time(1)
        self.reset_simulator()
        return np.mean(exit_times)
    
    def sim_max_expected_exit_time(self):
        f = self.sim_expected_exit_time
        pts = self.domains[0].get_grid(self.dX)
        times = np.apply_along_axis(f, 0, pts)
        return np.max(times)

    def run(self):
        start_t = time.perf_counter()
        max_exit_time = self.sim_max_expected_exit_time()
        end_t = time.perf_counter()

        report = f"""ExitTimeSimulator3D
            Domain: {self.domains[0]}
            Discretization: dt={self.dt}, dX={self.dX}
            No. Samples: {self.n} per gridpoint
            Max. Exit Time: {max_exit_time}
            Performance: {datetime.timedelta(seconds=end_t-start_t)}"""
        report = [l.strip().split(": ") for l in report.splitlines()]
        print(tabulate(report, headers="firstrow", tablefmt="fancy_grid"))


class OccupationTimeSimulator3D(Simulator3D):
    def __init__(self, D, V, T, dt, dX, n):
        super().__init__([D,V], T, dt, dX, n)

    def sim_expected_occupation_time(self, B0):
        pass

    def sim_min_expected_occupation_time(self):
        f = self.sim_expected_occupation_time
        pts = self.domains[1].get_grid(self.dX)
        times = np.apply_along_axis(f, 0, pts)
        return np.min(times)

    def run(self):
        start_t = time.perf_counter()
        min_occup_time = self.sim_min_expected_occupation_time()
        end_t = time.perf_counter()

        report = f"""OccupationTimeSimulator3D
            Domain D: {self.domains[0]}
            Domain V: {self.domains[1]}
            No. Samples: {self.n} per gridpoint
            Min. Occup. Time: {min_occup_time}
            Performance: {datetime.timedelta(seconds=end_t-start_t)}"""
        report = [l.strip().split(": ") for l in report.splitlines()]
        print(tabulate(report, headers="firstrow", tablefmt="fancy_grid"))
