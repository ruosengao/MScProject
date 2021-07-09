import numpy as np


_rng = np.random.default_rng()


# sampling, exit and occupation times
def generate_samples(b0, max_t, dt, n):
    num = np.int_(np.rint(max_t/dt))
    increments = _rng.normal(0., np.sqrt(dt), size=(n, num-1, np.size(b0)))
    return np.cumsum(np.insert(increments, 0, b0, axis=1), axis=1)

def get_exit_times(samples, dt, indicator):
    bool_arr = indicator(samples)
    idx = np.argmin(bool_arr, axis=1)
    if not np.all(idx):
        raise RuntimeError("exit time is out of reach")
    return idx * dt

def get_occupation_times(samples, dt, indicator, stop_times):
    bool_arr = indicator(samples)
    idx = np.int_(np.rint(stop_times/dt))
    rw_func = lambda row, i: np.sum(row[:i])
    func = np.vectorize(rw_func, signature="(m),()->()")
    return func(bool_arr, idx) * dt


# domain functions
def indicator_func(domain):
    name, *para = domain
    if name == "OpenBall":
        c, r = para
        return lambda pts: np.linalg.norm(pts-c, axis=-1) < r
    else: # name == "OpenAnnulus"
        c, r1, r2 = para
        return lambda pts: r1 < np.linalg.norm(pts-c, axis=-1) < r2

def generate_grid(domain, dx):
    name, *para = domain
    if name == "OpenBall":
        c, r = para
        xs = np.linspace(-r, r, np.int_(np.rint(2*r/dx))+1)
    else: # name == "OpenAnnulus"
        c, _, r2 = para
        xs = np.linspace(-r2, r2, np.int_(np.rint(2*r2/dx))+1)
    xxs = tuple(xs for _ in range(np.size(c)))
    grid = np.array(np.meshgrid(*xxs)).T.reshape(-1, np.size(c)) + c
    indices = np.nonzero(indicator_func(domain)(grid))
    return grid[indices]


# simulator functions
def simulate_expected_exit_time(indicator, b0, max_t, dt, n):
    samples = generate_samples(b0, max_t, dt, n)
    exit_times = get_exit_times(samples, dt, indicator)
    return np.mean(exit_times)

def simulate_expected_occupation_time(ind_d, ind_v, b0, max_t, dt, n):
    samples = generate_samples(b0, max_t, dt, n)
    stop_times = get_exit_times(samples, dt, ind_d)
    occup_times = get_occupation_times(samples, dt, ind_v, stop_times)
    return np.mean(occup_times)

def simulate_max_expected_exit_time(domain, max_t, dt, dx, n):
    indicator = indicator_func(domain)
    grid = generate_grid(domain, dx)
    pw_f = lambda pt: simulate_expected_exit_time(indicator, pt, max_t, dt, n)
    times = np.vectorize(pw_f)(grid)
    return np.max(times)

def simulate_min_expected_occupation_time(domain_d, domain_v, max_t, dt, dx, n):
    indicator_d = indicator_func(domain_d)
    indicator_v = indicator_func(domain_v)
    grid = generate_grid(domain_v, dx)
    pw_f = lambda pt: simulate_expected_occupation_time(
        indicator_d, indicator_v, pt, max_t, dt, n)
    times = np.vectorize(pw_f)(grid)
    return np.min(times)


def main(simulator, **kwargs):
    if simulator == "exit-time":
        return simulate_max_expected_exit_time(kwargs["domain"],
            kwargs["max_t"], kwargs["dt"], kwargs["dx"], kwargs["n"])
    else: # simulator == "occup-time"
        return simulate_min_expected_occupation_time(
            kwargs["domain_d"], kwargs["domain_v"],
            kwargs["max_t"], kwargs["dt"], kwargs["dx"], kwargs["n"])
