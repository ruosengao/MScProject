import numpy as np


_rng = np.random.default_rng()


# sampling, exit and occupation times
def generate_sample(b0, max_t, dt):
    num = np.int_(np.rint(max_t/dt))
    increments = _rng.normal(0., np.sqrt(dt), size=(num-1, np.size(b0)))
    return np.cumsum(np.insert(increments, 0, b0, axis=0), axis=0)

def get_exit_time(sample, dt, indicator):
    bool_arr = indicator(sample)
    idx = np.argmin(bool_arr)
    if idx == 0:
        raise RuntimeError("exit time is out of reach")
    return idx * dt

def get_occupation_time(sample, dt, indicator, stop_time):
    bool_arr = indicator(sample)
    idx = np.int_(np.rint(stop_time/dt))
    return np.sum(bool_arr[:idx]) * dt


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
    def sw_f(_):
        sample = generate_sample(b0, max_t, dt)
        return get_exit_time(sample, dt, indicator)
    exit_times = np.vectorize(sw_f)(np.arange(n))
    return np.mean(exit_times)

def simulate_expected_occupation_time(ind_d, ind_v, b0, max_t, dt, n):
    def sw_f(_):
        sample = generate_sample(b0, max_t, dt)
        exit_time = get_exit_time(sample, dt, ind_d)
        return get_occupation_time(sample, dt, ind_v, exit_time)
    occupation_times = np.vectorize(sw_f)(np.arange(n))
    return np.mean(occupation_times)

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
