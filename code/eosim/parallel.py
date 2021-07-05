import dask
import dask.array as da
import numpy as np


_rng = np.random.default_rng()
_dim = lambda pt: np.array(pt).size


# sampling
def generate_samples(b0, max_t, dt, n):
    num = np.int_(np.rint(max_t/dt))
    increments = _rng.normal(0., np.sqrt(dt), size=(n, num-1, _dim(b0)))
    samples = np.cumsum(np.insert(increments, 0, b0, axis=1), axis=1)
    samples = da.from_array(samples, chunks=("auto", -1, -1))
    ts = np.linspace(0., max_t, num, False)
    return samples, ts


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

    xxs = tuple(xs for _ in range(_dim(c)))
    grid = np.array(np.meshgrid(*xxs)).T.reshape(-1, _dim(c)) + c
    indices = np.nonzero(indicator_func(domain)(grid))
    return grid[indices]


# exit and occupation times
@dask.delayed
def get_exit_times(samples, ts, indicator):
    bool_arr = indicator(samples)
    idx = np.argmin(bool_arr, axis=1)
    if not np.all(idx):
        raise RuntimeError("exit time is out of reach")
    return ts[idx]

@dask.delayed
def get_occupation_times(samples, indicator, stop_times, dt):
    bool_arr = indicator(samples)
    idx = np.int_(np.rint(stop_times/dt))
    rw_func = lambda row, i: np.sum(row[:i])
    func = np.vectorize(rw_func, signature="(m),()->()")
    return func(bool_arr, idx) * dt
    

# simulator functions
def simulate_expected_exit_time(indicator, b0, max_t, dt, n):
    samples, ts = generate_samples(b0, max_t, dt, n)
    exit_times = get_exit_times(samples, ts, indicator)
    return np.mean(exit_times).compute()

def simulate_expected_occupation_time(ind_d, ind_v, b0, max_t, dt, n):
    samples, ts = generate_samples(b0, max_t, dt, n)
    stop_times = get_exit_times(samples, ts, ind_d)
    occup_times = get_occupation_times(samples, ind_v, stop_times, dt)
    return np.mean(occup_times).compute()

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
