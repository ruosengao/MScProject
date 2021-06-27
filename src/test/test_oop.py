import unittest

import numpy as np

from .. import oop


class Test(unittest.TestCase):
    domain_d = oop.OpenBall(np.zeros(2,), 2.)
    domain_v = oop.OpenBall(np.zeros(2,), 1.)

    def test_exit_time_sim(self):
        sim = oop.ExitTimeSimulator(self.domain_v, 10, .01, 1, 100)
        sim.run()

    def test_occupation_time_sim(self):
        sim = oop.OccupationTimeSimulator(
            self.domain_d, self.domain_v, 20, .01, 1, 100
        )
        sim.run()

if __name__ == "__main__":
    unittest.main()
