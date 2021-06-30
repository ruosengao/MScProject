import unittest

import numpy as np

from . import oop


class TestBrownianMotion(unittest.TestCase):
    bm1d = oop.BrownianMotion(np.ones(1), 10, 1).bts
    bm2d = oop.BrownianMotion(np.ones(2), 10, 1).bts

    def test_shape(self):
        self.assertEqual(self.bm1d.shape, (10, 1))
        self.assertEqual(self.bm2d.shape, (10, 2))

    def test_initial_point(self):
        self.assertCountEqual(self.bm1d[0], np.ones(1))
        self.assertCountEqual(self.bm2d[0], np.ones(2))

    def test_get_exit_time(self):
        ...

    def test_get_occupation_time(self):
        ...


class TestOpenBall(unittest.TestCase):
    ob1d = oop.OpenBall(np.ones(1), 1)
    ob1d = oop.OpenBall(np.ones(2), 1)

    def test_indicator(self):
        ...

    def test_generate_grid(self):
        ...


class TestOpenAnnulus(unittest.TestCase):
    def test_indicator(self):
        ...

    def test_generate_grid(self):
        ...


class TestExitTimeSimulator(unittest.TestCase):
    ...


class TestOccupationTimeSimulator(unittest.TestCase):
    ...


if __name__ == "__main__":
    unittest.main()
