import unittest

import numpy as np

from . import procedural as pro


class TestDomainFunctions(unittest.TestCase):
    ob1d = ("OpenBall", np.ones(1), 1.)
    ob2d = ("OpenBall", np.ones(2), 1.)

    def test_generate_samples(self):
        sample1d, _ = pro.generate_samples(np.ones(1), 10, 1, 5)
        sample2d, _ = pro.generate_samples(np.ones(2), 10, 1, 5)
        self.assertEqual(sample1d.shape, (5, 10, 1))
        self.assertEqual(sample2d.shape, (5, 10, 2))
        self.assertCountEqual(sample1d[0,0], np.ones(1))
        self.assertCountEqual(sample2d[0,0], np.ones(2))

    def test_indicator_func(self):
        ...


if __name__ == "__main__":
    unittest.main()
