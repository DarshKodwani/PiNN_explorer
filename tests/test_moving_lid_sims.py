import unittest
import numpy as np
import sys
import os
# Add the directory containing moving_lid_sims.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/simulators')))
from moving_lid_sims import pressure_poisson

class TestMovingLidSims(unittest.TestCase):

    def setUp(self):
        # Set up common parameters for the tests
        self.Lx, self.Ly = 1.0, 1.0
        self.nx, self.ny = 41, 41
        self.dx, self.dy = self.Lx / (self.nx - 1), self.Ly / (self.ny - 1)
        self.nt = 500
        self.dt = 0.001
        self.rho = 1.0
        self.nu = 0.1
        self.lid_velocity = 1.0

        # Initialize fields
        self.u = np.zeros((self.ny, self.nx))
        self.v = np.zeros((self.ny, self.nx))
        self.p = np.zeros((self.ny, self.nx))
        self.b = np.zeros((self.ny, self.nx))
        self.u[-1, :] = self.lid_velocity

    def test_pressure_poisson(self):
        # Dummy data
        p = np.array([[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]])
        dx = 1.0
        dy = 1.0
        b = np.zeros_like(p)
        nit = 10

        # Expected result
        expected_p = np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]])

        # Call the function
        result = pressure_poisson(p, dx, dy, b, nit)

        # Check if the pressure field is updated correctly
        self.assertTrue((result==expected_p).all())


if __name__ == '__main__':
    unittest.main()