import unittest
import numpy as np
import sys
import os
# Add the directory containing moving_lid_sims.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/simulators')))
from moving_lid_sims import build_up_b, pressure_poisson, cavity_flow

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

    def test_build_up_b(self):
        # Test the build_up_b function
        build_up_b(self.b, self.rho, self.dt, self.u, self.v, self.dx, self.dy)
        # Check if the build-up term is calculated correctly
        expected_b_values = np.zeros_like(self.b)  # Replace with actual expected values
        self.assertTrue(np.allclose(self.b, expected_b_values))

    def test_pressure_poisson(self):
        # Test the pressure_poisson function
        p = pressure_poisson(self.p, self.dx, self.dy, self.b)
        # Check if the pressure field is updated correctly
        expected_p_values = np.zeros_like(self.p)  # Replace with actual expected values
        self.assertTrue(np.allclose(p, expected_p_values))

    def test_cavity_flow(self):
        # Test the cavity_flow function
        u, v, p = cavity_flow(self.nt, self.u, self.v, self.dt, self.dx, self.dy, self.p, self.rho, self.nu)
        # Check if the velocity and pressure fields are updated correctly
        expected_u_values = np.zeros_like(self.u)  # Replace with actual expected values
        expected_v_values = np.zeros_like(self.v)  # Replace with actual expected values
        expected_p_values = np.zeros_like(self.p)  # Replace with actual expected values
        self.assertTrue(np.allclose(u, expected_u_values))
        self.assertTrue(np.allclose(v, expected_v_values))
        self.assertTrue(np.allclose(p, expected_p_values))

if __name__ == '__main__':
    unittest.main()