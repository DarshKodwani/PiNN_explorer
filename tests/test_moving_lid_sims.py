import unittest
import numpy as np
import sys
import os
# Add the directory containing moving_lid_sims.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/simulators')))
from moving_lid_sims import pressure_poisson

class TestMovingLidSims(unittest.TestCase):
    def setUp(self):
        self.nx = 3
        self.ny = 3
        self.lid_velocity = 1.0
        self.p = np.zeros((self.ny, self.nx))
        self.b = np.zeros((self.ny, self.nx))
        self.u = np.zeros((self.ny, self.nx))
        self.u[-1, :] = self.lid_velocity
        self.dx = 1.0
        self.dy = 1.0
        self.nit = 10

    def test_pressure_poisson(self):
        """Test the pressure_poisson function with dummy data."""
        # Expected result
        expected_p = np.array([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])

        # Call the function
        result = pressure_poisson(self.p, self.dx, self.dy, self.b, self.nit)

        # Check if the pressure field is updated correctly
        np.testing.assert_array_equal(result, expected_p)

if __name__ == '__main__':
    unittest.main()