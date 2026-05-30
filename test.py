import unittest
import numpy as np
from LtVisualizer.tensor import Tensor


class TestTensor(unittest.TestCase):
    def setUp(self):
        # Initial state: Identity-like vectors [1, 0] and [0, 1]
        self.t = Tensor(1, 0, 0, 1)

    def test_initialization(self):
        with self.assertRaises(TypeError):
            Tensor("1", 0, 0, 1)

    def test_reflect_on_x(self):
        self.t.reflect_on_x()
        # Reflection on x means y values change sign: [[1, 0], [0, -1]]
        expected = np.array([[1, 0], [0, -1]])
        np.testing.assert_array_equal(self.t.obj, expected)

    def test_reflect_on_y(self):
        self.t.reflect_on_y()
        # Reflection on y means x values change sign: [[-1, 0], [0, 1]]
        expected = np.array([[-1, 0], [0, 1]])
        np.testing.assert_array_equal(self.t.obj, expected)

    def test_scaling(self):
        self.t.scaling_all_direction(3)
        expected = np.array([[3, 0], [0, 3]])
        np.testing.assert_array_equal(self.t.obj, expected)

    def test_horizontal_sheer(self):
        # Horizontal sheer: x' = x + m*y
        # Input [1, 0], [0, 1] with m=2
        # Vector 1: [1 + 2*0, 0] = [1, 0]
        # Vector 2: [0 + 2*1, 1] = [2, 1]
        self.t.horizontal_sheer(2)
        expected = np.array([[1, 0], [2, 1]])
        np.testing.assert_array_equal(self.t.obj, expected)

    def test_vertical_sheer(self):
        # Vertical sheer: y' = y + m*x
        self.t.vertical_sheer(3)
        # Vector 1: [1, 0 + 3*1] = [1, 3]
        # Vector 2: [0, 1 + 3*0] = [0, 1]
        expected = np.array([[1, 3], [0, 1]])
        np.testing.assert_array_equal(self.t.obj, expected)


if __name__ == "__main__":
    unittest.main()
