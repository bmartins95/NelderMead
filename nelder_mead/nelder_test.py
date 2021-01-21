import unittest
import numpy as np
from nelder_mead import NelderMead

class TestNelderMead(NelderMead):
    def buildSimplexPoints(self):
        self.simplex = np.vstack([np.eye(len(self.f_variables), dtype = float), self.f_variables])
        for index, value in enumerate(self.f_variables):
            h = 0.00025 if abs(value) < 1.0e-22 else 0.05
            self.simplex[index,:] = self.simplex[index,:] * h

class TestInit(unittest.TestCase):
    def setUp(self):
        self.nelder = TestNelderMead([1.0, 2.0, 0.0])
        self.simplex_to_check = np.array([[5.0e-02, 0.0e+00, 0.0e+00],
                                          [0.0e+00, 5.0e-02, 0.0e+00],
                                          [0.0e+00, 0.0e+00, 2.5e-04],
                                          [1.0e+00, 2.0e+00, 0.0e+00]])

    def testSimplexBuilder(self):
        self.assertTrue(np.allclose(self.nelder.simplex, self.simplex_to_check))

class TestGatherRoutine(unittest.TestCase):
    def setUp(self):
        self.f_variables = [1.0, 2.0, 0.0]
        self.nelder = TestNelderMead(self.f_variables)

        self.f_values_to_check = np.array([3.0e+00, 5.0e-02, 5.0e-02])

    def testGatherRoutine(self):
        for value in self.f_values_to_check:
            self.assertEqual(sum(self.f_variables), value)
            self.nelder.run(sum(self.f_variables))

        self.f_values_to_check = np.array([5.0e-02, 5.0e-02, 0.0e+00, 3.0e+00])
        self.assertTrue(np.allclose(self.nelder.f_values, self.f_values_to_check))

class TestReflectionSubroutine(unittest.TestCase):
    def setUp(self):
        self.f_variables = [1.0, 2.0, 0.0]
        self.nelder = TestNelderMead(self.f_variables)

        self.simplex_to_check1 = np.array([[0.0e+00, 0.0e+00, 2.5e-04],
                                          [5.0e-02, 0.0e+00, 0.0e+00],
                                          [0.0e+00, 5.0e-02, 0.0e+00],
                                          [1.0e+00, 2.0e+00, 0.0e+00]])

        self.simplex_to_check2 = np.array([[0.0e+00, 0.0e+00, 2.5e-04],
                                          [-2.9/3.0, -5.9/3.0, 0.0005/3.0],
                                          [5.0e-02, 0.0e+00, 0.0e+00],
                                          [0.0e+00, 5.0e-02, 0.0e+00]])

        self.f_values_to_check1 = np.array([2.5e-04, 5.0e-02, 5.0e-02, 3.0e+00])
        self.f_values_to_check2 = np.array([2.5e-04, 5.0e-04, 5.0e-02, 5.0e-02])
        self.centroid_to_check1 = np.array([0.05/3.0, 0.05/3.0, 0.00025/3.0])
        self.centroid_to_check2 = np.array([-2.75/9.0, -5.9/9.0, 1.25e-03/9.0])
        self.x_r_to_check = np.array([-2.9/3.0, -5.9/3.0, 0.0005/3.0])

    def testPreReflection(self):
        for i in range(0, 4):
            self.nelder.run(sum(self.f_variables))

        self.assertTrue(np.allclose(self.nelder.simplex, self.simplex_to_check1))
        self.assertTrue(np.allclose(self.nelder.f_values, self.f_values_to_check1))
        self.assertTrue(np.allclose(self.nelder.c, self.centroid_to_check1))
        self.assertTrue(np.allclose(self.nelder.x_r, self.x_r_to_check))
        self.assertTrue(np.allclose(self.nelder.x_r, self.nelder.f_variables))


    def testReflection(self):
        for i in range(0, 4):
            self.nelder.run(sum(self.f_variables))

        self.nelder.run(5.0e-04)
        self.assertTrue(np.allclose(self.nelder.simplex, self.simplex_to_check2))
        self.assertTrue(np.allclose(self.nelder.f_values, self.f_values_to_check2))
        self.assertTrue(np.allclose(self.nelder.c, self.centroid_to_check2))

class TestExpansionSubroutine(unittest.TestCase):
    def setUp(self):
        self.f_variables = [1.0, 2.0, 0.0]
        self.nelder = TestNelderMead(self.f_variables)

        self.simplex_to_check3 = np.array([[-5.85/3.0, -11.85/3.0, 7.5e-04/3.0],
                                          [0.0e+00, 0.0e+00, 2.5e-04],
                                          [5.0e-02, 0.0e+00, 0.0e+00],
                                          [0.0e+00, 5.0e-02, 0.0e+00]])

        self.simplex_to_check4 = np.array([[-2.9/3.0, -5.9/3.0, 0.0005/3.0],
                                          [0.0e+00, 0.0e+00, 2.5e-04],
                                          [5.0e-02, 0.0e+00, 0.0e+00],
                                          [0.0e+00, 5.0e-02, 0.0e+00]])

        self.f_values_to_check3 = np.array([1.0e-04, 2.5e-04, 5.0e-02, 5.0e-02])
        self.f_values_to_check4 = np.array([2.0e-04, 2.5e-04, 5.0e-02, 5.0e-02])
        self.x_e_to_check = np.array([-5.85/3.0, -11.85/3.0, 7.5e-04/3.0])

    def testExpansionBegin(self):
        for i in range(0, 4):
            self.nelder.run(sum(self.f_variables))

        self.nelder.run(2.0e-04)
        self.assertTrue(np.allclose(self.nelder.x_e, self.x_e_to_check))
        self.assertTrue(np.allclose(self.nelder.f_variables, self.x_e_to_check))

    def testGoodExpansion(self):
        for i in range(0, 4):
            self.nelder.run(sum(self.f_variables))

        self.nelder.run(2.0e-04)
        self.nelder.run(1.0e-04)
        self.assertTrue(np.allclose(self.nelder.simplex, self.simplex_to_check3))
        self.assertTrue(np.allclose(self.nelder.f_values, self.f_values_to_check3))

    def testBadExpansion(self):
        for i in range(0, 4):
            self.nelder.run(sum(self.f_variables))

        self.nelder.run(2.0e-04)
        self.nelder.run(3.0e-04)
        self.assertTrue(np.allclose(self.nelder.simplex, self.simplex_to_check4))
        self.assertTrue(np.allclose(self.nelder.f_values, self.f_values_to_check4))

class TestContractionSubroutine(unittest.TestCase):
    def setUp(self):
        self.f_variables = [1.0, 2.0, 0.0]
        self.nelder = TestNelderMead(self.f_variables)

        self.simplex_to_check5 = np.array([[0.0e+00, 0.0e+00, 2.5e-04],
                                          [5.0e-02, 0.0e+00, 0.0e+00],
                                          [0.0e+00, 5.0e-02, 0.0e+00],
                                          [5.08333333e-01, 1.00833333e+00, 4.16666667e-05]])

        self.simplex_to_check6 = np.array([[0.0e+00, 0.0e+00, 2.5e-04],
                                          [5.0e-02, 0.0e+00, 0.0e+00],
                                          [0.0e+00, 5.0e-02, 0.0e+00],
                                          [-4.75e-01, -9.75e-01,  1.25e-04]])

        self.f_values_to_check5 = np.array([2.5e-04, 5.0e-02, 5.0e-02, 2.0e+00])
        self.f_values_to_check6 = np.array([2.5e-04, 5.0e-02, 5.0e-02, 1.0e+00])
        self.centroid_to_check3 = np.array([1.66666667e-02, 1.66666667e-02, 8.33333333e-05])
        self.centroid_to_check4 = np.array([1.66666667e-02, 1.66666667e-02, 8.33333333e-05])
        self.x_c_inside_to_check = np.array([5.08333333e-01, 1.00833333e+00, 4.16666667e-05])
        self.x_c_outside_to_check = np.array([-4.75e-01, -9.75e-01,  1.25e-04])

    def testContractionInside(self):
        for i in range(0, 4):
            self.nelder.run(sum(self.f_variables))

        self.nelder.run(4.0e+00)
        self.nelder.run(2.0e+00)
        self.assertTrue(np.allclose(self.nelder.simplex, self.simplex_to_check5))
        self.assertTrue(np.allclose(self.nelder.f_values, self.f_values_to_check5))
        self.assertTrue(np.allclose(self.nelder.c, self.centroid_to_check3))
        self.assertTrue(np.allclose(self.nelder.x_c, self.x_c_inside_to_check))

    def testContractionOutside(self):
        for i in range(0, 4):
            self.nelder.run(sum(self.f_variables))

        self.nelder.run(2.0e+00)
        self.nelder.run(1.0e+00)
        self.assertTrue(np.allclose(self.nelder.simplex, self.simplex_to_check6))
        self.assertTrue(np.allclose(self.nelder.f_values, self.f_values_to_check6))
        self.assertTrue(np.allclose(self.nelder.c, self.centroid_to_check4))
        self.assertTrue(np.allclose(self.nelder.x_c, self.x_c_outside_to_check))

class TestShrinkSubroutine(unittest.TestCase):
    def setUp(self):
        self.f_variables = [1.0, 2.0, 0.0]
        self.nelder = TestNelderMead(self.f_variables)

        self.simplex_to_check7 = np.array([[5.0e-01, 1.0e+00, 1.25e-04],
                                          [0.0e+00, 2.5e-02, 1.25e-04],
                                          [2.5e-02, 0.0e+00, 1.25e-04],
                                          [0.0e+00, 0.0e+00, 2.5e-04]])

        self.simplex_to_check8 = np.array([[0.0e+00, 0.0e+00, 2.5e-04],
                                          [0.0e+00, 2.5e-02, 1.25e-04],
                                          [2.5e-02, 0.0e+00, 1.25e-04],
                                          [5.0e-01, 1.0e+00, 1.25e-04]])

        self.f_values_to_check7 = np.array([2.5e-04, 2.5125e-02, 2.5125e-02, 1.500125e+00])

    def testShrink(self):
        self.nelder = TestNelderMead(self.f_variables, use_shrink=True)

        for i in range(0, 4):
            self.nelder.run(sum(self.f_variables))

        self.nelder.run(4.0e+00)
        self.nelder.run(5.0e+00)
        self.assertTrue(np.allclose(self.nelder.simplex, self.simplex_to_check7))

        for i in range(0,3):
            self.nelder.run(sum(self.f_variables))

        self.assertTrue(np.allclose(self.nelder.simplex, self.simplex_to_check8))
        self.assertTrue(np.allclose(self.nelder.f_values, self.f_values_to_check7))

if __name__ == '__main__':
    unittest.main()
