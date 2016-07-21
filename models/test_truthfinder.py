#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import unittest
from truthfinder import TruthFinder

class TestCalc(unittest.TestCase):
    def setUp(self):
        self.model = TruthFinder(dataset=None, has_header=False,
                                 evaluation=False, verbose=True)

        self.model.n_sources = 5
        self.model.n_items = 3
        self.model.n_claims = 10

        # Test Cases
        test1_data = {'dataItemID', }
        test1_items = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        test1_values = [10, 9, 3, 4, 5, 10, 3, 1, 100, -3]
        test1_V = []
        test1_V.append([1, 0, 0, 0, 0, 1, 0, 1, 0, 0])  # source#0
        test1_V.append([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])  # source#1
        test1_V.append([1, 0, 0, 0, 0, 1, 0, 1, 0, 0])  # source#2
        test1_V.append([0, 1, 0, 0, 0, 1, 0, 0, 1, 0])  # source#3
        test1_V.append([0, 1, 0, 0, 1, 0, 0, 0, 0, 1])  # source#4

        # Set
        self.model.table_c_to_i = np.array(test1_items, dtype=int)
        self.model.table_c_to_v = np.array(test1_values, dtype=np.float64)
        self.model.V = np.array(test1_V, dtype=np.float64)

    def test_normalization(self):
        ans1_means = [np.mean([10, 10, 10, 9, 9]),
                      np.mean([10, 10, 10, 5]),
                      np.mean([1, 3, 1, 100, -3])]
        ans1_stds = [np.std([10, 10, 10, 9, 9]),
                     np.std([10, 10, 10, 5]),
                     np.std([1, 3, 1, 100, -3])]
        self.model.normalize_values()
        for i in range(self.model.n_items):
            self.assertEqual(ans1_means[i], self.model.means[i])

        for i in range(self.model.n_items):
            self.assertEqual(ans1_stds[i], self.model.stds[i])

    def test_initializeA(self):
        ans1_A = []
        ans1_A.append([1.0/3.0, 0, 0, 0, 0, 1.0/3.0, 0, 1.0/3.0, 0, 0])  # source#0
        ans1_A.append([1.0/2.0, 0, 0, 0, 0, 0, 1.0/2.0, 0, 0, 0])        # source#1
        ans1_A.append([1.0/3.0, 0, 0, 0, 0, 1.0/3.0, 0, 1.0/3.0, 0, 0])  # source#2
        ans1_A.append([0, 1.0/3.0, 0, 0, 0, 1.0/3.0, 0, 0, 1.0/3.0, 0])  # source#3
        ans1_A.append([0, 1.0/3.0, 0, 0, 1.0/3.0, 0, 0, 0, 0, 1.0/3.0])  # source#4
        ans1_A = np.array(ans1_A)
        self.model.normalize_values()
        self.model.rho = 0.5
        self.model.gamma = 0.3
        self.model.initialize_A()
        for i, j in zip(*np.nonzero(ans1_A)):
            self.assertEqual(ans1_A[i, j], self.model.A[i, j])

    def test_initializeB(self):
        rho = 0.3
        # test1_values = [10, 9, 3, 4, 5, 10, 3, 1, 100, -3]
        ans1_imp_mat = {}
        ans1_imp_mat[(0, 1)] = np.exp(-abs(10 - 9))  # = 0.36787944117144233
        ans1_imp_mat[(2, 3)] = np.exp(-abs(3 - 4))  # = 0.36787944117144233
        ans1_imp_mat[(2, 4)] = np.exp(-abs(3 - 5))  # = 0.1353352832366127
        ans1_imp_mat[(2, 5)] = np.exp(-abs(3 - 10))  # = 0.00091188196555451624
        ans1_imp_mat[(3, 4)] = np.exp(-abs(4 - 5))  # = 0.36787944117144233
        ans1_imp_mat[(3, 5)] = np.exp(-abs(4 - 10))  # = 0.0024787521766663585
        ans1_imp_mat[(4, 5)] = np.exp(-abs(5 - 10))  # = 0.006737946999085467
        ans1_imp_mat[(6, 7)] = np.exp(-abs(3 - 1))  # = 0.1353352832366127
        ans1_imp_mat[(6, 8)] = np.exp(-abs(3 - 100))  # = 7.4719723373429907e-43
        ans1_imp_mat[(6, 9)] = np.exp(-abs(3 + 3))  # = 0.0024787521766663585
        ans1_imp_mat[(7, 8)] = np.exp(-abs(1 - 100))  # = 1.0112214926104486e-43
        ans1_imp_mat[(7, 9)] = np.exp(-abs(1 + 3))  # = 0.018315638888734179
        ans1_imp_mat[(8, 9)] = np.exp(-abs(100 + 3))  # = 1.8521167695179754e-45
        for key in ans1_imp_mat.keys():
            ans1_imp_mat[key] *= rho

        # test1_V.append([1, 0, 0, 0, 0, 1, 0, 1, 0, 0])  # source#0
        # test1_V.append([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])  # source#1
        # test1_V.append([1, 0, 0, 0, 0, 1, 0, 1, 0, 0])  # source#2
        # test1_V.append([0, 1, 0, 0, 0, 1, 0, 0, 1, 0])  # source#3
        # test1_V.append([0, 1, 0, 0, 1, 0, 0, 0, 0, 1])  # source#4
        ans1_B = []
        # source#0
        ans1_B.append([1 - rho, ans1_imp_mat[(0, 1)],
                       ans1_imp_mat[(2, 5)], ans1_imp_mat[(3, 5)], ans1_imp_mat[(4, 5)], 1 - rho,
                       ans1_imp_mat[(6, 7)], 1 - rho, ans1_imp_mat[(7, 8)], ans1_imp_mat[(7, 9)]])
        # source#1
        ans1_B.append([1 - rho, ans1_imp_mat[(0, 1)],
                       0, 0, 0, 0,
                       1 - rho, ans1_imp_mat[(6, 7)], ans1_imp_mat[(6, 8)], ans1_imp_mat[(6, 9)]])
        # source#2
        ans1_B.append([1 - rho, ans1_imp_mat[(0, 1)],
                       ans1_imp_mat[(2, 5)], ans1_imp_mat[(3, 5)], ans1_imp_mat[(4, 5)], 1 - rho,
                       ans1_imp_mat[(6, 7)], 1 - rho, ans1_imp_mat[(7, 8)], ans1_imp_mat[(7, 9)]])
        # source#3
        ans1_B.append([ans1_imp_mat[(0, 1)], 1 - rho,
                       ans1_imp_mat[(2, 5)], ans1_imp_mat[(3, 5)], ans1_imp_mat[(4, 5)], 1 - rho,
                       ans1_imp_mat[(6, 8)], ans1_imp_mat[(7, 8)], 1 - rho, ans1_imp_mat[(8, 9)]])
        # source#4
        ans1_B.append([ans1_imp_mat[(0, 1)], 1 - rho,
                       ans1_imp_mat[(2, 4)], ans1_imp_mat[(3, 4)], 1 - rho, ans1_imp_mat[(4, 5)],
                       ans1_imp_mat[(6, 9)], ans1_imp_mat[(7, 9)], ans1_imp_mat[(8, 9)], 1 - rho])
        ans1_B = np.array(ans1_B).T

        # self.model.normalize_values()
        self.model.rho = rho
        self.model.gamma = 0.3
        self.model.initialize_B()
        for i, j in zip(*np.nonzero(ans1_B)):
            self.assertEqual(ans1_B[i, j], self.model.B[i, j])

if __name__ == '__main__':
    unittest.main()
