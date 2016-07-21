#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix

from basicmodel import BasicModel


u"""
Example:
python models/truthfinder.py population -e -v
"""


class TruthFinder(BasicModel):
    def __init__(self, dataset='population', has_header=False,
                 tol=0.1, max_iter=10,
                 evaluation=False,
                 verbose=False):
        BasicModel.__init__(self, dataset=dataset, has_header=has_header,
                            tol=0.1, max_iter=10,
                            evaluation=evaluation,
                            name='TruthFinder', verbose=verbose)

    def initialize(self):
        if self.verbose:
            self.logger.info('Initialize')

        self.trust = np.ones(self.n_sources) * 0.9
        self.tau = -np.log(1.0 - self.trust)
        self.rho = 0.3  # controls the influence of related facts
        self.gamma = 0.3  # dampening factor
        self.normalize_values()
        self.initialize_A()
        self.initialize_B()

    def initialize_A(self):
        if self.verbose:
            self.logger.info('Iniaitlize A')

        idx_nonzero = np.nonzero(self.V)

        denom = np.zeros(self.n_sources)
        k, v = zip(*Counter(idx_nonzero[0]).items())
        denom[list(k)] = v

        self.A = lil_matrix((self.n_sources, self.n_claims))
        self.A[idx_nonzero] = 1.0 / denom[idx_nonzero[0]]
        self.A = self.A.tocsr()

    def initialize_B(self, is_symmetric=True, base_thr=0):
        if self.verbose:
            self.logger.info('Iniaitlize B')
        self.B = lil_matrix((self.n_claims, self.n_sources))
        self.B[np.nonzero(self.V.T)] = 1 - self.rho * (1 - base_thr)

        # Make implication matrix
        imp_mat = lil_matrix((self.n_claims, self.n_claims))
        for item in range(self.n_items):  # dataItemID (alias)
            claims = np.where(self.table_c_to_i == item)[0]  # claimID (alias)
            for i, j in combinations(claims, 2):
                v_i = self.table_c_to_v[i]
                v_j = self.table_c_to_v[j]
                d = np.exp(-abs(v_i - v_j)) - base_thr
                imp_mat[i, j] = d
                if is_symmetric:
                    imp_mat[j, i] = d

        # Fill B
        for source in range(self.n_sources):
            idx = np.nonzero(self.V[source])
            gain = imp_mat[idx].sum(axis=0).getA()[0]
            for idx in np.nonzero(gain)[0]:
                self.B[idx, source] += self.rho * gain[idx]
        self.B = self.B.tocsr()

    def run(self, base_thr=0):
        self.initialize()

        if self.verbose:
            self.logger.info('Running')

        diff = self.tol * 2
        iter_count = 0
        while (diff > self.tol) and (iter_count < self.max_iter):
            iter_count += 1
            # trust_prev = np.copy(self.trust)
            tau_prev = np.copy(self.tau)
            self.update_claim()
            self.update_source()
            # diff = np.dot(trust_prev, self.trust)
            # diff /= np.linalg.norm(trust_prev)
            # diff /= np.linalg.norm(self.trust)
            diff = np.dot(tau_prev, self.tau)
            diff /= np.linalg.norm(tau_prev)
            diff /= np.linalg.norm(self.tau)
            diff = 1 - diff
            if self.verbose:
                self.logger.info('Diff: {}'.format(diff))

        if self.verbose:
            self.logger.info('Done')

        self.normalize_values_inv()
        self.get_results()

        if self.evaluation:
            self.evaluate('mae')

    def update_claim(self):
        print(self.tau)
        sigma_star = self.B.dot(self.tau)
        # self.sigma = np.dot(self.V.T, self.tau)
        # sigma_star = (1 - self.rho * (1 - base_thr)) * self.sigma
        # # + interaction
        self.conf = 1 / (1 + np.exp(-self.gamma * sigma_star))

    def update_source(self):
        self.trust = self.A.dot(self.conf)
        self.tau = -np.log(1 - self.trust)
        self.tau[self.trust >= 1] = np.log(1e10)

    def get_results(self):
        df = pd.DataFrame(self.conf, columns=['conf'])
        # Value
        df = df.join(pd.DataFrame(self.table_c_to_v, columns=['value']))
        # item idx
        df = df.join(pd.DataFrame(self.table_c_to_i, columns=['item_idx']))
        df_item_alias = pd.DataFrame(list(self.alias_item.items()),
                                     columns=['dataItemID', 'item_idx'])
        df = pd.merge(df, df_item_alias, on='item_idx')
        df.sort_values('conf', ascending=False, inplace=True)
        df.drop_duplicates('dataItemID', keep='first', inplace=True)
        self.result = df[['dataItemID', 'value']]


def main(args):
    model = TruthFinder(has_header=args.flag_has_header,
                        evaluation=args.flag_evaluation,
                        verbose=args.verbose)

    model.run()

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset type')
    parser.add_argument('--has-header', dest='flag_has_header',
                        action='store_true', default=False,
                        help='Has header or not')
    parser.add_argument('-e', '--evaluation', dest='flag_evaluation',
                        action='store_true', default=False)
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
