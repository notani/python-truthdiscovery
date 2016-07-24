#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from reader import ReaderPopulation
import utils
import evaluation

filepaths = {'population': './data/population'}
readers = {'population': ReaderPopulation}


class BasicModel:
    def __init__(self, dataset='population',
                 has_header=False,
                 tol=0.1, max_iter=10,
                 evaluation=False,
                 name='Model', verbose=False):
        self.verbose = verbose
        self.logger = utils.init_logger(name=name)

        self.tol = tol
        self.max_iter = max_iter

        self.n_sources = 0
        self.n_items = 0
        self.n_claims = 0
        self.alias_source = {}
        self.alias_claim = {}
        self.alias_item = {}
        self.table_c_to_v = []
        self.table_c_to_i = []
        self.V = None

        self.dataset = dataset
        self.has_header = has_header
        self.reader = None
        self.read()
        self.evaluation = evaluation
        if self.evaluation:
            self.read_groundtruth()

    def read(self):
        if self.dataset is None:
            self.logger.warning('Dataset is not specified')
            return

        if self.reader is None:
            try:
                self.reader = readers[self.dataset](verbose=self.verbose)
            except KeyError:
                self.logger.error('Invalid dataset: ' + self.dataset)
                raise
        self.reader.read()
        self.reader.report_statistical_info()

        self.n_sources = len(self.reader.data['sourceID'].unique())
        self.n_items = len(self.reader.data['dataItemID'].unique())
        self.n_claims = len(self.reader.data['claimID'].unique())
        self.get_matrix()

    def read_groundtruth(self):
        if self.reader is None:
            try:
                self.reader = readers[self.dataset](verbose=self.verbose)
            except KeyError:
                self.logger.error('Invalid dataset: ' + self.dataset)
                raise
        self.reader.read_groundtruth()

    def get_matrix(self):
        if self.verbose:
            self.logger.info('Building a sparse matrix')

        self.get_aliases()

        df = pd.DataFrame(index=self.reader.data.index)
        df['i'] = self.reader.data['sourceID'].apply(lambda v: self.alias_source[v])
        df['j'] = self.reader.data['claimID'].apply(lambda v: self.alias_claim[v])
        df['k'] = self.reader.data['dataItemID'].apply(lambda v: self.alias_item[v])
        df['val'] = self.reader.data['value']
        df['vote'] = 1

        self.get_mapping_tables(df)

        sparse_s = df.set_index(['i', 'j'])['vote'].to_sparse()
        S, rows, cols = sparse_s.to_coo(row_levels=['i'], column_levels=['j'],
                                        sort_labels=True)
        self.V = S.todense().getA().astype(np.float64)

    def get_aliases(self):
        # Alias
        sources = sorted(self.reader.data['sourceID'].unique())
        self.alias_source = {sources[i]: i for i in range(self.n_sources)}
        claims = sorted(self.reader.data['claimID'].unique())
        self.alias_claim = {claims[i]: i for i in range(self.n_claims)}
        items = sorted(self.reader.data['dataItemID'].unique())
        self.alias_item = {items[i]: i for i in range(self.n_items)}

    def get_mapping_tables(self, df):
        # Mapping table: from claim to value
        table_ = df[['j', 'val']].drop_duplicates().sort_values('j')
        self.table_c_to_v = table_['val'].values.astype(np.float64)

        # Mapping table: from claim to item
        table_ = df[['j', 'k']].drop_duplicates().sort_values('j')
        self.table_c_to_i = table_['k'].values.astype(int)

    def get_mean_std(self):
        u"""Calculate mean and std for each data item
        """
        self.means = np.empty(self.n_items)
        self.stds = np.empty(self.n_items)
        for i in range(self.n_items):
            buff = []
            for c in np.where(self.table_c_to_i == i)[0]:
                buff += [self.table_c_to_v[c]] * len(np.nonzero(self.V[:, c])[0])
            self.means[i] = np.mean(buff)
            self.stds[i] = np.std(buff)

    def normalize_values(self):
        if self.verbose:
            self.logger.info('Normalize')
        self.get_mean_std()
        self.stds[self.stds == 0] = 1.0  # modify to normalize

        self.table_c_to_v -= self.means[self.table_c_to_i]
        self.table_c_to_v /= self.stds[self.table_c_to_i]

    def normalize_values_inv(self):
        self.table_c_to_v *= self.stds[self.table_c_to_i]
        self.table_c_to_v += self.means[self.table_c_to_i]

    def evaluate(self, how):
        if not self.evaluation:
            self.logger.error('Require a evaluation flag')
            raise
        self.result.set_index('dataItemID', inplace=True)
        self.reader.gt.set_index('dataItemID', inplace=True)
        shared_idx = set(self.reader.gt.index)
        shared_idx = shared_idx.intersection(self.result.index)
        shared_idx = sorted(list(shared_idx))

        of = open('hoge.txt', 'w')
        for val in sorted(self.result.ix[shared_idx]['value'].values.astype(int)):
            of.write('{}\n'.format(val))
        of.close()
        if self.verbose:
            self.logger.info('Test samples: {}'.format(len(shared_idx)))
        gt = self.reader.gt.ix[shared_idx]['value'].values
        pred = self.result.ix[shared_idx]['value'].values
        if how == 'mae':
            val = evaluation.mae(gt, pred)
        if how == 'accuracy':
            val = evaluation.accuracy(gt, pred)
        print(val)
        self.result.reset_index(inplace=True)
        self.reader.gt.reset_index(inplace=True)
