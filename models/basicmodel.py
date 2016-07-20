#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
from reader import ReaderPopulation
import utils
import evaluation

filepaths = {'population': './data/population'}
readers = {'population': ReaderPopulation}


class BasicModel:
    def __init__(self, dataset='population',
                 has_header=False,
                 evaluation=False,
                 name='Model', verbose=False):
        self.verbose = verbose
        self.logger = utils.init_logger(name=name)

        self.dataset = dataset
        self.has_header = has_header
        self.reader = None
        self.read()
        self.evaluation = evaluation
        if self.evaluation:
            self.read_groundtruth()

    def read(self):
        if self.reader is None:
            try:
                self.reader = readers[self.dataset](verbose=self.verbose)
            except KeyError:
                self.logger.error('Invalid dataset: ' + self.dataset)
                raise
        self.reader.read()
        self.reader.report_statistical_info()

    def read_groundtruth(self):
        if self.reader is None:
            try:
                self.reader = readers[self.dataset](verbose=self.verbose)
            except KeyError:
                self.logger.error('Invalid dataset: ' + self.dataset)
                raise
        self.reader.read_groundtruth()

    def evaluate(self, how):
        if not self.evaluation:
            self.logger.error('Require a evaluation flag')
            raise
        self.result.set_index('dataItemID', inplace=True)
        self.reader.gt.set_index('dataItemID', inplace=True)
        shared_idx = set(self.reader.gt.index)
        shared_idx = shared_idx.intersection(self.result.index)
        shared_idx = sorted(list(shared_idx))

        if self.verbose:
            self.logger.info('Test samples: {}'.format(len(shared_idx)))
        if how == 'mae':
            val = evaluation.mae(self.reader.gt.ix[shared_idx].value,
                                 self.result.ix[shared_idx].value)
        print(val)
        self.result.reset_index(inplace=True)
        self.reader.gt.reset_index(inplace=True)
