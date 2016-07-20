#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import argparse
import codecs
import logging
import numpy as np
import scipy as sp
import pandas as pd

verbose = False
enc = 'utf_8'


def init_logger():
    logger = logging.getLogger('Reader')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


class Reader:
    """Data reader
    """

    def __init__(self, path_input='', path_groundtruth='',
                 verbose=False):
        self.path_input = path_input
        self.path_groundtruth = path_groundtruth
        self.data = None
        self.gt = None
        self.verbose = verbose
        self.logger = init_logger()

    def read(self, has_header=False, columns='infer',
             names=['claimID', 'sourceID', 'dataItemID', 'value']):
        if len(self.path_input) == 0:
            self.logger.error('Invalid file path: ' + self.path_input)

        header = None
        if has_header:
            header = 'infer'

        self.data = pd.read_table(self.path_input, header=header, names=names)
        if self.verbose:
            self.logger.info('Read {} rows from {}'.format(
                len(self.data), self.path_input))

        self.modify_data()
        self.filter_data()

    def modify_data(self):
        pass

    def filter_data(self):
        pass

    def read_groundtruth(self, has_header=False, columns='infer', delimiter='\t',
                         names=['claimID', 'sourceID', 'dataItemID', 'value']):
        if len(self.path_groundtruth) == 0:
            self.logger.error('Invalid file path: ' + self.path_groundtruth)

        header = None
        if has_header:
            header = 'infer'

        self.gt = pd.read_table(self.path_groundtruth, delimiter=delimiter,
                                header=header, names=names)
        if self.verbose:
            self.logger.info('Read {} rows from {}'.format(
                len(self.gt), self.path_groundtruth))

        self.modify_groundtruth()
        self.filter_groundtruth()

    def modify_groundtruth(self):
        pass

    def filter_groundtruth(self):
        pass

    def report_statistical_info(self):
        self.logger.info('Claims\t{}'.format(len(self.data)))
        self.logger.info('Sources\t{}'.format(len(self.data['sourceID'].unique())))
        self.logger.info('Objects\t{}'.format(len(self.data['dataItemID'].unique())))


class ReaderPopulation(Reader):
    def __init__(self, verbose=False):
        Reader.__init__(self,
                        './data/population/popTuples.txt',
                        './data/population/popAnswersOut.txt',
                        verbose=verbose)

    def read(self):
        names = ['city', 'dataset name',
                 'source infobox type', 'edit location',
                 'sourceID', 'revision ID',
                 'time span',
                 'value']
        Reader.read(self, has_header=False, names=names)

    def modify_data(self):
        Reader.modify_data(self)
        self.data['year'] = self.data['time span'].apply(lambda s: s[5:9])
        self.data['city'] = self.data['city'].apply(lambda s: s.lower())
        self.data['dataItemID'] = self.data['city'] + ':' + self.data['year']
        self.data['sourceID'] = self.data['sourceID'].apply(lambda s: s.lower())
        self.data['claimID'] = self.data.index

    def filter_data(self):
        u"""Filtering outliers

        This procedure follows [Wan+ 16], resulting:
        - Claims   4008
        - Sources  2344
        - Objects  1124
        """
        Reader.filter_data(self)

        # Remove some obviously-wrong claims
        idx = self.data[self.data['value'] > 1e8].index
        self.data.drop(idx, inplace=True)

        # Keep only the latest claim for the same source and the same entity
        self.data.drop_duplicates(['dataItemID', 'sourceID'], keep='last',
                                  inplace=True)

        # Remove entities whose claims are all the same
        agg = self.data.drop_duplicates(['dataItemID', 'value']).groupby('dataItemID').size()
        srcIDs = set(agg[agg == 1].index)
        idx = self.data[self.data['dataItemID'].isin(srcIDs)].index
        self.data.drop(idx, inplace=True)

    def read_groundtruth(self):
        names = ['city1', 'city2', 'year', 'value']
        Reader.read_groundtruth(self, delimiter=',',
                                has_header=False, names=names)

    def modify_groundtruth(self):
        Reader.modify_groundtruth(self)

        self.gt['dataItemID'] = self.gt['city1'] + ',' + self.gt['city2'] + ':' \
                                + self.gt['year'].apply(lambda v: str(v))
        self.gt.drop(['city1', 'city2', 'year'], axis=1, inplace=True)
