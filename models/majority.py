#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

from basicmodel import BasicModel


u"""
Example:
python models/majority.py population -e -v
"""


class Majority(BasicModel):
    def __init__(self, dataset='population', has_header=False,
                 evaluation=False,
                 verbose=False):
        BasicModel.__init__(self, dataset=dataset, has_header=has_header,
                            evaluation=evaluation,
                            name='Majority', verbose=verbose)

    def run(self):
        if self.verbose:
            self.logger.info('Running')

        g = self.reader.data.groupby('dataItemID')['value']
        f_agg = lambda l: sorted(list(l.value_counts().iteritems()),
                                 key=lambda t: (t[1], t[0]))[-1][0]
        g = g.agg(f_agg)
        self.result = pd.DataFrame(g)
        self.result.reset_index(inplace=True)

        if self.verbose:
            self.logger.info('Done')

        if self.evaluation:
            self.evaluate('mae')


def main(args):
    model = Majority(has_header=args.flag_has_header,
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
