#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.metrics import mean_absolute_error


def mae(gt, pred):
    return mean_absolute_error(gt, pred)
