#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.metrics import accuracy_score, mean_absolute_error


def accuracy(gt, pred):
    gt = list(map(lambda v: str(v), gt))
    pred = list(map(lambda v: str(int(v)), pred))
    return accuracy_score(gt, pred)


def mae(gt, pred):
    return mean_absolute_error(gt, pred)
