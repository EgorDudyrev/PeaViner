from typing import Tuple

import numpy as np
from bitarray import frozenbitarray as fbitarray


def meas_tp(extents: Tuple[fbitarray], y: fbitarray):
    return np.array([(ext & y).count() for ext in extents])


def meas_tn(extents: Tuple[fbitarray], y: fbitarray):
    return np.array([((~ext) & (~y)).count() for ext in extents])


def meas_fp(extents: Tuple[fbitarray], y: fbitarray):
    return np.array([(ext & (~y)).count() for ext in extents])


def meas_fn(extents: Tuple[fbitarray], y: fbitarray):
    return np.array([((~ext) & y).count() for ext in extents])


def meas_tp_perc(extents: Tuple[fbitarray], y: fbitarray):
    return meas_tp(extents, y)/len(y)


def meas_tn_perc(extents: Tuple[fbitarray], y: fbitarray):
    return meas_tn(extents, y)/len(y)


def meas_fp_perc(extents: Tuple[fbitarray], y: fbitarray):
    return meas_fp(extents, y)/len(y)


def meas_fn_perc(extents: Tuple[fbitarray], y: fbitarray):
    return meas_fn(extents, y)/len(y)


def meas_f1(extents: Tuple[fbitarray], y: fbitarray):
    return 2*np.array([(ext & y).count()/(y.count() + ext.count()) for ext in extents])


def meas_jacc(extents: Tuple[fbitarray], y: fbitarray):
    return np.array([(ext & y).count()/(ext | y).count() for ext in extents])


SCORES_NAMES = {
    'tp': meas_tp, 'tn': meas_tn, 'fp': meas_fp, 'fn': meas_fn,
    'tp_perc': meas_tp_perc, 'tn_perc': meas_tn_perc, 'fp_perc': meas_fp_perc, 'fn_perc': meas_fn_perc,
    'F1': meas_f1, 'Jacc': meas_jacc
}
