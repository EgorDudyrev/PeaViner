from typing import Tuple

import numpy as np
from bitarray import frozenbitarray as fbitarray


def meas_tp(extent: fbitarray, y: fbitarray):
    return (extent & y).count()


def meas_tps(extents: Tuple[fbitarray, ...], y: fbitarray):
    return np.array([(ext & y).count() for ext in extents])


def meas_tn(extent: fbitarray, y: fbitarray):
    return ((~extent) & (~y)).count()


def meas_tns(extents: Tuple[fbitarray, ...], y: fbitarray):
    return np.array([((~ext) & (~y)).count() for ext in extents])


def meas_fp(extent: fbitarray, y: fbitarray):
    return (extent & (~y)).count()


def meas_fps(extents: Tuple[fbitarray, ...], y: fbitarray):
    return np.array([(ext & (~y)).count() for ext in extents])


def meas_fn(extent: fbitarray, y: fbitarray):
    return ((~extent) & y).count()


def meas_fns(extents: Tuple[fbitarray, ...], y: fbitarray):
    return np.array([((~ext) & y).count() for ext in extents])


def meas_tps_perc(extents: Tuple[fbitarray, ...], y: fbitarray):
    return meas_tps(extents, y) / len(y)


def meas_tns_perc(extents: Tuple[fbitarray, ...], y: fbitarray):
    return meas_tns(extents, y) / len(y)


def meas_fps_perc(extents: Tuple[fbitarray, ...], y: fbitarray):
    return meas_fps(extents, y) / len(y)


def meas_fns_perc(extents: Tuple[fbitarray, ...], y: fbitarray):
    return meas_fns(extents, y) / len(y)


def meas_f1s(extents: Tuple[fbitarray, ...], y: fbitarray):
    return 2*np.array([(ext & y).count()/(y.count() + ext.count()) for ext in extents])


def meas_jaccs(extents: Tuple[fbitarray, ...], y: fbitarray):
    return np.array([(ext & y).count()/(ext | y).count() for ext in extents])


SCORES_NAMES = {
    'tp': meas_tps, 'tn': meas_tns, 'fp': meas_fps, 'fn': meas_fns,
    'tp_perc': meas_tps_perc, 'tn_perc': meas_tns_perc, 'fp_perc': meas_fps_perc, 'fn_perc': meas_fns_perc,
    'F1': meas_f1s, 'Jaccard': meas_jaccs
}
