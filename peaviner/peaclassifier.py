from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .peaviner import IntPremType


@dataclass(repr=True)
class PeaClassifier:
    premises: Tuple[IntPremType, ...]
    type: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.type in {1, 2, 3, 4}, f'Unsupported type value: {self.type}'
        assert all([op in {'>=', '<'} for (_, op, _) in self.premises]), 'Only ">=" and "<" operations are supported'

        pe, qe, re = [{'>=': np.greater_equal, '<': np.less}[op](X[:, f_id], th) for (f_id, op, th) in self.premises]

        if self.type == 1:
            preds = pe & qe & re
        elif self.type == 2:
            preds = pe & qe | re
        elif self.type == 3:
            preds = (pe | qe) & re
        else:  # self.type == 4:
            preds = pe | qe | re

        return preds
