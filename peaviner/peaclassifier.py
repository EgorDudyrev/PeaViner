from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from .peaviner import IntPremType
from .peaviner import PeaViner


@dataclass(repr=True)
class PeaClassifier:
    premises: Tuple[IntPremType, ...] = None
    type: int = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classifiers: int = 1, use_tqdm: bool = False)\
            -> Optional[Tuple[PeaClassifier, ...]]:
        pv = PeaViner()
        pv.load_dataset(X, y)
        conj_tp_stat, conj_fp_stat, disj_tp_stat, disj_fp_stat = pv.compute_pairwise_tpfp_stats(use_tqdm=use_tqdm)

        aext_tps, aext_fps, aext_jaccs = pv.form_extent_stats(
            scores_=['tp_perc', 'fp_perc', 'Jaccard'], dataframe=False).T
        aext_jaccs = np.sort(aext_jaccs)[-n_classifiers:]

        diexts = pv.generate_diextents()[0]  # TODO: Optimize. using conj_tp_stat etc
        diexts_jaccs = pv.form_extent_stats(extents=diexts, scores_=['Jaccard'], dataframe=False).flatten()
        diexts_jaccs = np.sort(diexts_jaccs)[-n_classifiers:]
        thold = np.sort(list(aext_jaccs)+list(diexts_jaccs))[-n_classifiers:][0]

        best_premises_ids_list = pv.find_best_premises_size3(
            thold, n_classifiers,
            aext_tps, aext_fps,
            conj_tp_stat, conj_fp_stat, disj_tp_stat, disj_fp_stat,
            update_thold=True, return_n_iters=False,
            types=(1, 2, 3, 4),
            use_tqdm=use_tqdm
        )

        best_premises_list = tuple([
            (tuple([list(pv.atom_premises[p_i])[0] for p_i in prem_ids]), t)
            for (prem_ids, t, _) in best_premises_ids_list
        ])

        if n_classifiers == 1:
            self.premises = best_premises_list[0][0]
            self.type = best_premises_list[0][1]
            return

        clfs = tuple([self.__class__(premises=prems, type=t) for prems, t in best_premises_list])
        return clfs

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

    def explain(self, feature_names: Tuple[str, ...] = None) -> str:
        assert self.type in {1, 2, 3, 4}

        prem_fis = [f_i for (f_i, _, _) in self.premises]
        feat_names = [feature_names[f_i] if feature_names else f"f_{f_i}" for f_i in prem_fis]

        p, q, r = [f"{fname} {op} {thold}" for fname, (f_i, op, thold) in zip(feat_names, self.premises)]

        if self.type == 1:
            expl = f"{p} AND {q} AND {r}"
        elif self.type == 2:
            expl = f"{p} AND {q} OR {r}"
        elif self.type == 3:
            expl = f"( {p} OR {q} ) AND {r}"
        else:  # self.type == 4:
            expl = f"{p} OR {q} OR {r}"
        return expl
