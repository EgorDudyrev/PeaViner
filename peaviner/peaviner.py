from dataclasses import dataclass
from typing import Tuple, FrozenSet

import numpy as np
from tqdm import tqdm
from bitarray import frozenbitarray as fbitarray
from . import scores

IntPremType = Tuple[int, str, float]  # Feature_id, '>=' or '<', numeric threshold


@dataclass(repr=False)
class PeaViner:
    gamma: float = None
    atom_premises: Tuple[FrozenSet[IntPremType], ...] = None  # For each atomic extent, list its premises
    atom_extents: Tuple[fbitarray, ...] = None  # Tuple of extents that can be obtained by a single premise
    y: fbitarray = None  # Target labels

    def load_dataset(self, X: np.ndarray, y: np.ndarray, use_tqdm=False):
        self.y = fbitarray(y.tolist())
        self.gamma = self.y.count() / len(self.y)

        self.atom_extents, self.atom_premises = self._generate_atomic_extents(X, use_tqdm)

    def form_extent_stats(self, extents: Tuple[fbitarray, ...] = None, scores_='all', dataframe=True):
        extents = extents if extents is not None else self.atom_extents

        if scores_ == 'all':
            scores_ = tuple(scores.SCORES_NAMES)

        stats = np.array([scores.SCORES_NAMES[score](extents, self.y) for score in scores_]).T

        if dataframe:
            import pandas as pd
            stats = pd.DataFrame(stats, columns=scores_)
            stats.index.name = 'extent_idx'
        return stats

    def generate_diextents(self, extents: Tuple[fbitarray, ...] = None, operations: tuple = ('^', 'v'), use_tqdm=False):
        extents = extents if extents is not None else self.atom_extents

        diextents_stat = {}

        if use_tqdm:
            n_ops = len(operations)
            pbar = tqdm(total=len(extents)*(len(extents)-1)//2 * n_ops)

        for i, a in enumerate(extents):
            for j, b in enumerate(extents[i+1:]):
                for op in operations:
                    if op == '^':
                        c = a & b
                    elif op == 'v':
                        c = a | b
                    else:
                        raise NotImplementedError(f'Operation {op} is not implemented')

                    if c == a or c == b:
                        continue

                    op_stat = (i, j+i+1, op)

                    if c in diextents_stat:
                        diextents_stat[c].append(op_stat)
                    else:
                        diextents_stat[c] = [op_stat]

                if use_tqdm:
                    pbar.update(n_ops)

        if use_tqdm:
            pbar.close()

        diextents = []
        oper_stats = []
        for ext, ops in diextents_stat.items():
            diextents.append(ext)
            oper_stats.append(tuple(ops))
        diextents, oper_stats = tuple(diextents), tuple(diextents)

        return diextents, oper_stats

    def compute_pairwise_tpfp_stats(self, extents: Tuple[fbitarray, ...] = None, use_tqdm=False):
        from scipy import sparse

        extents = extents if extents is not None else self.atom_extents

        row, col = [], []
        conj_data_tp, conj_data_fp, disj_data_tp, disj_data_fp = [], [], [], []

        if use_tqdm:
            pbar = tqdm(total=len(extents)*(len(extents)-1)//2)

        for i, a in enumerate(extents):
            for j, b in enumerate(extents[i+1:]):
                c = a & b
                if c == a or c == b:
                    if use_tqdm:
                        pbar.update(1)
                    continue
                row.append(i)
                col.append(j)

                conj_data_tp.append(scores.meas_tp(c, self.y))
                conj_data_fp.append(scores.meas_fp(c, self.y))

                c = a | b
                disj_data_tp.append(scores.meas_tp(c, self.y))
                disj_data_fp.append(scores.meas_fp(c, self.y))

                if use_tqdm:
                    pbar.update(1)

        if use_tqdm:
            pbar.close()

        row, col = np.array(row), np.array(col)

        conj_tp_mx, conj_fp_mx, disj_tp_mx, disj_fp_mx = [
            sparse.csr_matrix((np.array(data), (row, col)))/len(self.y)
            for data in [conj_data_tp, conj_data_fp, disj_data_tp, disj_data_fp]
        ]

        return conj_tp_mx, conj_fp_mx, disj_tp_mx, disj_fp_mx

    @staticmethod
    def _generate_atomic_extents(X: np.ndarray, use_tqdm=False):
        extents_i_map, extents_prem_list = {}, []  # extent -> extent_id

        ext_i = 0
        for col_i, col in tqdm(enumerate(X.T), total=X.shape[1], disable=not use_tqdm):
            vals = np.sort(np.unique(col))
            vals = vals[~np.isnan(vals)]

            ext_chain_geq = []
            for v in vals:
                ext = col >= v
                ext = fbitarray(ext.tolist())
                ext_neg = ~ext

                ext_chain_geq.append(ext)

                p_geq, p_ngeq = (col_i, '>=', v), (col_i, 'not >=', v)

                if ext in extents_i_map:
                    ext_i_ = extents_i_map[ext]
                    extents_prem_list[ext_i_].add(p_geq)

                    ext_i_neg_ = extents_i_map[ext_neg]
                    extents_prem_list[ext_i_neg_].add(p_ngeq)
                    continue

                extents_i_map[ext] = ext_i
                extents_prem_list.append({p_geq})
                ext_i += 1

                extents_i_map[ext_neg] = ext_i
                extents_prem_list.append({p_ngeq})
                ext_i += 1

            assert ext_i == len(extents_i_map)

        assert ext_i == len(extents_i_map)

        extents = sorted(extents_i_map, key=lambda ext: extents_i_map[ext])
        extents_prem_list = tuple([frozenset(prems) for prems in extents_prem_list])

        return extents, extents_prem_list

    def __repr__(self):
        if any([v is None for v in [self.gamma, self.y, self.atom_premises, self.atom_extents]]):
            s = 'PeaViner (no dataset loaded)'
            return s

        s = "PeaViner: " \
            f"gamma: {self.gamma:.3f}" + \
            f"; #objects: {len(self.y):,}" + \
            f"; #atom_extents: {len(self.atom_extents):,}" + \
            f"; #atom_premises: {sum([len(ps) for ps in self.atom_premises]):,}"
        return s
