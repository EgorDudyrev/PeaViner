from dataclasses import dataclass
from typing import Tuple, FrozenSet

import numpy as np
from tqdm import tqdm
from bitarray import frozenbitarray as fbitarray

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

        self.atom_extents, self.atom_premises = self._generate_extents(X, use_tqdm)

    def _generate_extents(self, X: np.ndarray, use_tqdm=False):
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
