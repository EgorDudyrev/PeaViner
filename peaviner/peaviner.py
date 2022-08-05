from dataclasses import dataclass
from typing import Tuple, FrozenSet, Iterator

import numpy as np
from scipy import sparse
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
    score_func: str = 'Jaccard'  # Scoring function to maximize

    def load_dataset(self, X: np.ndarray, y: np.ndarray, use_tqdm=False):
        self.y = fbitarray(y.tolist())
        self.gamma = self.y.count() / len(self.y)

        self.atom_extents, self.atom_premises = self._generate_atomic_extents(X, use_tqdm)
        extent_order = (-scores.SCORES_NAMES[self.score_func](self.atom_extents, self.y)).argsort()
        self.atom_extents, self.atom_premises = [tuple([lst[i] for i in extent_order])
                                                 for lst in [self.atom_extents, self.atom_premises]]

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
        extents = extents if extents is not None else self.atom_extents
        n_exts = len(extents)

        row, col = [], []
        conj_data_tp, conj_data_fp, disj_data_tp, disj_data_fp = [], [], [], []

        if use_tqdm:
            pbar = tqdm(total=n_exts*(n_exts-1)//2)

        for i, a in enumerate(extents):
            for j, b in enumerate(extents[i+1:]):
                c = a & b
                if c == a or c == b:
                    if use_tqdm:
                        pbar.update(1)
                    continue
                row.append(i)
                col.append(j+i+1)

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
            sparse.csr_matrix((np.array(data), (row, col)), shape=(n_exts, n_exts))/len(self.y)
            for data in [conj_data_tp, conj_data_fp, disj_data_tp, disj_data_fp]
        ]

        return conj_tp_mx, conj_fp_mx, disj_tp_mx, disj_fp_mx

    @staticmethod
    def calc_thold_tpfp(gamma: float, thold: float, score_name: str = 'Jaccard') -> Tuple[float, float]:
        if score_name == 'Jaccard':
            thold_fp = gamma*(1-thold)/thold
            thold_tp = gamma*thold
        else:
            raise NotImplementedError('Only Jaccard score is implemented at the moment')
        return thold_tp, thold_fp

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

    @staticmethod
    def calc_alphas(tps: np.ndarray, thold: float, gamma: float) -> np.ndarray:
        return tps/thold - gamma

    @staticmethod
    def calc_betas(fps: np.ndarray, thold: float, gamma: float) -> np.ndarray:
        return thold/(gamma + fps)

    def iterate_potentials_type3_1(
            self, thold: float,
            tps: np.ndarray, fps: np.ndarray,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            use_tqdm: bool = False
    ) -> Iterator[Tuple[int, int, int]]:
        """Iterating potential premises of type pqr"""
        gamma, omg = self.gamma, 1 - self.gamma

        thold_tp, _ = self.calc_thold_tpfp(gamma, thold, 'Jaccard')

        alphas = self.calc_alphas(tps, thold, gamma)
        conj_alphas = conj_tps.copy()
        conj_alphas.data = self.calc_alphas(conj_alphas.data, thold, gamma)

        potential_ps = (thold_tp <= conj_tps).sum(1).nonzero()[0]

        for p in tqdm(potential_ps, disable=not use_tqdm, desc='Iter pqr'):
            tps_pq = conj_tps[p].toarray()[0]  # list of values for q

            potential_qs_flag = thold_tp <= tps_pq
            if not potential_qs_flag.any():
                continue

            potential_qs = potential_qs_flag.nonzero()[0]

            fp_p, alpha_p = fps[p], alphas[p]  # const
            fps_r, alphas_r = fps, alphas  # list of values for r
            fps_pr, alphas_pr = [mx[p].toarray()[0] for mx in [conj_fps, conj_alphas]]  # list of values for r

            for q in potential_qs:
                potential_rs_flg = potential_qs_flag.copy()

                fp_pq, alpha_pq = conj_fps[p, q], conj_alphas[p, q]  # const
                potential_rs_flg &= (fp_pq + fps_r - alpha_pq <= omg)
                if not potential_rs_flg.any():
                    continue

                potential_rs_flg &= (fp_pq + fps_r - alphas_r <= omg)
                if not potential_rs_flg.any():
                    continue

                fp_q, alpha_q = fps[q], alphas[q]  # const
                potential_rs_flg &= (fps_pr + fp_q - alphas_pr <= omg)
                if not potential_rs_flg.any():
                    continue

                potential_rs_flg &= (fps_pr + fp_q - alpha_q <= omg)
                if not potential_rs_flg.any():
                    continue

                tps_qr = conj_tps[q].toarray()[0]  # list of values for r
                potential_rs_flg &= thold_tp <= tps_qr
                if not potential_rs_flg.any():
                    continue

                fps_qr, alphas_qr = [mx[q].toarray()[0] for mx in [conj_fps, conj_alphas]]  # list of values for r
                potential_rs_flg &= (fps_qr + fp_p - alphas_qr <= omg)
                if not potential_rs_flg.any():
                    continue

                potential_rs_flg &= (fps_qr + fp_p - alpha_p <= omg)
                potential_rs = potential_rs_flg.nonzero()[0]
                for r in potential_rs:
                    yield p, q, r

    def iterate_potentials_type3_2(
            self, thold: float,
            tps: np.ndarray, fps: np.ndarray,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
            use_tqdm: bool = False
    ) -> Iterator[Tuple[int, int, int]]:
        """Iterating potential premises of type pq|r"""
        gamma, omg = self.gamma, 1 - self.gamma

        thold_tp, thold_fp = self.calc_thold_tpfp(gamma, thold, 'Jaccard')

        alphas = self.calc_alphas(tps, thold, gamma)
        disj_alphas = disj_tps.copy()
        disj_alphas.data = self.calc_alphas(disj_alphas.data, thold, gamma)

        betas = self.calc_betas(fps, thold, gamma)
        conj_betas = conj_fps.copy()
        conj_betas.data = self.calc_betas(conj_fps.data, thold, gamma)

        potential_ps = (thold_tp <= conj_tps).sum(1).nonzero()[0]

        for p in tqdm(potential_ps, disable=not use_tqdm, desc='Iter pq|r'):
            fps_pq = conj_fps[p].toarray()[0]   # list of values for q

            potential_qs_flag = fps_pq <= thold_fp
            if not potential_qs_flag.any():
                continue

            potential_qs = potential_qs_flag.nonzero()[0]

            tps_r, fps_r, alphas_r, betas_r = tps, fps, alphas, betas  # list of values for r
            tps_p_r, fps_p_r = [mx[p].toarray()[0] for mx in [disj_tps, disj_fps]]  # list of values for r

            for q in potential_qs:
                potential_rs_flg = (fps_r <= thold_fp)

                tp_pq, beta_pq, alpha_p_q = conj_tps[p, q], conj_betas[p, q], disj_alphas[p, q]

                potential_rs_flg &= (0 <= tp_pq + tps_r - beta_pq)
                if not potential_rs_flg.any():
                    continue

                potential_rs_flg &= (0 <= tp_pq + tps_r - betas_r)
                if not potential_rs_flg.any():
                    continue

                potential_rs_flg &= (thold_tp <= tps_p_r)
                if not potential_rs_flg.any():
                    continue

                tps_q_r = disj_tps[q].toarray()[0]
                potential_rs_flg &= (thold_tp <= tps_q_r)
                if not potential_rs_flg.any():
                    continue

                fps_q_r = disj_fps[q].toarray()[0]
                potential_rs_flg &= (fps_p_r + fps_q_r - alpha_p_q <= omg)
                if not potential_rs_flg.any():
                    continue

                alphas_q_r = disj_alphas[q].toarray()[0]
                potential_rs_flg &= (fps_p_r + fps_q_r - alphas_q_r <= omg)
                if not potential_rs_flg.any():
                    continue

                potential_rs = potential_rs_flg.nonzero()[0]
                for r in potential_rs:
                    yield p, q, r

    def iterate_potentials_type3_3(
            self, thold: float,
            tps: np.ndarray, fps: np.ndarray,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
            use_tqdm: bool = False
    ) -> Iterator[Tuple[int, int, int]]:
        """Iterating potential premises of type (p|q)r"""
        gamma, omg = self.gamma, 1 - self.gamma

        thold_tp, thold_fp = self.calc_thold_tpfp(gamma, thold, 'Jaccard')

        alphas = self.calc_alphas(tps, thold, gamma)
        disj_alphas = disj_tps.copy()
        disj_alphas.data = self.calc_alphas(disj_alphas.data, thold, gamma)

        conj_betas = conj_fps.copy()
        conj_betas.data = self.calc_betas(conj_fps.data, thold, gamma)

        potential_ps_flg = np.array((thold_tp <= disj_tps).sum(1) > 0).flatten()
        conj_tns = conj_fps.copy()
        conj_tns.data = omg - conj_tns.data
        thold_tn = omg - thold_fp
        potential_ps_flg &= np.array((conj_tns >= thold_tn).sum(1) > 0).flatten()
        del conj_tns, thold_tn
        potential_ps = potential_ps_flg.nonzero()[0]

        for p in tqdm(potential_ps, disable=not use_tqdm, desc='Iter (p|q)r'):
            tps_p_q = disj_tps[p].toarray()[0]   # list of values for q

            potential_qs_flag = thold_tp <= tps_p_q
            if not potential_qs_flag.any():
                continue

            potential_qs = potential_qs_flag.nonzero()[0]

            tps_r, fps_r, alphas_r = tps, fps, alphas  # list of values for r
            tps_pr, fps_pr, tps_p_r, fps_p_r = [mx[p].toarray()[0] for mx in [conj_tps, conj_fps, disj_tps, disj_fps]]

            for q in potential_qs:
                potential_rs_flg = (thold_tp <= tps_r)
                if not potential_rs_flg.any():
                    continue

                fp_p_q, alpha_p_q = disj_fps[p, q], disj_alphas[p, q]
                potential_rs_flg &= (fp_p_q + fps_r - alphas_r <= omg)
                if not potential_rs_flg.any():
                    continue

                potential_rs_flg &= (fp_p_q + fps_r - alpha_p_q <= omg)
                if not potential_rs_flg.any():
                    continue

                tps_qr, betas_pr = conj_tps[q].toarray()[0], conj_betas[p].toarray()[0]
                potential_rs_flg &= (0 <= tps_pr + tps_qr - betas_pr)
                if not potential_rs_flg.any():
                    continue

                betas_qr = conj_betas[q].toarray()[0]
                potential_rs_flg &= (0 <= tps_pr + tps_qr - betas_qr)
                if not potential_rs_flg.any():
                    continue

                potential_rs_flg &= (fps_pr <= thold_fp)
                if not potential_rs_flg.any():
                    continue

                fps_qr = conj_fps[q].toarray()[0]
                potential_rs_flg &= (fps_qr <= thold_fp)
                if not potential_rs_flg.any():
                    continue

                potential_rs = potential_rs_flg.nonzero()[0]
                for r in potential_rs:
                    yield p, q, r

    def iterate_potentials_type3_4(
            self, thold: float,
            tps: np.ndarray, fps: np.ndarray,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
            use_tqdm: bool = False
    ) -> Iterator[Tuple[int, int, int]]:
        """Iterating potential premises of type p|q|r"""
        gamma, omg = self.gamma, 1 - self.gamma

        thold_tp, thold_fp = self.calc_thold_tpfp(gamma, thold, 'Jaccard')

        betas = self.calc_betas(fps, thold, gamma)
        disj_betas = disj_fps.copy()
        disj_betas.data = self.calc_betas(disj_fps.data, thold, gamma)

        disj_tns = disj_fps.copy()
        disj_tns.data = omg - disj_tns.data
        thold_tn = omg - thold_fp
        potential_ps_flg = np.array((disj_tns >= thold_tn).sum(1) > 0).flatten()
        del disj_tns, thold_tn
        potential_ps = potential_ps_flg.nonzero()[0]

        for p in tqdm(potential_ps, disable=not use_tqdm, desc='Iter p|q|r'):
            fps_p_q = disj_fps[p].toarray()[0]  # list of values for q

            potential_qs_flag = fps_p_q <= thold_fp
            if not potential_qs_flag.any():
                continue

            potential_qs = potential_qs_flag.nonzero()[0]

            tp_p, beta_p = tps[p], betas[p]
            tps_r, betas_r = tps, betas
            fps_p_r = disj_fps[p].toarray()[0]

            for q in potential_qs:
                potential_rs_flg = (fps_p_r <= thold_fp)
                if not potential_rs_flg.any():
                    continue

                tp_p_q, beta_p_q = disj_tps[p, q], disj_betas[p, q]
                potential_rs_flg &= (0 <= tp_p_q + tps_r - beta_p_q)
                if not potential_rs_flg.any():
                    continue

                potential_rs_flg &= (0 <= tp_p_q + tps_r - betas_r)
                if not potential_rs_flg.any():
                    continue

                fps_q_r = disj_fps[q].toarray()[0]
                potential_rs_flg &= (fps_q_r <= thold_fp)
                if not potential_rs_flg.any():
                    continue

                tp_q, beta_q = tps[q], betas[q]
                tps_p_r = disj_tps[p].toarray()[0]
                potential_rs_flg &= (0 <= tps_p_r + tp_q - beta_q)
                if not potential_rs_flg.any():
                    continue

                betas_p_r = disj_betas[p].toarray()[0]
                potential_rs_flg &= (0 <= tps_p_r + tp_q - betas_p_r)
                if not potential_rs_flg.any():
                    continue

                tps_q_r = disj_tps[q].toarray()[0]
                potential_rs_flg &= (0 <= tps_q_r + tp_p - beta_p)
                if not potential_rs_flg.any():
                    continue

                betas_q_r = disj_betas[q].toarray()[0]
                potential_rs_flg &= (0 <= tps_q_r + tp_p - betas_q_r)
                if not potential_rs_flg.any():
                    continue

                potential_rs = potential_rs_flg.nonzero()[0]
                for r in potential_rs:
                    yield p, q, r

    def count_potentials_size3(
            self, thold: float,
            tps: np.ndarray, fps: np.ndarray,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
            types=(1, 2, 3, 4), use_tqdm: bool = False
    ) -> int:
        gen_t3_1 = self.iterate_potentials_type3_1(thold, tps, fps, conj_tps, conj_fps, use_tqdm)
        gen_t3_2 = self.iterate_potentials_type3_2(thold, tps, fps, conj_tps, conj_fps, disj_tps, disj_fps, use_tqdm)
        gen_t3_3 = self.iterate_potentials_type3_3(thold, tps, fps, conj_tps, conj_fps, disj_tps, disj_fps, use_tqdm)
        gen_t3_4 = self.iterate_potentials_type3_4(thold, tps, fps, disj_tps, disj_fps, use_tqdm)

        cnt = 0
        if 1 in types:
            cnt += sum(1 for _ in gen_t3_1)
        if 2 in types:
            cnt += sum(1 for _ in gen_t3_2)
        if 3 in types:
            cnt += sum(1 for _ in gen_t3_3)
        if 4 in types:
            cnt += sum(1 for _ in gen_t3_4)
        return cnt
