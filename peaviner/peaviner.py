from dataclasses import dataclass
import enum
from typing import Tuple, FrozenSet, Iterator, Generator, Union

import numpy as np
from scipy import sparse
from tqdm.auto import tqdm
from bitarray import frozenbitarray as fbitarray
from . import scores


class EnumContainMeta(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True


class Num2BinOperations(enum.Enum, metaclass=EnumContainMeta):
    GEQ = '>='
    LT = '<'


IntPremType = Tuple[int, Num2BinOperations, float]  # Feature_id, '>=' or '<', numeric threshold


@dataclass(repr=False)
class PeaViner:
    gamma: float = None
    atom_premises: Tuple[FrozenSet[IntPremType], ...] = None  # For each atomic extent, list its premises
    atom_extents: Tuple[fbitarray, ...] = None  # Tuple of extents that can be obtainedext_ by a single premise
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

    def generate_diextents(self, extents: Tuple[fbitarray, ...] = None, operations: tuple = ('^', 'v'), use_tqdm=False)\
            -> Tuple[Tuple[fbitarray, ...], Tuple[Tuple[int, int, str], ...]]:
        extents = extents if extents is not None else self.atom_extents

        diextents_stat = {}

        if use_tqdm:
            n_ops = len(operations)
            pbar = tqdm(total=len(extents)*(len(extents)-1)//2 * n_ops, desc='Generate pq, p|q')

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
            pbar = tqdm(total=n_exts*(n_exts-1)//2, desc='Compute stats for pq, p|q')

        for i, a in enumerate(extents):
            for j, b in enumerate(extents[i+1:]):
                c = a & b
                if c == a or c == b:
                    if use_tqdm:
                        pbar.update(1)
                    continue

                row.append(i)
                col.append(j+i+1)

                row.append(j+i+1)
                col.append(i)

                for _ in range(2):
                    conj_data_tp.append(scores.meas_tp(c, self.y))
                    conj_data_fp.append(scores.meas_fp(c, self.y))

                c = a | b
                for _ in range(2):
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

        assert len(conj_tp_mx.data) == len(conj_fp_mx.data)
        assert len(disj_tp_mx.data) == len(disj_fp_mx.data)

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
        for col_i, col in tqdm(enumerate(X.T), total=X.shape[1], disable=not use_tqdm, desc="Generate p's"):
            vals = np.sort(np.unique(col))
            vals = vals[~np.isnan(vals)]

            ext_chain_geq = []
            for v in vals:
                ext = col >= v
                ext = fbitarray(ext.tolist())
                ext_neg = fbitarray((col < v).tolist())

                ext_chain_geq.append(ext)

                p_geq, p_ngeq = (col_i, '>=', v), (col_i, '<', v)

                if ext in extents_i_map:
                    ext_i_ = extents_i_map[ext]
                    extents_prem_list[ext_i_].add(p_geq)
                else:
                    extents_i_map[ext] = ext_i
                    extents_prem_list.append({p_geq})
                    ext_i += 1

                if ext_neg in extents_i_map:
                    ext_i_neg_ = extents_i_map[ext_neg]
                    extents_prem_list[ext_i_neg_].add(p_ngeq)
                else:
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
        return thold*(gamma + fps)

    def iterate_potentials_type3_1(
            self,
            thold: float, thold_tp: float,
            tps: np.ndarray, fps: np.ndarray,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            use_tqdm: bool = False
    ) -> Generator[Tuple[int, int, int], Tuple[float, float, float], None]:
        """Iterating potential premises of type pqr"""
        yield None  # Placeholder yield

        gamma, omg = self.gamma, 1 - self.gamma

        ext_ids = np.arange(len(tps))
        potential_ps = (thold_tp <= conj_tps).sum(1).nonzero()[0]

        for p in tqdm(potential_ps, disable=not use_tqdm, desc='Iter pqr'):
            tps_pq = conj_tps[p].toarray()[0]  # list of values for q

            potential_qs_flag = thold_tp <= tps_pq
            potential_qs_flag &= p < ext_ids  # lexicographical order
            if not potential_qs_flag.any():
                continue

            potential_qs = potential_qs_flag.nonzero()[0]

            fp_p = fps[p]
            fps_r = fps
            tps_pr, fps_pr = [mx[p].toarray()[0] for mx in [conj_tps, conj_fps]]

            for q in potential_qs:
                potential_rs_flg = potential_qs_flag.copy()
                potential_rs_flg &= q < ext_ids  # lexicographical order

                fp_pq, alpha_pq = conj_fps[p, q], self.calc_alphas(conj_tps[p, q], thold, gamma)  # const
                potential_rs_flg &= (fp_pq + fps_r - alpha_pq <= omg)
                if not potential_rs_flg.any():
                    continue

                alphas = alphas_r = self.calc_alphas(tps, thold, gamma)
                potential_rs_flg &= (fp_pq + fps_r - alphas_r <= omg)
                if not potential_rs_flg.any():
                    continue

                fp_q, alpha_q = fps[q],  alphas[q]  # const
                alphas_pr = self.calc_alphas(tps_pr, thold, gamma)
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

                fps_qr = conj_fps[q].toarray()[0]  # list of values for r
                alphas_qr = self.calc_alphas(tps_qr, thold, gamma)
                potential_rs_flg &= (fps_qr + fp_p - alphas_qr <= omg)
                if not potential_rs_flg.any():
                    continue

                alpha_p = alphas[p]
                potential_rs_flg &= (fps_qr + fp_p - alpha_p <= omg)
                potential_rs = potential_rs_flg.nonzero()[0]
                for r in potential_rs:
                    thold, thold_tp, _ = yield p, q, r

    def iterate_potentials_type3_2(
            self,
            thold: float, thold_tp: float, thold_fp: float,
            tps: np.ndarray, fps: np.ndarray,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
            use_tqdm: bool = False
    ) -> Generator[Tuple[int, int, int], Tuple[float, float, float], None]:
        """Iterating potential premises of type pq|r"""
        yield None

        gamma, omg = self.gamma, 1 - self.gamma

        ext_ids = np.arange(len(tps))
        potential_ps = (thold_tp <= conj_tps).sum(1).nonzero()[0]

        for p in tqdm(potential_ps, disable=not use_tqdm, desc='Iter pq|r'):
            fps_pq = conj_fps[p].toarray()[0]   # list of values for q

            potential_qs_flag = fps_pq <= thold_fp
            potential_qs_flag &= p < ext_ids  # lexicographical order
            if not potential_qs_flag.any():
                continue

            potential_qs = potential_qs_flag.nonzero()[0]

            tps_r, fps_r = tps, fps  # list of values for r
            tps_p_r, fps_p_r = [mx[p].toarray()[0] for mx in [disj_tps, disj_fps]]  # list of values for r

            for q in potential_qs:
                potential_rs_flg = (fps_r <= thold_fp)

                tp_pq = conj_tps[p, q]
                alpha_p_q = self.calc_alphas(disj_tps[p, q], thold, gamma)
                beta_pq = self.calc_betas(conj_fps[p, q], thold, gamma)

                potential_rs_flg &= (0 <= tp_pq + tps_r - beta_pq)
                if not potential_rs_flg.any():
                    continue

                betas_r = self.calc_betas(fps, thold, gamma)
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

                alphas_q_r = self.calc_alphas(tps_q_r, thold, gamma)
                potential_rs_flg &= (fps_p_r + fps_q_r - alphas_q_r <= omg)
                if not potential_rs_flg.any():
                    continue

                potential_rs = potential_rs_flg.nonzero()[0]
                for r in potential_rs:
                    thold, thold_tp, thold_fp = yield p, q, r

    def iterate_potentials_type3_3(
            self,
            thold: float, thold_tp: float, thold_fp: float,
            tps: np.ndarray, fps: np.ndarray,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
            use_tqdm: bool = False
    ) -> Generator[Tuple[int, int, int], Tuple[float, float, float], None]:
        """Iterating potential premises of type (p|q)r"""
        yield None  # Placeholder yield

        gamma, omg = self.gamma, 1 - self.gamma

        ext_ids = np.arange(len(tps))

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
            potential_qs_flag &= p < ext_ids  # lexicographical order
            if not potential_qs_flag.any():
                continue

            potential_qs = potential_qs_flag.nonzero()[0]

            tps_r, fps_r = tps, fps  # list of values for r
            tps_pr, fps_pr, tps_p_r, fps_p_r = [mx[p].toarray()[0] for mx in [conj_tps, conj_fps, disj_tps, disj_fps]]

            for q in potential_qs:
                potential_rs_flg = (thold_tp <= tps_r)
                if not potential_rs_flg.any():
                    continue

                fp_p_q = disj_fps[p, q]
                alphas_r = self.calc_alphas(tps, thold, gamma)
                potential_rs_flg &= (fp_p_q + fps_r - alphas_r <= omg)
                if not potential_rs_flg.any():
                    continue

                alpha_p_q = self.calc_alphas(disj_tps[p, q], thold, gamma)
                potential_rs_flg &= (fp_p_q + fps_r - alpha_p_q <= omg)
                if not potential_rs_flg.any():
                    continue

                tps_qr = conj_tps[q].toarray()[0]
                betas_pr = self.calc_betas(fps_pr, thold, gamma)
                potential_rs_flg &= (0 <= tps_pr + tps_qr - betas_pr)
                if not potential_rs_flg.any():
                    continue

                fps_qr = conj_fps[q].toarray()[0]
                potential_rs_flg &= (fps_qr <= thold_fp)
                if not potential_rs_flg.any():
                    continue

                potential_rs_flg &= (fps_pr <= thold_fp)
                if not potential_rs_flg.any():
                    continue

                betas_qr = self.calc_betas(fps_qr, thold, gamma)
                potential_rs_flg &= (0 <= tps_pr + tps_qr - betas_qr)
                if not potential_rs_flg.any():
                    continue

                potential_rs = potential_rs_flg.nonzero()[0]
                for r in potential_rs:
                    thold, thold_tp, thold_fp = yield p, q, r

    def iterate_potentials_type3_4(
            self,
            thold: float, thold_fp: float,
            tps: np.ndarray, fps: np.ndarray,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
            use_tqdm: bool = False
    ) -> Generator[Tuple[int, int, int], Tuple[float, float, float], None]:
        """Iterating potential premises of type p|q|r"""
        yield None  # Placeholder yield

        gamma, omg = self.gamma, 1 - self.gamma

        ext_ids = np.arange(len(tps))

        disj_tns = disj_fps.copy()
        disj_tns.data = omg - disj_tns.data
        thold_tn = omg - thold_fp
        potential_ps_flg = np.array((disj_tns >= thold_tn).sum(1) > 0).flatten()
        del disj_tns, thold_tn
        potential_ps = potential_ps_flg.nonzero()[0]

        for p in tqdm(potential_ps, disable=not use_tqdm, desc='Iter p|q|r'):
            fps_p_q = disj_fps[p].toarray()[0]  # list of values for q

            potential_qs_flag = fps_p_q <= thold_fp
            potential_qs_flag &= p < ext_ids  # lexicographical order
            if not potential_qs_flag.any():
                continue

            potential_qs = potential_qs_flag.nonzero()[0]

            tp_p = tps[p]
            tps_r = tps
            fps_p_r = disj_fps[p].toarray()[0]

            for q in potential_qs:
                potential_rs_flg = (fps_p_r <= thold_fp)
                potential_rs_flg &= q < ext_ids  # lexicographical order
                if not potential_rs_flg.any():
                    continue

                tp_p_q = disj_tps[p, q]
                beta_p_q = self.calc_betas(disj_fps[p, q], thold, gamma)
                potential_rs_flg &= (0 <= tp_p_q + tps_r - beta_p_q)
                if not potential_rs_flg.any():
                    continue

                betas = betas_r = self.calc_betas(fps, thold, gamma)
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

                betas_p_r = self.calc_betas(fps_p_r, thold, gamma)
                potential_rs_flg &= (0 <= tps_p_r + tp_q - betas_p_r)
                if not potential_rs_flg.any():
                    continue

                tps_q_r = disj_tps[q].toarray()[0]
                beta_p = betas[p]
                potential_rs_flg &= (0 <= tps_q_r + tp_p - beta_p)
                if not potential_rs_flg.any():
                    continue

                betas_q_r = self.calc_betas(fps_q_r, thold, gamma)
                potential_rs_flg &= (0 <= tps_q_r + tp_p - betas_q_r)
                if not potential_rs_flg.any():
                    continue

                potential_rs = potential_rs_flg.nonzero()[0]
                for r in potential_rs:
                    thold, _, thold_fp = yield p, q, r

    def count_potentials_size3(
            self, thold: float,
            tps: np.ndarray, fps: np.ndarray,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
            types=(1, 2, 3, 4), use_tqdm: bool = False
    ) -> int:
        thold_tp, thold_fp = self.calc_thold_tpfp(self.gamma, thold, 'Jaccard')

        gen_t3_1 = self.iterate_potentials_type3_1(
            thold, thold_tp, tps, fps, conj_tps, conj_fps, use_tqdm)
        gen_t3_2 = self.iterate_potentials_type3_2(
            thold, thold_tp, thold_fp, tps, fps, conj_tps, conj_fps, disj_tps, disj_fps, use_tqdm)
        gen_t3_3 = self.iterate_potentials_type3_3(
            thold, thold_tp, thold_fp, tps, fps, conj_tps, conj_fps, disj_tps, disj_fps, use_tqdm)
        gen_t3_4 = self.iterate_potentials_type3_4(
            thold, thold_fp, tps, fps, disj_tps, disj_fps, use_tqdm)

        cnt = 0
        for t in types:
            gen = (None, gen_t3_1, gen_t3_2, gen_t3_3, gen_t3_4)[t]
            next(gen)
            while True:
                try:
                    _ = gen.send((thold, thold_tp, thold_fp))
                except StopIteration:
                    break
                cnt += 1

        return cnt

    def find_best_premises_size3(
            self, thold: float, k: int,
            tps: np.ndarray, fps: np.ndarray,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
            update_thold: bool = True, return_n_iters: bool = False,
            types=(1, 2, 3, 4), use_tqdm: bool = False
    ) -> Union[Tuple[Tuple[Tuple[int, int, int], int, float], ...],
         Tuple[Tuple[Tuple[Tuple[int, int, int], int, float], ...], int]]:
        score_name = 'Jaccard'
        thold_tp, thold_fp = self.calc_thold_tpfp(self.gamma, thold, score_name)

        aexts = self.atom_extents
        ext_f_dict = {
            1: lambda p, q, r: aexts[p] & aexts[q] & aexts[r],
            2: lambda p, q, r: aexts[p] & aexts[q] | aexts[r],
            3: lambda p, q, r: (aexts[p] | aexts[q]) & aexts[r],
            4: lambda p, q, r: aexts[p] | aexts[q] | aexts[r]
        }

        n_iters = 0
        best_premises = []
        for t in types:
            ext_f = ext_f_dict[t]

            if t == 1:
                gen = self.iterate_potentials_type3_1(thold, thold_tp, tps, fps, conj_tps, conj_fps, use_tqdm)
            elif t == 2:
                gen = self.iterate_potentials_type3_2(
                    thold, thold_tp, thold_fp, tps, fps, conj_tps, conj_fps, disj_tps, disj_fps, use_tqdm)
            elif t == 3:
                gen = self.iterate_potentials_type3_3(
                    thold, thold_tp, thold_fp, tps, fps, conj_tps, conj_fps, disj_tps, disj_fps, use_tqdm)
            elif t == 4:
                gen = self.iterate_potentials_type3_4(thold, thold_fp, tps, fps, disj_tps, disj_fps, use_tqdm)
            else:
                raise ValueError(f'Unsupported type of premise: {t}')
            next(gen)

            while True:
                try:
                    comb = gen.send((thold, thold_tp, thold_fp))
                except StopIteration:
                    break

                n_iters += 1
                ext = ext_f(*comb)
                score = scores.meas_jacc(ext, self.y)
                if score >= thold:
                    i = 0
                    for prem_data in best_premises:
                        if prem_data[2] < score:
                            break
                        i += 1
                    best_premises.insert(i, (comb, t, score))

                    if update_thold:
                        try:
                            while True:
                                best_premises.pop(k)
                        except IndexError:
                            pass

                        thold = best_premises[-1][2]
                        thold_tp, thold_fp = self.calc_thold_tpfp(self.gamma, thold, score_name)

        best_premises = tuple(best_premises)

        if return_n_iters:
            return best_premises, n_iters
        return best_premises

    def find_best_premises_size2(
            self, k: int,
            conj_tps: sparse.csr_matrix, conj_fps: sparse.csr_matrix,
            disj_tps: sparse.csr_matrix, disj_fps: sparse.csr_matrix,
    ) -> Tuple[Tuple[Tuple[int, int], int, float], ...]:
        score_name = 'Jaccard'
        thold = 0

        best_premises = []

        for p in range(conj_tps.shape[0] - 1000):
            tps_conj_p = conj_tps[p].toarray()[0]
            fps_conj_p = conj_fps[p].toarray()[0]

            jaccs_conj_p = tps_conj_p / (self.gamma + fps_conj_p)
            jaccs_conj_p[:p] = 0
            jaccs_conj_p[jaccs_conj_p < thold] = 0

            potential_qs = jaccs_conj_p.nonzero()[0]
            best_premises += [((p, q + p + 1), 1, jaccs_conj_p[q]) for q in potential_qs]

            tps_disj_p = disj_tps[p].toarray()[0]
            fps_disj_p = disj_fps[p].toarray()[0]

            jaccs_disj_p = tps_disj_p / (self.gamma + fps_disj_p)
            jaccs_disj_p[:p] = 0
            jaccs_disj_p[jaccs_disj_p < thold] = 0

            potential_qs = jaccs_conj_p.nonzero()[0]
            best_premises += [((p, q + p + 1), 2, jaccs_disj_p[q]) for q in potential_qs]

            best_premises = sorted(best_premises, key=lambda prem_data: -prem_data[2])[:k]
            if len(best_premises) >= k:
                thold = best_premises[-1][2]

        best_premises = tuple(best_premises)
        return best_premises
