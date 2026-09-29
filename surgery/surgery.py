"""
surgery.py — MergedPatch, seam construction, stim circuit assembly

핵심 설계:
    - Seam은 새로운 Qubit("seam", i) 객체들로 기술
    - MergedPatch는 세 개의 qubit set (patch1, seam, patch2)을 가짐
    - global QubitRegistry는 MergedPatch가 만들고 외부에 노출
    - stim circuit은 이 registry 기준으로 생성됨
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from core.CSS import LinearCode
from core.primitives import Qubit, Check, QubitRegistry
from core.codes import *
from docplex.mp.model import Model
import itertools


# ---------------------------------------------------------------------------
# MergedPatch
# ---------------------------------------------------------------------------

class MergedPatch(CodePatch):
    """
    두 CodePatch를 seam check들로 연결한 merged code.

    Qubit namespace:
        patch1.data_qubits()  — patch1의 Qubit 객체 그대로
        seam_qubits           — Qubit("seam", 0..n_seam-1)
        patch2.data_qubits()  — patch2의 Qubit 객체 그대로

    세 set이 모두 distinct하므로 offset 계산 불필요.
    registry가 이 순서로 int index를 부여.
    """

    def __init__(
        self,
        patch1: CodePatch,
        patch2: CodePatch,
        seam_data_qubits: list[Qubit],
        seam_x_checks: list[Check],
        seam_z_checks: list[Check],
        name: str = "merged",
    ):
        self._name = name
        self.patch1 = patch1
        self.patch2 = patch2
        self._seam_qubits = seam_data_qubits

        self._x_checks = (
            patch1.x_checks() +
            seam_x_checks +
            patch2.x_checks()
        )
        self._z_checks = (
            patch1.z_checks() +
            seam_z_checks +
            patch2.z_checks()
        )

        self._seam_x_slice = slice(
            len(patch1.x_checks()),
            len(patch1.x_checks()) + len(seam_x_checks)
        )
        self._seam_z_slice = slice(
            len(patch1.z_checks()),
            len(patch1.z_checks()) + len(seam_z_checks)
        )

        self._x_logicals = patch1.x_logicals() + patch2.x_logicals()
        self._z_logicals = patch1.z_logicals() + patch2.z_logicals()

        self._registry = QubitRegistry()
        self._registry.register(patch1.data_qubits())
        self._registry.register(seam_data_qubits)
        self._registry.register(patch2.data_qubits())

    @property
    def name(self): return self._name

    def data_qubits(self) -> list[Qubit]:
        return (self.patch1.data_qubits() +
                self._seam_qubits +
                self.patch2.data_qubits())

    def x_checks(self) -> list[Check]: return self._x_checks
    def z_checks(self) -> list[Check]: return self._z_checks
    def x_logicals(self):              return self._x_logicals
    def z_logicals(self):              return self._z_logicals

    def registry(self) -> QubitRegistry:
        return self._registry

    def seam_x_checks(self) -> list[Check]:
        return self._x_checks[self._seam_x_slice]

    def seam_z_checks(self) -> list[Check]:
        return self._z_checks[self._seam_z_slice]

    def make_registry(self) -> QubitRegistry:
        return self._registry


class DirectSum(CodePatch):
    """두 코드 패치의 direct sum. check는 단순히 합집합."""

    def __init__(self, patch1: CodePatch, patch2: CodePatch, name: str = "direct-sum"):
        self._name = name
        self.patch1 = patch1
        self.patch2 = patch2

        self._x_checks = patch1.x_checks() + patch2.x_checks()
        self._z_checks = patch1.z_checks() + patch2.z_checks()
        
        self._registry = QubitRegistry()
        self._registry.register(patch1.data_qubits())
        self._registry.register(patch2.data_qubits())

    @property
    def name(self): return self._name

    def data_qubits(self) -> list[Qubit]:
        return self.patch1.data_qubits() + self.patch2.data_qubits()

    def x_checks(self) -> list[Check]: return self._x_checks
    def z_checks(self) -> list[Check]: return self._z_checks
    def make_registry(self) -> QubitRegistry:
        return self._registry
    def local_coord(self) -> tuple[float, float]: raise NotImplementedError


# ---------------------------------------------------------------------------
# ConePatch: cone code construction for homological measurement
# ---------------------------------------------------------------------------
# 논문: "Fault-Tolerant Logical Measurements via Homological Measurement"
#       Ide, Gowda, Nadkarni, Dauphinais (PRX 2025)
#
# C̃ 의 check matrix (eq.13):
#     H̃_X = [ H_X         ]    H̃_Z = [ H_Z   f0  ]
#           [ f1^T  ∂1^T  ]          [       ∂0  ]

class ConePatch(CodePatch):
    """
    Cone code C̃ = cone(f) for homological measurement.
    (Ide, Gowda, Nadkarni, Dauphinais, PRX 2025)

        H̃_X = [ H_X   0   ]      H̃_Z = [ H_Z   f0 ]
              [ f1^T  ∂1^T ]            [  0    ∂0 ]

    데이터 블록은 **리스트** 로 받는다.  Cs = [C_1, ..., C_m] 이고
    C = C_1 ⊕ ... ⊕ C_m 로 취급한다.  m=1 이면 단일 블록 surgery,
    m=2 면 두 블록 merge.  (예전의 _ConePatch / ConePatch 를 통합한 것.)

    입력:
        Cs   : list[CodePatch]  — 데이터 블록들
        A    : CodePatch        — ancilla
        f1t  : list[Check]      — A 의 i 번째 X-check 에 추가될 데이터 qubit 조합
        f0   : list[Check]      — 데이터의 i 번째 Z-check 에 추가될 ancilla qubit 조합
                                  (순서는 C_1.z_checks() + ... + C_m.z_checks())
    """

    def __init__(
        self,
        Cs,                       # list[CodePatch]  (또는 단일 CodePatch)
        A: CodePatch,
        f1t: list[Check],
        f0: list[Check],
        name: str = "cone",
    ):
        if isinstance(Cs, CodePatch):
            Cs = [Cs]
        self._name = name
        self.Cs = list(Cs)
        self.A = A
        self._f0 = f0
        self._f1t = f1t

        c_qubits = [q for C in self.Cs for q in C.data_qubits()]
        a_qubits = A.data_qubits()
        self._data_qubits = c_qubits + a_qubits

        self._registry = QubitRegistry()
        self._registry.register(c_qubits)
        self._registry.register(a_qubits)

        self.patch_name_dict = {C._name: C for C in self.Cs}
        self.patch_name_dict[A._name] = A

        partial0  = A.z_checks() if A.z_checks() else []
        partial1t = A.x_checks() if A.x_checks() else []

        # local_coord_chk 용 블록별 경계
        self._n_zC = [len(C.z_checks()) for C in self.Cs]
        self._n_xC = [len(C.x_checks()) for C in self.Cs]

        z_all = [chk for C in self.Cs for chk in C.z_checks()]
        x_all = [chk for C in self.Cs for chk in C.x_checks()]

        assert len(z_all) == len(f0), f"f0 길이 {len(f0)} != Z-check 총 {len(z_all)}"
        self._z_checks = list(z_all)
        for i in range(len(f0)):
            if f0[i].weight() > 0:
                self._z_checks[i] += f0[i]
        self._z_checks.extend(partial0)

        assert len(partial1t) == len(f1t), \
            f"f1t 길이 {len(f1t)} != A 의 X-check {len(partial1t)}"
        self._x_checks = list(x_all)
        for i in range(len(f1t)):
            self._x_checks.append(f1t[i] + partial1t[i])

        self.css = None       # CodePatch.__init__ 을 안 거치므로 직접 초기화

    # -- 편의 접근자 (예전 코드 호환) -----------------------------------------
    @property
    def C(self):
        assert len(self.Cs) == 1, "블록이 여러 개다. self.Cs 를 쓸 것."
        return self.Cs[0]

    @property
    def C1(self): return self.Cs[0]

    @property
    def C2(self):
        assert len(self.Cs) >= 2, "블록이 하나뿐이다."
        return self.Cs[1]

    @property
    def name(self) -> str: return self._name

    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self)   -> list[Check]:  return self._x_checks
    def z_checks(self)   -> list[Check]:  return self._z_checks
    def make_registry(self) -> QubitRegistry: return self._registry

    def x_logicals(self):
        raise NotImplementedError(
            "logical 대표원은 init_logical_representatives() 로 css 에 설정된다. "
            "self.css.logical_x_repr 를 쓸 것.")

    def z_logicals(self):
        raise NotImplementedError(
            "logical 대표원은 init_logical_representatives() 로 css 에 설정된다. "
            "self.css.logical_z_repr 를 쓸 것.")

    def local_coord(self, q: Qubit) -> tuple[float, float]:
        lx, ly = self.patch_name_dict[q.patch_name].local_coord(q)
        ox, oy = self.patch_name_dict[q.patch_name]._offset
        return lx + ox, ly + oy

    def local_coord_chk(self, typ: str, idx: int) -> tuple[float, float]:
        """merged 체크의 좌표. C_1, ..., C_m 순서로 이어붙인 뒤 A 가 온다.
        f1t/f0 는 기존 체크에 큐빗을 덧붙이기만 하므로 체크의 정체성은
        원래 패치 것 그대로다."""
        if typ == "X":
            ns = self._n_xC
        elif typ == "Z":
            ns = self._n_zC
        else:
            raise ValueError("typ must be 'X' or 'Z'")
        local = idx
        for C, n in zip(self.Cs, ns):
            if local < n:
                lx, ly = C.local_coord_chk(typ, local)
                ox, oy = C._offset
                return lx + ox, ly + oy
            local -= n
        lx, ly = self.A.local_coord_chk(typ, local)
        ox, oy = self.A._offset
        return lx + ox, ly + oy

    def get_idx(self, q: Qubit) -> int:
        return self._registry[q]

    def h_x_tilde(self) -> np.ndarray:
        """H̃_X = [H_X | 0 ; f1^T | ∂1^T].  x_checks() 에 이미 implicit."""
        raise NotImplementedError

    def h_z_tilde(self) -> np.ndarray:
        """H̃_Z = [H_Z | f0 ; ∂0 | 0]"""
        raise NotImplementedError


class _ConePatch(ConePatch):
    """DEPRECATED. ConePatch 가 데이터 블록 리스트를 받게 일반화되면서 흡수됨.
    _ConePatch(C, A, f1t, f0) == ConePatch([C], A, f1t, f0)."""
    def __init__(self, C, A, f1t, f0, name: str = "cone"):
        super().__init__([C], A, f1t, f0, name=name)


class MergedCodeWithGraph(ConePatch):
    """ 
    Codepatch C1, C2를 AncillaGraph를 통해 merge.
    일단 logical XX 측정만 구현
    C1, C2의 logical X operator들의 support가 port function을 통해 graph의 vertex(X-check)로 injective하게 mapping(각 codeblock 내에서의 injective)
    edge(ancilla qubit)가 C1, C2의 Z-check의 support에 추가되는데, codeblock Ci의 Z-check s에 대해 logical X operator의 support와 overlap하는 qubit들을 L_s라고 하자.
    port function은 L_s를 V_s=f(L_s)\subseteq V로 mapping하는데, V_s의 perfect matching에 대응되는 edge들이 Z-check s에 추가된다.
    이를 고려해서 ConePatch를 만들어주는 class.

    dict = None이면 default로 mapping해줌.
    """
    def __init__(
        self,
        C1: CodePatch,
        C2: CodePatch,
        anc_graph: AncillaGraph,
        X1: list[Qubit],
        X2: list[Qubit],
        port_func: dict|None = None,
        name: str = "merged-graph",
    ):
        self._name = name
        self.C1 = C1
        self.C2 = C2
        self.anc_graph = anc_graph
        self.X1 = X1
        self.X2 = X2
        self.direct_sum = DirectSum(C1, C2)

        # # qubit register
        # c_qubits = self.direct_sum.data_qubits()
        # a_qubits = self.anc_graph.data_qubits()
        # self._data_qubits = c_qubits + a_qubits

        # self._registry = QubitRegistry()
        # self._registry.register(c_qubits)
        # self._registry.register(a_qubits)

        # self.patch_name_dict = {
        #     C1.name: C1,
        #     C2.name: C2,
        #     anc_graph.name: anc_graph
        # }

        # port function

        assert max(len(X1), len(X2)) <= len(anc_graph.x_checks())   # for injectivity condition

        if port_func == None:
            self.port_func = {}
            # Qubit -> check idx 매핑
            for i, q in enumerate(X1):
                self.port_func[q] = i
            for i, q in enumerate(X2):
                self.port_func[q] = i
            
        else:
            self.port_func = port_func
        
        # check construction

        ## Ancilla X check에 추가할 qubit들
        f1t = [Check.get_empty_check()] * len(anc_graph.x_checks())
        for q in X1:
            idx = self.port_func[q]
            f1t[idx] += Check(frozenset([q]), "X")
        for q in X2:
            idx = self.port_func[q]
            f1t[idx] += Check(frozenset([q]), "X")

        ## C1, C2의 Z check에 추가할 qubit들
        ### C1 Z check 개수
        cnt_z1 = len(C1.z_checks())
        f0 = [Check.get_empty_check()] * len(self.direct_sum.z_checks())
        for i, s in enumerate(C1.z_checks()):
            L_s = [q for q in s.qubits if q in X1]
            if len(L_s) == 0:
                continue
            V_s = [self.port_func[q] for q in L_s]
            matching = anc_graph.get_perfect_matching(V_s)
            for src, tar, path in matching:
                for edge in path:
                    q_edge = anc_graph.edge_index[edge]
                    f0[i] += Check(frozenset([anc_graph.data_qubits()[q_edge]]), "Z")
        for i, s in enumerate(C2.z_checks(), start=cnt_z1):
            L_s = [q for q in s.qubits if q in X2]
            if len(L_s) == 0:
                continue
            V_s = [self.port_func[q] for q in L_s]
            matching = anc_graph.get_perfect_matching(V_s)
            for src, tar, path in matching:
                for edge in path:
                    q_edge = anc_graph.edge_index[edge]
                    f0[i] += Check(frozenset([anc_graph.data_qubits()[q_edge]]), "Z")

        super().__init__(C1, C2, anc_graph, f1t, f0, name)
                
    
    @property
    def name(self) -> str: return self._name
    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self)   -> list[Check]:  return self._x_checks
    def z_checks(self)   -> list[Check]:  return self._z_checks
    def x_logicals(self):
        raise NotImplementedError(
            "logical 대표원은 init_logical_representatives() 로 css 에 설정된다. "
            "self.css.logical_x_repr 를 쓸 것.")
    def z_logicals(self):
        raise NotImplementedError(
            "logical 대표원은 init_logical_representatives() 로 css 에 설정된다. "
            "self.css.logical_z_repr 를 쓸 것.")
    def make_registry(self) -> QubitRegistry: return self._registry

    # def local_coord(self, q: Qubit) -> tuple[float, float]:
    #     # ConePatch에서의 local_coord
    #     lx, ly = q._local_coord()
    #     ox, oy = self.patch_name_dict[q.patch_name]._offset
    #     return lx + ox, ly + oy

    # def get_idx(self, q: Qubit) -> int:
    #     return self._registry[q]
def _v(x):
    return "".join(str(int(b)) for b in np.asarray(x, dtype=int))
 
class Homomorphism:
    """A -> B classical code homomorphism.
       ltol : (k_B, k_A)  logical 방향 지정
       ctol : (n_B, r_A)  complement 방향 지정 (r_A = n_A - k_A). None 이면 자동 탐색."""
 
    def __init__(self, A, B, ltol, ctol=None, auto=True, greedy=True, **kw):
        self.A, self.B = A, B
        self.F = A.F
        self.rA = A.n - A.k
        self.ltol = self.F(np.asarray(ltol, dtype=int))
        self.ctol_stats = None
        if ctol is None:
            ctol, self.ctol_stats = (self.search_ctol_greedy(**kw) if greedy
                                     else self.search_ctol_exhaustive(**kw))
        self.ctol = self.F(np.asarray(ctol, dtype=int))
        self._validate()
        self.gamma1 = None
        self.gamma0 = None
        if auto:
            self.set_map()
            self.set_gamma0()
 
    def _validate(self):
        assert self.ltol.shape == (self.B.k, self.A.k), \
            f"ltol must be (k_B,k_A)=({self.B.k},{self.A.k}), got {self.ltol.shape}"
        assert self.ctol.shape == (self.B.n, self.rA), \
            f"ctol must be (n_B,rank H_A)=({self.B.n},{self.rA}), got {self.ctol.shape}"
 
    # ------------------------------------------------------------ ctol 탐색
    # gamma1 = C0 + ctol @ Q 이고 ctol 의 i번째 행은 gamma1 의 i번째 행에만 영향.
    # -> 행별로 독립이라 "행별 최소 무게" 로 시작해서 열 무게 초과분만 고치면 된다.
 
    def _row_candidates(self, max_cand=1 << 16):
        """각 data qubit 행 i 에 대해 (gamma1 행벡터들, ctol 행벡터들)."""
        if (1 << self.rA) > max_cand:
            raise ValueError(f"2^r_A = {1 << self.rA} > max_cand; r_A 가 너무 큼")
        Pinv = np.linalg.inv(self._basis_change())
        C0 = np.asarray((self.B.G.T @ self.ltol) @ Pinv[:self.A.k, :], dtype=int)
        Q = np.asarray(Pinv[self.A.k:, :], dtype=int)
        xs = np.array(list(itertools.product([0, 1], repeat=self.rA)), dtype=int)
        rows_add = (xs @ Q) % 2
        return [((C0[i] + rows_add) % 2, xs) for i in range(self.B.n)]
 
    @staticmethod
    def _objective(rows):
        mr, mc = int(rows.sum(axis=1).max()), int(rows.sum(axis=0).max())
        return (max(mr, mc), mc, mr, int(rows.sum()))
 
    @staticmethod
    def _stats(o):
        return dict(max_degree=o[0], max_col_wt=o[1], max_row_wt=o[2], total_wt=o[3])
 
    def search_ctol_greedy(self, n_restarts=30, max_iter=500, seed=0, max_cand=1 << 16):
        """행별 최소무게 초기화 -> 과부하 열을 고치는 local search (random restart)."""
        cands = self._row_candidates(max_cand)
        nB = self.B.n
        rng = np.random.default_rng(seed)
        wts = [c[0].sum(axis=1) for c in cands]
        best = None
        for t in range(n_restarts):
            pick = []
            for i in range(nB):
                w = wts[i]
                lo = w.min() if t == 0 else w.min() + (rng.integers(0, 2) if w.max() > w.min() else 0)
                opts = np.nonzero(w <= lo)[0]
                pick.append(int(rng.choice(opts)) if t else int(opts[0]))
            pick = np.array(pick)
            rows = np.array([cands[i][0][pick[i]] for i in range(nB)])
            obj = self._objective(rows)
            for _ in range(max_iter):
                improved = False
                for i in rng.permutation(nB):
                    cur = pick[i]
                    for j in range(cands[i][0].shape[0]):
                        if j == cur:
                            continue
                        rows[i] = cands[i][0][j]
                        o = self._objective(rows)
                        if o < obj:
                            obj, pick[i], cur, improved = o, j, j, True
                        else:
                            rows[i] = cands[i][0][cur]
                if not improved:
                    break
            if best is None or obj < best[0]:
                best = (obj, pick.copy())
        obj, pick = best
        ctol = np.array([cands[i][1][pick[i]] for i in range(nB)])
        return ctol, self._stats(obj)
 
    def search_ctol_exhaustive(self, max_enum=1 << 22, max_cand=1 << 16):
        """작은 경우용 전수 탐색 (정답 확인용)."""
        cands = self._row_candidates(max_cand)
        nc = cands[0][0].shape[0]
        if nc ** self.B.n > max_enum:
            raise ValueError(f"search space {nc ** self.B.n} > max_enum")
        best = None
        for combo in itertools.product(range(nc), repeat=self.B.n):
            rows = np.array([cands[i][0][combo[i]] for i in range(self.B.n)])
            o = self._objective(rows)
            if best is None or o < best[0]:
                best = (o, combo)
        o, combo = best
        ctol = np.array([cands[i][1][combo[i]] for i in range(self.B.n)])
        return ctol, self._stats(o)
 
    # ------------------------------------------------------------ 구성
    def _basis_change(self):
        """P: (logical ⊕ complement) 좌표 -> A_1 standard 좌표."""
        A, F = self.A, self.F
        free = A.complement()
        assert len(free) == self.rA
        E = np.zeros((A.n, self.rA), dtype=int)
        for c, j in enumerate(free):
            E[j, c] = 1
        return F(np.hstack((np.asarray(A.G.T, dtype=int), E)))
 
    def set_map(self):
        F = self.F
        left = self.B.G.T @ self.ltol
        M = F(np.hstack((np.asarray(left, dtype=int), np.asarray(self.ctol, dtype=int))))
        self.gamma1 = M @ np.linalg.inv(self._basis_change())
        return self.gamma1
 
    @staticmethod
    def _solve_left(H, Y, F):
        """X with X @ H = Y over GF(2). rank-deficient / redundant H 도 처리."""
        m, n = H.shape
        R = np.asarray(F(np.hstack((np.asarray(H.T, dtype=int),
                                    np.asarray(Y.T, dtype=int)))).row_reduce(), dtype=int)
        for r in R:
            if not r[:m].any() and r[m:].any():
                return None
        XT = F(np.zeros((m, Y.shape[0]), dtype=int))
        for r in R:
            nz = np.nonzero(r[:m])[0]
            if len(nz):
                XT[nz[0], :] = F(r[m:])
        return XT.T
 
    def set_gamma0(self):
        if self.gamma1 is None:
            self.set_map()
        g0 = self._solve_left(self.A.H, self.B.H @ self.gamma1, self.F)
        assert g0 is not None, "no gamma0: gamma1(ker H_A) not in ker H_B"
        self.gamma0 = g0
        return self.gamma0
 
    def verify(self):
        if self.gamma0 is None:
            self.set_gamma0()
        assert np.array_equal(self.gamma0 @ self.A.H, self.B.H @ self.gamma1), "diagram not commuting"
        assert not np.any(self.B.H @ (self.gamma1 @ self.A.G.T)), "kernel condition violated"
        return True
 
    # ------------------------------------------------------------ 분석
    def degrees(self):
        g = np.asarray(self.gamma1, dtype=int)
        return dict(max_col_wt=max(int(g[:, j].sum()) for j in range(self.A.n)),
                    max_row_wt=max(int(g[i, :].sum()) for i in range(self.B.n)),
                    total_wt=int(g.sum()))
 
    def logical_action(self):
        """[(이름, A의 logical, 상, B에서의 label)]"""
        out = []
        for i, L in enumerate(self.A.logicals()):
            img = self.gamma1 @ L
            out.append((f"L{i+1}(A)", L, img, self.B.label(img)))
        return out
 
    def connections(self):
        """ancilla bit -> data qubit"""
        g = np.asarray(self.gamma1, dtype=int)
        return {j: [int(x) for x in np.nonzero(g[:, j])[0]] for j in range(self.A.n)}
 
    def check_action(self):
        """A 의 check -> B 의 check"""
        g0 = np.asarray(self.gamma0, dtype=int)
        return {j: [int(t) for t in np.nonzero(g0[:, j])[0]] for j in range(g0.shape[1])}
 
    def report(self, show_matrices=True):
        print("=" * 60)
        print(f"Homomorphism  A[{self.A.n},{self.A.k},{self.A.get_distance()}]"
              f"  ->  B[{self.B.n},{self.B.k},{self.B.get_distance()}]")
        print("=" * 60)
        if self.ctol_stats:
            print("ctol search:", self.ctol_stats)
        if show_matrices:
            print("ltol:\n", np.asarray(self.ltol, dtype=int))
            print("ctol:\n", np.asarray(self.ctol, dtype=int))
            print("\ngamma1:\n", np.asarray(self.gamma1, dtype=int))
            print("gamma0:\n", np.asarray(self.gamma0, dtype=int))
        print("\ndegrees:", self.degrees())
 
        print("\nlogical action:")
        for nm, L, img, lab in self.logical_action():
            wl = int(np.asarray(L, dtype=int).sum())
            wi = int(np.asarray(img, dtype=int).sum())
            print(f"   {nm} = {_v(L)} (wt {wl})  ->  {_v(img)} (wt {wi})  =  {lab}")
 
        print("\nconnections (ancilla bit -> data qubit):")
        for j, t in self.connections().items():
            print(f"   {j} -> {t}" + ("   << 빈 연결" if not t else ""))
 
        print("\ncheck action (A check -> B check):")
        for j, t in self.check_action().items():
            print(f"   {j} -> {t}" + ("   << 빈 연결" if not t else ""))
 
        print("\ncommuting on complement basis:")
        for j in self.A.complement():
            e = self.F(np.eye(self.A.n, dtype=int)[j])
            l, r = self.gamma0 @ (self.A.H @ e), self.B.H @ (self.gamma1 @ e)
            print(f"   e_{j}: gamma0(H_A e)={_v(l)}  H_B(gamma1 e)={_v(r)}  "
                  f"{'OK' if np.array_equal(l, r) else 'MISMATCH'}")
        print(f"\nverify(): {self.verify()}")

class TrivialHomomorphism(Homomorphism):
    """A -> B trivial homomorphism. ltol = 0, ctol = 0."""
    def __init__(self, A, B):
        ltol = np.zeros((B.k, A.k), dtype=int)
        ctol = np.zeros((B.n, A.n - A.k), dtype=int)
        super().__init__(A, B, ltol, ctol)

class HighRateSurgeryPatch(ConePatch):
    """
    High-rate surgery (arXiv:2510.08523) — HGP 데이터 블록 + HGP 앤실라.

    데이터 블록을 **리스트** 로 받는다.  Cs = [C_1, ..., C_m].
        m = 1 : 단일 블록 surgery (논문 원본)
        m = 2 : 두 블록 merge, 즉 L^(1)·L^(2) 측정

    확정 컨벤션: HGPCode(Ca, Cb) = HGP(D=Ca, B=Cb),  ∂ = H.
        writeup 표기의 B 는 gamma_1 이 작용하는 코드, 즉 Cb 쪽이다.
        Gamma_1 은 x[q_t, :] 를 읽고 그 슬라이스는 Cb 좌표로 indexing 된다.
        q_t 는 Ca 의 information bit = writeup 의 D^A 쪽 index.

        A1 = Ca1_A ⊗ Cb1_A          ker ∂1^A = ker(Ha_A) ⊗ ker(Hb_A)
        측정 대상 = 데이터의 F-II logical  e_c^{Ca0_C} ⊗ ker(Hb_C)   (block1)

    chain map 은 column 별로 쪼개진다 (논문 Eq.11, D23):

        Γ1^(m) = Σ_t ( e_{c_t^(m)} e_{q_t}^T ) ⊗ γ1^(t,m)
        Γ0^(m) = Σ_t ( e_{c_t^(m)} e_{q_t}^T ) ⊗ γ0^(t,m)

    교환조건 Γ0 ∂1^A = ∂1^C Γ1 은 블록마다 독립적으로
        Hb_C^(m) γ1^(t,m) = γ0^(t,m) Hb_A
    로 환원되므로, 블록들은 서로 달라도 되고 공유하는 것은 A 뿐이다.

    측정되는 것:
        Γ1(ker ∂1^A) = span{ Σ_m e_{c_t^(m)} ⊗ γ1^(t,m)(v) : v ∈ ker Hb_A }
    m ≥ 2 이면 ker ∂1^A 의 원소 하나가 모든 블록에 동시에 상을 남기므로
    측정되는 연산자는 항상 블록별 X-logical 의 곱 꼴이다 (가정 A1).

    Parameters
    ----------
    Cs      : list[HGPCodePatch]  (또는 단일 HGPCodePatch)
    A       : HGPCodePatch        (patch_name 이 전부 달라야 함)
    columns : list of (ia, ics, gammas)
        ia     : A.css.Ca 의 logical index            -> 앤실라 column
        ics    : list[int], 길이 m. ics[j] 는 Cs[j].css.CaT 의 logical index
        gammas : list, 길이 m. Homomorphism(A.css.Cb, Cs[j].css.Cb) 또는 ltol 배열
        (m = 1 이면 ics, gammas 에 스칼라를 줘도 된다)

    Note
    ----
    d2 >= 18 (r_A >= 17) 이면 Homomorphism 의 ctol 자동탐색이 2^{r_A} 열거로
    죽는다. 그 경우 Homomorphism 을 직접 만들어 넘길 것:
        ctol = gamma1[:, A.css.Cb.complement()]
        h = Homomorphism(A.css.Cb, C.css.Cb, ltol, ctol=ctol)
    """

    def __init__(self, Cs, A, columns, name: str = "high-rate-surgery"):
        if isinstance(Cs, CodePatch):
            Cs = [Cs]
        Cs = list(Cs)
        names = [C._name for C in Cs] + [A._name]
        assert len(set(names)) == len(names), f"patch_name 이 겹치면 Qubit 이 충돌함: {names}"
        self.Cs, self.A = Cs, A
        m = len(Cs)
        cssA = A.css

        qA  = cssA.Ca.pivots()                       # 앤실라 column 후보
        cCs = [C.css.CaT.pivots() for C in Cs]       # 블록별 column 후보
        Hb_A = np.asarray(cssA.Cb.H, dtype=int)

        self.columns = []
        used_q = set()
        used_c = [set() for _ in range(m)]
        for ia, ics, gammas in columns:
            if np.isscalar(ics):    ics    = [ics]
            if not isinstance(gammas, (list, tuple)): gammas = [gammas]
            assert len(ics) == m and len(gammas) == m, \
                f"블록 {m} 개인데 ics={len(ics)}, gammas={len(gammas)}"
            assert 0 <= ia < cssA.Ca.k, f"ia={ia} 는 [0,{cssA.Ca.k}) 범위여야 함"
            assert ia not in used_q, f"앤실라 column {ia} 중복"
            used_q.add(ia)

            gs = []
            for j, (C, ic, gam) in enumerate(zip(Cs, ics, gammas)):
                assert 0 <= ic < C.css.CaT.k, \
                    f"ics[{j}]={ic} 는 [0,{C.css.CaT.k}) 범위여야 함"
                assert ic not in used_c[j], f"{C._name} column {ic} 중복"
                used_c[j].add(ic)

                if not isinstance(gam, Homomorphism):
                    gam = Homomorphism(cssA.Cb, C.css.Cb, gam)
                g1 = np.asarray(gam.gamma1, dtype=int) % 2
                g0 = np.asarray(gam.gamma0, dtype=int) % 2
                assert g1.shape == (C.css.nb, cssA.nb), \
                    f"{C._name}: gamma1 shape {g1.shape} != {(C.css.nb, cssA.nb)}"
                assert g0.shape == (C.css.rb, cssA.rb), \
                    f"{C._name}: gamma0 shape {g0.shape} != {(C.css.rb, cssA.rb)}"
                Hb_C = np.asarray(C.css.Cb.H, dtype=int)
                assert not ((Hb_C @ g1 - g0 @ Hb_A) % 2).any(), \
                    f"commuting diagram 실패 ({C._name}): Hb @ gamma1 != gamma0 @ Hb_A"
                gs.append((gam, g1, g0))

            self.columns.append((qA[ia], [cCs[j][ics[j]] for j in range(m)], gs))

        f1t = self._build_f1t()
        f0  = self._build_f0()
        super().__init__(Cs, A, f1t, f0, name=name)

    # -- column -> (블록, column index, g1, g0) 리스트 -------------------------
    def _legs(self, col):
        q_t, cts, gs = col
        return q_t, [(self.Cs[j], cts[j], gs[j][1], gs[j][2]) for j in range(len(self.Cs))]

    # -- Γ1 : 앤실라 X check -> 데이터 qubit -----------------------------------
    def _build_f1t(self):
        cssA = self.A.css
        f1t = [Check.get_empty_check()] * len(self.A.x_checks())
        for col in self.columns:
            q_t, legs = self._legs(col)
            for C, c_t, g1, _ in legs:
                cq, css = C.data_qubits(), C.css
                for j in range(cssA.nb):                      # Cb_A bit
                    sup = [cq[css.qubit1_idx(c_t, u)] for u in np.nonzero(g1[:, j])[0]]
                    if sup:
                        idx = cssA.x_check_idx(q_t, j)
                        f1t[idx] = f1t[idx] + Check(frozenset(sup), "X")
        return f1t

    # -- Γ0 : 데이터 Z check -> 앤실라 qubit -----------------------------------
    def _build_f0(self):
        """Z-check 순서는 C_1.z_checks() + ... + C_m.z_checks()."""
        cssA = self.A.css
        aq = self.A.data_qubits()
        offs, acc = [], 0
        for C in self.Cs:
            offs.append(acc); acc += len(C.z_checks())
        f0 = [Check.get_empty_check()] * acc
        for col in self.columns:
            q_t, legs = self._legs(col)
            for (C, c_t, _, g0), off in zip(legs, offs):
                css = C.css
                for s in range(css.rb):                       # Cb_C check
                    sup = [aq[cssA.qubit2_idx(q_t, sig)] for sig in np.nonzero(g0[s, :])[0]]
                    if sup:
                        f0[off + css.z_check_idx(c_t, s)] += Check(frozenset(sup), "Z")
        return f0

    # -- 측정되는 operator -----------------------------------------------------
    def measured_operators(self) -> np.ndarray:
        """Γ1(ker ∂1^A) 의 **생성원** 을 C_1 ⊕ ... ⊕ C_m qubit 위 벡터로 반환 (rows).
        basis 가 아니므로 rank 는 따로 재야 한다 (report 참고)."""
        cssA = self.A.css
        ns = [C.css.n for C in self.Cs]
        offs, acc = [], 0
        for n in ns:
            offs.append(acc); acc += n
        Ga = np.asarray(cssA.Ca.G, dtype=int)                 # ker(Ha_A)
        Gb = np.asarray(cssA.Cb.G, dtype=int)                 # ker(Hb_A)
        out = []
        for w in Ga:
            for v in Gb:
                vec = np.zeros(acc, dtype=int)
                for col in self.columns:
                    q_t, legs = self._legs(col)
                    if not w[q_t]:
                        continue
                    for (C, c_t, g1, _), off in zip(legs, offs):
                        for u in np.nonzero((g1 @ v) % 2)[0]:
                            vec[off + C.css.qubit1_idx(c_t, u)] ^= 1
                out.append(vec)
        return np.array(out, dtype=int).reshape(len(out), acc)

    def report(self):
        cssA = self.A.css
        F = self.Cs[0].css.F
        M = self.measured_operators()
        dim_M = int(np.linalg.matrix_rank(F(M % 2))) if M.size else 0
        k_data = sum(C.css.k for C in self.Cs)
        deg = max(int(g1.sum(axis=0).max())
                  for col in self.columns for _, _, g1, _ in self._legs(col)[1])
        blocks = " + ".join(f"[[{C.css.n},{C.css.k}]]" for C in self.Cs)
        print("=" * 66)
        print(f"{self._name}:  {blocks}  +  ancilla [[{cssA.n},{cssA.k}]]")
        print("=" * 66)
        print(f"  columns          : {len(self.columns)}")
        print(f"     앤실라 q      : {[c[0] for c in self.columns]}")
        for j, C in enumerate(self.Cs):
            print(f"     {C._name} c      : {[c[1][j] for c in self.columns]}")
        print(f"  dim ker ∂1^A     : {cssA.Ca.k * cssA.Cb.k}")
        print(f"  dim M = dim Γ1(ker ∂1^A) : {dim_M}")
        print(f"  |Γ1|             : {deg}   (transversal 이면 1)")
        print(f"  merged commutativity : {self.check_commutativity()}")
        print(f"  merged n / k     : {len(self.data_qubits())} / {self.get_k()}"
              f"   (기대값 k = {k_data} - {dim_M} + {cssA.k} = {k_data - dim_M + cssA.k})")

    # -- logical representatives ----------------------------------------------
    def _column_quotient_basis(self, g1s, max_enum: int = 1 << 20):
        """
        column tuple (c_t^(1), ..., c_t^(m)) 에서 살아남는 X-logical class 의 대표원.

            ( ⊕_j ker(Hb_C^(j)) ) / { (γ1^(1)v, ..., γ1^(m)v) : v ∈ ker Hb_A }

        길이 Σ_j nb_j 벡터로 반환 (블록 순서대로 이어붙임).
        coset leader (최소 weight) 를 weight 오름차순 greedy 로 고른다.
        """
        cssA = self.A.css
        Ga = np.asarray(cssA.Cb.G, dtype=int) % 2
        Gbs = [np.asarray(C.css.Cb.G, dtype=int) % 2 for C in self.Cs]
        kbs = [C.css.Cb.k for C in self.Cs]
        nbs = [C.css.nb for C in self.Cs]
        kb, nb = sum(kbs), sum(nbs)
        F = self.Cs[0].css.F
        rk = lambda M: 0 if np.size(M) == 0 else int(
            np.linalg.matrix_rank(F(np.asarray(M, dtype=int) % 2)))

        Gb = np.zeros((kb, nb), dtype=int)
        ro = co = 0
        for Gbj, kj, nj in zip(Gbs, kbs, nbs):
            Gb[ro:ro+kj, co:co+nj] = Gbj
            ro += kj; co += nj

        meas = np.array([np.concatenate([(g1 @ v) % 2 for g1 in g1s]) for v in Ga],
                        dtype=int).reshape(len(Ga), nb)
        d_meas = rk(meas)

        if (1 << kb) > max_enum or d_meas == 0:
            leaders = list(Gb)                                # fallback / 측정 없음
        else:
            span = [np.zeros(nb, dtype=int)]
            for mm in range(1, 1 << len(meas)):
                span.append(np.bitwise_xor.reduce(
                    [meas[i] for i in range(len(meas)) if mm >> i & 1], axis=0))
            span = np.array(span, dtype=int)
            seen = {}
            for mm in range(1 << kb):
                v = (np.bitwise_xor.reduce([Gb[i] for i in range(kb) if mm >> i & 1], axis=0)
                     if mm else np.zeros(nb, dtype=int))
                key = tuple(sorted(map(tuple, (v + span) % 2)))
                if key not in seen:
                    seen[key] = min(((v + s) % 2 for s in span), key=lambda x: int(x.sum()))
            leaders = sorted((r for r in seen.values() if r.any()), key=lambda x: int(x.sum()))

        basis = meas.copy()
        out = []
        for r in leaders:
            if rk(np.vstack([basis, r])) > rk(basis):
                basis = np.vstack([basis, r])
                out.append(r)
        assert len(out) == kb - d_meas, f"quotient dim {len(out)} != {kb - d_meas}"
        return out

    def init_logical_representatives(self, minimize: bool = True):
        """
        merged code 의 logical representative 설정.

          1. X : F-I (block2) 는 Γ1 이 안 건드리므로 모든 블록에서 그대로.
                 F-II (block1) 는 column tuple 별로, touched 는 quotient 의
                 coset leader, untouched 는 데이터 코드 것 그대로.
          2. 나머지 (앤실라 gauge logical 포함) 와 Z 전부는 set_missing_logicals().
          3. minimize=True 면 _minimize_logical_op 으로 weight 최소화.
             같은 타입 stabilizer 를 곱하는 것이라 symplectic pairing 은 보존된다.
        """
        if self.css is None:
            self.make_css()
        css = self.css
        N = css.n
        m = len(self.Cs)
        cssS = [C.css for C in self.Cs]
        offs, acc = [], 0
        for cs in cssS:
            offs.append(acc); acc += cs.n

        # touched column tuple 들
        tuples = {}                       # c_t^(0) -> (나머지 c 들, g1 리스트)
        touched = [set() for _ in range(m)]
        for col in self.columns:
            q_t, legs = self._legs(col)
            cts = [c_t for _, c_t, _, _ in legs]
            g1s = [g1  for _, _, g1, _ in legs]
            tuples[cts[0]] = (cts, g1s)
            for j, c in enumerate(cts):
                touched[j].add(c)

        xreps = []
        # F-I : block2 — 모든 블록 그대로
        for cs, off in zip(cssS, offs):
            Ga = np.asarray(cs.Ca.G, dtype=int) % 2
            pbT = cs.CbT.pivots()
            for p in range(cs.Ca.k):
                for q in range(cs.CbT.k):
                    v = np.zeros(N, dtype=int)
                    for i in np.nonzero(Ga[p])[0]:
                        v[off + cs.qubit2_idx(i, pbT[q])] = 1
                    xreps.append(v)

        # F-II : block1 — touched tuple 은 묶어서, 나머지는 블록별로
        nbs = [cs.nb for cs in cssS]
        for b in range(cssS[0].CaT.k):
            c0 = cssS[0].CaT.pivots()[b]
            if c0 not in tuples:                              # 블록 0 의 untouched
                for r in np.asarray(cssS[0].Cb.G, dtype=int) % 2:
                    v = np.zeros(N, dtype=int)
                    for u in np.nonzero(r)[0]:
                        v[offs[0] + cssS[0].qubit1_idx(c0, u)] = 1
                    xreps.append(v)
                continue
            cts, g1s = tuples[c0]
            for r in self._column_quotient_basis(g1s):
                v = np.zeros(N, dtype=int)
                lo = 0
                for j, nbj in enumerate(nbs):
                    for u in np.nonzero(r[lo:lo+nbj])[0]:
                        v[offs[j] + cssS[j].qubit1_idx(cts[j], u)] = 1
                    lo += nbj
                xreps.append(v)
        for j in range(1, m):                                 # 나머지 블록의 untouched
            for b in range(cssS[j].CaT.k):
                c = cssS[j].CaT.pivots()[b]
                if c in touched[j]:
                    continue
                for r in np.asarray(cssS[j].Cb.G, dtype=int) % 2:
                    v = np.zeros(N, dtype=int)
                    for u in np.nonzero(r)[0]:
                        v[offs[j] + cssS[j].qubit1_idx(c, u)] = 1
                    xreps.append(v)

        assert len(xreps) <= css.k, f"X reps {len(xreps)} > k_merged {css.k}"
        self.n_data_logicals = len(xreps)
        self.n_gauge_logicals = css.k - len(xreps)            # = k_A

        css.logical_x_repr = [None] * css.k
        css.logical_z_repr = [None] * css.k
        for i, v in enumerate(xreps):
            css.set_logical_representative("X", i, v)
        css.set_missing_logicals()

        if minimize:
            for i in range(css.k):
                css.logical_x_repr[i] = css._minimize_logical_op("X", css.logical_x_repr[i])
                css.logical_z_repr[i] = css._minimize_logical_op("Z", css.logical_z_repr[i])
        return self


def MergedHighRateSurgeryPatch(C1, C2, A, columns, name="merged-high-rate-surgery"):
    """두 블록 편의 래퍼.  columns 는 (ia, ic1, ic2, gamma1, gamma2) 형식."""
    cols = [(ia, [ic1, ic2], [g1, g2]) for ia, ic1, ic2, g1, g2 in columns]
    return HighRateSurgeryPatch([C1, C2], A, cols, name=name)


def build_xx_cone_surface(d: int) -> ConePatch:
    C1 = SurfaceCode(d, patch_name="surf1", offset=(0.0,0.0))
    C2 = SurfaceCode(d, patch_name="surf2", offset=(d,0.0))
    A  = EmptyPatch(patch_name="ancilla", offset=(d-0.5,0.5))

    for i in range(d-1):
        A.add_qubit(0, i)
    A.add_check([A.data_qubits()[0]], "X")
    for i in range(1,d-1):
        A.add_check([A.data_qubits()[i-1], A.data_qubits()[i]], "X")
    A.add_check([A.data_qubits()[d-2]], "X")

    f0 = [Check.get_empty_check()]*len(C1.z_checks()+C2.z_checks())
    for i in range(d-1):
        f0[i*d + (d-1)] = Check(frozenset([A.data_qubits()[i]]), "Z")
    for i in range(d-1):
        f0[len(C1.z_checks()) + i*d] = Check(frozenset([A.data_qubits()[i]]), "Z")
    
    f1t = [Check.get_empty_check()]*len(A.x_checks())
    for i in range(d):
        f1t[i] = Check(frozenset([C1._get_qubit_from_coord(d-1,i), C2._get_qubit_from_coord(0,i)]), "X")
    

    return ConePatch(C1, C2, A, f1t, f0, name="xx-cone-surface")


# ---------------------------------------------------------------------------
# ZZ surgery 용 ConePatch builder
# ---------------------------------------------------------------------------

def build_zz_cone(
    C1: CodePatch,
    C2: CodePatch,
    A: CodePatch,
    f1: np.ndarray,
    f0: np.ndarray,
    delta1: np.ndarray,
    delta0: np.ndarray | None = None,
) -> ConePatch:
    """
    f1, f0, delta1, delta0 를 직접 받아 ConePatch 생성.
    자동 계산이 필요하면 compute_zz_cone_maps() 사용.
    """
    return ConePatch(C1, C2, A, f1, f0, delta1, delta0, name="ZZ-cone")


def compute_zz_cone_maps(
    C1: CodePatch,
    C2: CodePatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ZZ surgery 에 필요한 f1, f0, delta1, delta0 자동 계산.
    논문 eq.35, 47, 48 → Algorithm 1 → Algorithm 2 순서.
    Returns: (f1, f0, delta1, delta0)
    """
    c1_qubits = C1.data_qubits()
    c2_qubits = C2.data_qubits()
    n1 = len(c1_qubits)
    n2 = len(c2_qubits)
    n  = n1 + n2

    reg1 = C1.make_registry()
    reg2 = C2.make_registry()

    Hz1 = C1.h_z(reg1)
    Hz2 = C2.h_z(reg2)
    nZ1, nZ2 = Hz1.shape[0], Hz2.shape[0]
    nZ = nZ1 + nZ2

    Hz = np.zeros((nZ, n), dtype=np.uint8)
    Hz[:nZ1, :n1] = Hz1
    Hz[nZ1:, n1:] = Hz2

    xl1 = C1.x_logicals()[0]
    xl2 = C2.x_logicals()[0]
    Q_indices = (
        sorted(reg1[q] for q in xl1) +
        sorted(n1 + reg2[q] for q in xl2)
    )
    w = len(Q_indices)

    # f1: eq.35
    f1 = np.zeros((n, w), dtype=np.uint8)
    for j, i in enumerate(Q_indices):
        f1[i, j] = 1

    # delta1*, f0*: eq.47, 48
    Hz_Q = Hz[:, Q_indices]
    nonzero_rows = [i for i in range(nZ) if Hz_Q[i].any()]
    n0Z = len(nonzero_rows)

    delta1 = Hz_Q[nonzero_rows, :]

    f0 = np.zeros((nZ, n0Z), dtype=np.uint8)
    for j, i in enumerate(nonzero_rows):
        f0[i, j] = 1

    # Algorithm 1
    delta1 = _algorithm1(delta1)

    n_new_edges = delta1.shape[0] - n0Z
    if n_new_edges > 0:
        f0 = np.hstack([f0, np.zeros((nZ, n_new_edges), dtype=np.uint8)])

    # Algorithm 2
    delta0 = _algorithm2(delta1, Hz, f0)

    return f1, f0, delta1, delta0


def _algorithm1(delta1: np.ndarray) -> np.ndarray:
    """Cheeger constant >= 1 이 될 때까지 edge 추가."""
    delta1 = delta1.copy()
    while True:
        h, S = _cheeger_constant(delta1)
        if h >= 1:
            break
        w = delta1.shape[1]
        Sc = [v for v in range(w) if v not in S]
        deg = delta1.sum(axis=0)
        best_h = -1
        best_edge = None
        for v1 in sorted(S, key=lambda v: deg[v]):
            for v2 in sorted(Sc, key=lambda v: deg[v]):
                new_row = np.zeros(w, dtype=np.uint8)
                new_row[v1] = 1
                new_row[v2] = 1
                candidate = np.vstack([delta1, new_row])
                h_new, _ = _cheeger_constant(candidate)
                if h_new > best_h:
                    best_h = h_new
                    best_edge = new_row
                    if h_new >= 1:
                        break
            if best_h >= 1:
                break
        delta1 = np.vstack([delta1, best_edge])
    return delta1


def _cheeger_constant(delta1: np.ndarray) -> tuple[float, list[int]]:
    from itertools import combinations
    w = delta1.shape[1]
    best_h = float('inf')
    best_S = []
    for size in range(1, w // 2 + 1):
        for S in combinations(range(w), size):
            S_set = set(S)
            boundary = 0
            for row in delta1:
                support = set(np.where(row)[0])
                if len(support & S_set) % 2 == 1:
                    boundary += 1
            h = boundary / len(S)
            if h < best_h:
                best_h = h
                best_S = list(S)
    return best_h, best_S


def _algorithm2(
    delta1: np.ndarray,
    Hz: np.ndarray,
    f0: np.ndarray,
) -> np.ndarray:
    """cycle basis (∂0) 계산. ker delta0 ≅ im delta1."""
    n0Z = delta1.shape[0]
    cycle_basis = _f2_null_space(delta1.T)
    if cycle_basis.shape[0] == 0:
        return np.zeros((0, n0Z), dtype=np.uint8)
    return cycle_basis.astype(np.uint8)


def _f2_null_space(M: np.ndarray) -> np.ndarray:
    """F2 위에서 M 의 null space basis (rows = basis vectors)."""
    M = M.copy().astype(np.uint8) % 2
    m, n = M.shape
    aug = np.hstack([M, np.eye(n, dtype=np.uint8)])
    pivot_cols = []
    row = 0
    for col in range(m):
        pivot = None
        for r in range(row, n):
            if aug[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] = (aug[r] + aug[row]) % 2
        pivot_cols.append(col)
        row += 1
    null_basis = aug[len(pivot_cols):, m:]
    return null_basis % 2