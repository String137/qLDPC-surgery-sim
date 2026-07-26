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

    partial1대신 transpose된 partial1t를 쓸거임. shape도 고쳐주길 바람.
    입력:
        C1, C2   : 두 코드 패치. C = C1 ⊕ C2 로 취급.
        A        : ancilla code 
        f1       : np.ndarray, shape (n, w), F2  — eq.35
        f0       : np.ndarray, shape (nZ, n0Z), F2  — eq.48
                   commutativity: H_Z @ f1 = f0 @ partial1  (mod 2)
        partial1t   : np.ndarray, shape (w, n0Z), F2  — eq.47 + Algorithm 1
        partial0   : np.ndarray, shape (n_cyc, n0Z), F2 or None  — Algorithm 2

    f0의 i번째 행은 기존 code의 i번째 Z check에 추가될 ancilla qubit들의 조합. Check(Z) 객체로 들어옴.
    f1t의 i번째 행은 ancilla code의 i번째 X check에 추가될 기존 data qubit들의 조합. Check(X) 객체로 들어옴.
    """

    def __init__(
        self,
        C1: CodePatch,
        C2: CodePatch,
        A: CodePatch,
        f1t: list[Check],
        f0: list[Check],
        name: str = "cone",
    ):
        self._name = name
        self.C1 = C1
        self.C2 = C2
        self.A  = A
        self._f0 = f0
        self._f1t = f1t

        direct_sum = DirectSum(C1, C2)
        c_qubits = direct_sum.data_qubits()

        a_qubits = A.data_qubits()
        self._data_qubits = c_qubits + a_qubits

        self._registry = QubitRegistry()
        self._registry.register(c_qubits)
        self._registry.register(a_qubits)

        self.patch_name_dict = {
            C1._name: C1,
            C2._name: C2,
            A._name: A
        }

        partial0 = A.z_checks() if A.z_checks() else []
        partial1t = A.x_checks() if A.x_checks() else []

        assert len(direct_sum.z_checks()) == len(f0)
        self._z_checks = direct_sum.z_checks()
        for i in range(len(f0)):
            if f0[i].weight() > 0:
                self._z_checks[i] += f0[i]
        for i in range(len(partial0)):
            self._z_checks.append(partial0[i])

        assert len(partial1t) == len(f1t)
        self._x_checks = direct_sum.x_checks()
        for i in range(len(f1t)):
            self._x_checks.append(f1t[i] + partial1t[i])

        

    @property
    def name(self) -> str: return self._name

    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self)   -> list[Check]:  return self._x_checks
    def z_checks(self)   -> list[Check]:  return self._z_checks
    def x_logicals(self):                 return self._x_logicals
    def z_logicals(self):                 return self._z_logicals
    def make_registry(self) -> QubitRegistry: return self._registry

    def local_coord(self, q: Qubit) -> tuple[float, float]:
        # ConePatch에서의 local_coord
        lx, ly = self.patch_name_dict[q.patch_name].local_coord(q)
        ox, oy = self.patch_name_dict[q.patch_name]._offset
        return lx + ox, ly + oy
    
    def get_idx(self, q: Qubit) -> int:
        return self._registry[q]

    def h_x_tilde(self) -> np.ndarray:
        """H̃_X = [H_X | 0 ; f1^T | ∂1^T]"""
        # TODO
        # 굳이 구현 안해도 될 듯. x_checks()에서 이미 implicit하게 표현되어 있음.
        raise NotImplementedError

    def h_z_tilde(self) -> np.ndarray:
        """H̃_Z = [H_Z | f0 ; ∂0 | 0]"""
        # TODO
        raise NotImplementedError


class _ConePatch(CodePatch):
    """
    ConePatch의 C 하나 있는 버전.
    사실 이게 원래 ConePatch여야하는게 맞음.
    # TODO: 언젠가 _ConePatch를 ConePatch로 바꾸고, ConePatch는 ConePatch2같은 다른 이름으로 바꾸자.
    """

    def __init__(
        self,
        C: CodePatch,
        A: CodePatch,
        f1t: list[Check],
        f0: list[Check],
        name: str = "cone",
    ):
        self._name = name
        self.C = C
        self.A  = A
        self._f0 = f0
        self._f1t = f1t

        c_qubits = self.C.data_qubits()

        a_qubits = A.data_qubits()
        self._data_qubits = c_qubits + a_qubits

        self._registry = QubitRegistry()
        self._registry.register(c_qubits)
        self._registry.register(a_qubits)

        self.patch_name_dict = {
            C._name: C,
            A._name: A
        }

        partial0 = A.z_checks() if A.z_checks() else []
        partial1t = A.x_checks() if A.x_checks() else []

        assert len(C.z_checks()) == len(f0)
        self._z_checks = C.z_checks()
        for i in range(len(f0)):
            if f0[i].weight() > 0:
                self._z_checks[i] += f0[i]
        for i in range(len(partial0)):
            self._z_checks.append(partial0[i])

        assert len(partial1t) == len(f1t)
        self._x_checks = C.x_checks()
        for i in range(len(f1t)):
            self._x_checks.append(f1t[i] + partial1t[i])

        self.css = None
        

    @property
    def name(self) -> str: return self._name

    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self)   -> list[Check]:  return self._x_checks
    def z_checks(self)   -> list[Check]:  return self._z_checks
    def x_logicals(self):                 return self._x_logicals
    def z_logicals(self):                 return self._z_logicals
    def make_registry(self) -> QubitRegistry: return self._registry

    def local_coord(self, q: Qubit) -> tuple[float, float]:
        # ConePatch에서의 local_coord
        lx, ly = self.patch_name_dict[q.patch_name].local_coord(q)
        ox, oy = self.patch_name_dict[q.patch_name]._offset
        return lx + ox, ly + oy
    
    def get_idx(self, q: Qubit) -> int:
        return self._registry[q]

    def h_x_tilde(self) -> np.ndarray:
        """H̃_X = [H_X | 0 ; f1^T | ∂1^T]"""
        # TODO
        # 굳이 구현 안해도 될 듯. x_checks()에서 이미 implicit하게 표현되어 있음.
        raise NotImplementedError

    def h_z_tilde(self) -> np.ndarray:
        """H̃_Z = [H_Z | f0 ; ∂0 | 0]"""
        # TODO
        raise NotImplementedError



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
    def x_logicals(self):                 return self._x_logicals
    def z_logicals(self):                 return self._z_logicals
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

class HighRateSurgeryPatch(_ConePatch):
    """
    High-rate surgery (arXiv:2510.08523) — HGP 데이터 코드 + HGP 앤실라.
 
    확정 컨벤션: HGPCode(Ca, Cb) = HGP(B=Ca, D=Cb),  ∂ = H.
 
        A1 = Ca1_A (x) Cb1_A        ker ∂1^A = ker(Ha_A) (x) ker(Hb_A)
        측정 대상 = 데이터의 F-II logical  e_c^{Ca0_C} (x) ker(Hb_C)   (block1)
 
    chain map 은 column 별로 쪼개진다 (논문 Eq.11, D23):
 
        Γ1 = Σ_t ( e_{c_t} e_{q_t}^T ) (x) γ1^t   :  A1        -> C block1
        Γ0 = Σ_t ( e_{c_t} e_{q_t}^T ) (x) γ0^t   :  A block2  -> C0
 
        q_t : 앤실라 column = Ca_A 의 information bit
        c_t : 데이터 column = CaT_C 의 information bit
        γ^t : Cb_A -> Cb_C  classical homomorphism
 
    교환조건 Γ0 ∂1^A = ∂1^C Γ1 은 t 마다 Hb_C γ1^t = γ0^t Hb_A 로 환원된다.
    측정되는 것은 Γ1(ker ∂1^A) = span{ e_{c_t} (x) γ1^t(v) : v ∈ ker Hb_A }.
 
    Parameters
    ----------
    C, A : HGPCodePatch
    columns : list of (ia, ic, gamma)
        ia    : A.css.Ca  의 logical index   -> 앤실라 column
        ic    : C.css.CaT 의 logical index   -> 데이터 column
        gamma : Homomorphism(A.css.Cb, C.css.Cb, ...) 또는 ltol 배열
                (Homomorphism 을 직접 주면 ctol 자동탐색을 건너뛸 수 있음)
    """
 
    def __init__(self, C, A, columns, name: str = "high-rate-surgery"):
        assert C._name != A._name, "C 와 A 의 patch_name 이 같으면 Qubit 이 충돌함"
        self.C, self.A = C, A
        cssC, cssA = C.css, A.css
 
        qA = cssA.Ca.pivots()     # 앤실라 column 후보 (Ca_A 의 information bit)
        cC = cssC.CaT.pivots()    # 데이터 column 후보 (CaT_C 의 information bit)
 
        self.columns = []
        used_q, used_c = set(), set()
        for ia, ic, gamma in columns:
            assert 0 <= ia < cssA.Ca.k,  f"ia={ia} 는 [0,{cssA.Ca.k}) 범위여야 함"
            assert 0 <= ic < cssC.CaT.k, f"ic={ic} 는 [0,{cssC.CaT.k}) 범위여야 함"
            assert ia not in used_q, f"앤실라 column {ia} 중복"
            assert ic not in used_c, f"데이터 column {ic} 중복"
            used_q.add(ia); used_c.add(ic)
 
            if not isinstance(gamma, Homomorphism):
                gamma = Homomorphism(cssA.Cb, cssC.Cb, gamma)
            g1 = np.asarray(gamma.gamma1, dtype=int) % 2
            g0 = np.asarray(gamma.gamma0, dtype=int) % 2
            assert g1.shape == (cssC.nb, cssA.nb), f"gamma1 shape {g1.shape}"
            assert g0.shape == (cssC.rb, cssA.rb), f"gamma0 shape {g0.shape}"
            Hb_A = np.asarray(cssA.Cb.H, dtype=int)
            Hb_C = np.asarray(cssC.Cb.H, dtype=int)
            assert not ((Hb_C @ g1 - g0 @ Hb_A) % 2).any(), \
                "commuting diagram 실패: Hb_C @ gamma1 != gamma0 @ Hb_A"
 
            self.columns.append((qA[ia], cC[ic], gamma, g1, g0))
 
        f1t = self._build_f1t()
        f0  = self._build_f0()
        super().__init__(C, A, f1t, f0, name=name)
 
    # -- Γ1 : 앤실라 X check -> 데이터 qubit -------------------------------
    def _build_f1t(self):
        cssC, cssA = self.C.css, self.A.css
        cq = self.C.data_qubits()
        f1t = [Check.get_empty_check()] * len(self.A.x_checks())
        for q_t, c_t, _, g1, _ in self.columns:
            for j in range(cssA.nb):                       # Cb_A bit
                sup = [cq[cssC.qubit1_idx(c_t, u)]
                       for u in np.nonzero(g1[:, j])[0]]
                if sup:
                    idx = cssA.x_check_idx(q_t, j)
                    f1t[idx] = f1t[idx] + Check(frozenset(sup), "X")
        return f1t
 
    # -- Γ0 : 데이터 Z check -> 앤실라 qubit -------------------------------
    def _build_f0(self):
        cssC, cssA = self.C.css, self.A.css
        aq = self.A.data_qubits()
        f0 = [Check.get_empty_check()] * len(self.C.z_checks())
        for q_t, c_t, _, _, g0 in self.columns:
            for s in range(cssC.rb):                       # Cb_C check
                sup = [aq[cssA.qubit2_idx(q_t, sig)]
                       for sig in np.nonzero(g0[s, :])[0]]
                if sup:
                    idx = cssC.z_check_idx(c_t, s)
                    f0[idx] = f0[idx] + Check(frozenset(sup), "Z")
        return f0
 
    # -- 측정되는 operator ---------------------------------------------------
    def measured_operators(self) -> np.ndarray:
        """Γ1(ker ∂1^A) 의 생성원들을 데이터 코드 qubit 위 벡터로 반환. (rows)"""
        cssC, cssA = self.C.css, self.A.css
        Ga = np.asarray(cssA.Ca.G, dtype=int)              # ker(Ha_A)
        Gb = np.asarray(cssA.Cb.G, dtype=int)              # ker(Hb_A)
        out = []
        for w in Ga:
            for v in Gb:
                vec = np.zeros(cssC.n, dtype=int)
                for q_t, c_t, _, g1, _ in self.columns:
                    if w[q_t]:
                        img = (g1 @ v) % 2
                        for u in np.nonzero(img)[0]:
                            vec[cssC.qubit1_idx(c_t, u)] ^= 1
                out.append(vec)
        return np.array(out, dtype=int).reshape(len(out), cssC.n)
 
    def report(self):
        cssC, cssA = self.C.css, self.A.css
        F = cssC.F
        M = self.measured_operators()
        dim_M = int(np.linalg.matrix_rank(F(M % 2))) if M.size else 0
        print("=" * 62)
        print(f"{self._name}:  data [[{cssC.n},{cssC.k}]]  +  ancilla [[{cssA.n},{cssA.k}]]")
        print("=" * 62)
        print(f"  columns          : {len(self.columns)}   "
              f"(앤실라 q={[c[0] for c in self.columns]}, 데이터 c={[c[1] for c in self.columns]})")
        print(f"  dim ker ∂1^A     : {cssA.Ca.k * cssA.Cb.k}")
        print(f"  dim M = dim Γ1(ker ∂1^A) : {dim_M}")
        deg = max(max(int(g.sum(axis=0).max()), int(g.sum(axis=1).max()))
                  for *_, g, _ in self.columns)
        print(f"  |Γ1|             : {deg}   (transversal 이면 1)")
        print(f"  merged commutativity : {self.check_commutativity()}")
        print(f"  merged n / k     : {len(self.data_qubits())} / {self.get_k()}"
              f"   (기대값 k = {cssC.k} - {dim_M} + {cssA.k} = {cssC.k - dim_M + cssA.k})")
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