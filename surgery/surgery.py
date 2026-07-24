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

class Homomorphism:
    def __init__(self, A, B, ltol, ctol):
        self.A, self.B = A, B
        self.F = A.F
        self.ltol = self.F(np.asarray(ltol, dtype=int))
        self.ctol = self.F(np.asarray(ctol, dtype=int))
        self.rA = A.n - A.k
        self._validate()
        self.gamma1 = None
        self.gamma0 = None
 
    def _validate(self):
        assert self.ltol.shape == (self.B.k, self.A.k), \
            f"ltol must be (k_B,k_A)=({self.B.k},{self.A.k}), got {self.ltol.shape}"
        assert self.ctol.shape == (self.B.n, self.rA), \
            f"ctol must be (n_B,rank H_A)=({self.B.n},{self.rA}), got {self.ctol.shape}"
 
    @staticmethod
    def _pivots(M):
        Mi = np.asarray(M, dtype=int)
        return [int(np.nonzero(r)[0][0]) for r in Mi if r.any()]
 
    def _basis_change(self):
        A, F = self.A, self.F
        piv = set(self._pivots(A.G))
        free = [j for j in range(A.n) if j not in piv]
        assert len(free) == self.rA
        E = F(np.zeros((A.n, self.rA), dtype=int))
        for c, j in enumerate(free):
            E[j, c] = 1
        return F(np.hstack((np.asarray(A.G.T, dtype=int), np.asarray(E, dtype=int))))
 
    def set_map(self):
        A, B, F = self.A, self.B, self.F
        left = B.G.T @ self.ltol
        M = F(np.hstack((np.asarray(left, dtype=int),
                         np.asarray(self.ctol, dtype=int))))
        P = self._basis_change()
        self.gamma1 = M @ np.linalg.inv(P)
        return self.gamma1
 
    @staticmethod
    def _solve_left(H, Y, F):
        """X with X @ H = Y over GF(2). Handles rank-deficient / redundant H."""
        m, n = H.shape
        HT = np.asarray(H.T, dtype=int)
        YT = np.asarray(Y.T, dtype=int)
        R = np.asarray(F(np.hstack((HT, YT))).row_reduce(), dtype=int)
        for r in R:
            if not r[:m].any() and r[m:].any():
                return None                      # unsolvable
        XT = F(np.zeros((m, YT.shape[1]), dtype=int))
        for r in R:
            nz = np.nonzero(r[:m])[0]
            if len(nz):
                XT[nz[0], :] = F(r[m:])          # free vars -> 0
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


class HighRateSurgeryPatch(_ConePatch):
    """
    ### Input
    Data Code C: possess a generalized parallel structure
    Ancilla Code A = HGP(B,D), where B and D are classical linear codes
    어려우니까 일단 C도 HGP라고 가정하자.
    ### Description
    This framework measures a set of logical operators $M$, where $M=\bigcup_i M_i$.
    For i\in[k_D], 
        \gamma_1^i(\ker \partial^B) = span\{M_i\} \in \ker \partial^C.
    
        Since \gamma_1 maps \ker \partial^B into \ker \partial^C, there exists \gamma_0 that makes the "classical" diagram commutes

        $\partial_1^C\gamma_1^i=\gamma_0^i\partial^B$

    From "classical homomorphisms" $\gamma_0^i$ and $\gamma_1^i$, we can construct the chain map as following.

        $$\Gamma_{0,1} = \sum\limits_{i\le k_D} \gamma_{0,1}^i\otimes (e_i^D)^T.$$

    ### Constructing $\gamma_0$ from $\gamma_1$
        $\partial_1^C\gamma_1^i=\gamma_0^i\partial^B$
        $\partial_1^B$ is a $r\times n$ matrix and full row rank.
        Therefore, there exists an **right inverse** $R$ of $\partial_1^B$, i.e. $\partial_1^B R = I_r$.
        Then, $\gamma_0^i=\partial_1^C\gamma_1^i R$.

    ### When the data code is HGP
    If the data code $C$ is also a HGP Code, say C=HGP(B',F),
        $\partial_1^B'\gamma_1^i=\gamma_0^i\partial^B$
        - \gamma_1 maps \ker \partial^B into \ker \partial^B
    """
    def __init__(
            self,
            C: HGPCodePatch,
            A: HGPCodePatch
        ):
        print(C.x_checks())
        self.f1t = None
        self.f0 = None
        self.gamma0 = []
        self.gamma1 = []    # A.Ca -> B.Ca로 가는 homomorphism들의 list
        # A.Ca의 logical을 B.Ca의 logical(trivial포함)으로 매핑하면 됨.
        # [1 0 0 0 1]과 같이 A.Ca의 logical이 주어지면 HGPCodePatch에서의 Check객체들을 생성해야함. 
        # i in [k_D]에 대해 gamma1^i, gamma0^i를 정의해야함.
        super().__init__(C, A, self.f1t, self.f0, name="high-rate-surgery")

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