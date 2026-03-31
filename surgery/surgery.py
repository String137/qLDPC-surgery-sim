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
from core.primitives import Qubit, Check, QubitRegistry
from core.codes import CodePatch, RotatedSurfaceCode, TrivialCode


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


# ---------------------------------------------------------------------------
# MergedPatchWithAncilla
# ---------------------------------------------------------------------------

class MergedPatchWithAncilla(CodePatch):
    """
    Three-way merge: patch1 ↔ ancilla ↔ patch2
    """

    def __init__(
        self,
        patch1: CodePatch,
        ancilla: CodePatch,
        patch2: CodePatch,
        seam_x_checks: list[Check],
        seam_z_checks: list[Check],
        name: str = "merged-ancilla",
    ):
        self._name = name
        self.patch1 = patch1
        self.ancilla = ancilla
        self.patch2 = patch2

        self._x_checks = (
            patch1.x_checks() +
            seam_x_checks +
            ancilla.x_checks() +
            patch2.x_checks()
        )

        self._z_checks = (
            patch1.z_checks() +
            seam_z_checks +
            ancilla.z_checks() +
            patch2.z_checks()
        )

        self._seam_x_checks = seam_x_checks
        self._seam_z_checks = seam_z_checks

        n_patch1_x = len(patch1.x_checks())
        n_seam_x = len(seam_x_checks)
        self._seam_x_slice = slice(n_patch1_x, n_patch1_x + n_seam_x)

        n_patch1_z = len(patch1.z_checks())
        n_seam_z = len(seam_z_checks)
        self._seam_z_slice = slice(n_patch1_z, n_patch1_z + n_seam_z)

        self._x_logicals = patch1.x_logicals() + patch2.x_logicals()
        self._z_logicals = patch1.z_logicals() + patch2.z_logicals()

        self._registry = QubitRegistry()
        self._registry.register(patch1.data_qubits())
        self._registry.register(ancilla.data_qubits())
        self._registry.register(patch2.data_qubits())

    @property
    def name(self): return self._name

    def data_qubits(self) -> list[Qubit]:
        return (self.patch1.data_qubits() +
                self.ancilla.data_qubits() +
                self.patch2.data_qubits())

    def x_checks(self) -> list[Check]:  return self._x_checks
    def z_checks(self) -> list[Check]:  return self._z_checks
    def x_logicals(self):               return self._x_logicals
    def z_logicals(self):               return self._z_logicals

    def registry(self) -> QubitRegistry:
        return self._registry

    def seam_x_checks(self) -> list[Check]:
        return self._seam_x_checks

    def seam_z_checks(self) -> list[Check]:
        return self._seam_z_checks

    def make_registry(self) -> QubitRegistry:
        return self._registry


# ---------------------------------------------------------------------------
# ZZ / XX seam builder
# ---------------------------------------------------------------------------

def build_zz_merged(
    patch1: CodePatch,
    patch2: CodePatch,
    boundary_pairs: list[tuple[Qubit, Qubit]],
    seam_name: str = "seam",
) -> MergedPatch:
    seam_qubits = [Qubit(seam_name, i) for i in range(len(boundary_pairs))]

    seam_z_checks = [
        Check(
            frozenset([q1, seam_qubits[i], q2]),
            "Z",
            f"seam_Z_{i}",
        )
        for i, (q1, q2) in enumerate(boundary_pairs)
    ]

    return MergedPatch(
        patch1=patch1,
        patch2=patch2,
        seam_data_qubits=seam_qubits,
        seam_x_checks=[],
        seam_z_checks=seam_z_checks,
        name="ZZ-merged",
    )


def build_xx_merged(
    patch1: CodePatch,
    patch2: CodePatch,
    boundary_pairs: list[tuple[Qubit, Qubit]],
    seam_name: str = "seam",
) -> MergedPatch:
    seam_qubits = [Qubit(seam_name, i) for i in range(len(boundary_pairs))]

    seam_x_checks = [
        Check(
            frozenset([q1, seam_qubits[i], q2]),
            "X",
            f"seam_X_{i}",
        )
        for i, (q1, q2) in enumerate(boundary_pairs)
    ]

    return MergedPatch(
        patch1=patch1,
        patch2=patch2,
        seam_data_qubits=seam_qubits,
        seam_x_checks=seam_x_checks,
        seam_z_checks=[],
        name="XX-merged",
    )


# ---------------------------------------------------------------------------
# Surface code seam helper
# ---------------------------------------------------------------------------

def surface_zz_boundary_pairs(
    patch1: RotatedSurfaceCode,
    patch2: RotatedSurfaceCode,
) -> list[tuple[Qubit, Qubit]]:
    assert patch1.d == patch2.d
    b1 = patch1.boundary_qubits("left")
    b2 = patch2.boundary_qubits("left")
    return list(zip(b1, b2))


def surface_xx_boundary_pairs(
    patch1: RotatedSurfaceCode,
    patch2: RotatedSurfaceCode,
) -> list[tuple[Qubit, Qubit]]:
    assert patch1.d == patch2.d
    b1 = patch1.boundary_qubits("top")
    b2 = patch2.boundary_qubits("top")
    return list(zip(b1, b2))


# ---------------------------------------------------------------------------
# Merge with Ancilla Code (TrivialCode)
# ---------------------------------------------------------------------------

def build_zz_merged_with_ancilla(
    patch1: CodePatch,
    ancilla: CodePatch,
    patch2: CodePatch,
    boundary_pairs_1to2: list[tuple[Qubit, Qubit]],
) -> MergedPatchWithAncilla:
    anc_qubits = ancilla.data_qubits()
    n_seam = len(boundary_pairs_1to2)

    assert len(anc_qubits) == n_seam

    return MergedPatchWithAncilla(
        patch1=patch1,
        ancilla=ancilla,
        patch2=patch2,
        seam_x_checks=[],
        seam_z_checks=[],
        name="ZZ-merged-ancilla",
    )


def build_xx_merged_with_ancilla(
    patch1: CodePatch,
    ancilla: CodePatch,
    patch2: CodePatch,
    boundary_pairs_1to2: list[tuple[Qubit, Qubit]],
) -> MergedPatchWithAncilla:
    anc_qubits = ancilla.data_qubits()
    n_seam = len(boundary_pairs_1to2)

    assert len(anc_qubits) == n_seam

    return MergedPatchWithAncilla(
        patch1=patch1,
        ancilla=ancilla,
        patch2=patch2,
        seam_x_checks=[],
        seam_z_checks=[],
        name="XX-merged-ancilla",
    )


def surface_zz_boundary_pairs_with_ancilla(
    patch1: RotatedSurfaceCode,
    patch2: RotatedSurfaceCode,
) -> list[tuple[Qubit, Qubit]]:
    assert patch1.d == patch2.d
    b1 = patch1.boundary_qubits("right")
    b2 = patch2.boundary_qubits("left")
    return list(zip(b1, b2))


def surface_xx_boundary_pairs_with_ancilla(
    patch1: RotatedSurfaceCode,
    patch2: RotatedSurfaceCode,
) -> list[tuple[Qubit, Qubit]]:
    assert patch1.d == patch2.d
    b1 = patch1.boundary_qubits("top")
    b2 = patch2.boundary_qubits("top")
    return list(zip(b1, b2))


# ---------------------------------------------------------------------------
# Surgery circuit (stim)
# ---------------------------------------------------------------------------

def build_surgery_circuit(
    patch1: RotatedSurfaceCode,
    patch2: RotatedSurfaceCode,
    *,
    rounds: int,
    p_data: float,
    p_meas: float,
    basis: str = "ZZ",
):
    import stim

    if basis == "ZZ":
        pairs = surface_zz_boundary_pairs(patch1, patch2)
        merged = build_zz_merged(patch1, patch2, pairs)
        meas_basis = "Z"
    elif basis == "XX":
        pairs = surface_xx_boundary_pairs(patch1, patch2)
        merged = build_xx_merged(patch1, patch2, pairs)
        meas_basis = "X"
    else:
        raise ValueError(f"Unknown basis: {basis}")

    reg = merged.registry()
    n_total = len(reg)

    circuit = stim.Circuit()
    circuit.append("R", list(range(n_total)))

    circuit += _measure_checks_rounds(
        checks=patch1.z_checks() if meas_basis == "Z" else patch1.x_checks(),
        reg=reg,
        rounds=rounds,
        p_data=p_data,
        p_meas=p_meas,
        pauli=meas_basis,
        data_qubit_subset=patch1.data_qubits(),
        phase_t_offset=0,
    )
    circuit += _measure_checks_rounds(
        checks=patch2.z_checks() if meas_basis == "Z" else patch2.x_checks(),
        reg=reg,
        rounds=rounds,
        p_data=p_data,
        p_meas=p_meas,
        pauli=meas_basis,
        data_qubit_subset=patch2.data_qubits(),
        phase_t_offset=0,
    )

    merged_checks = (merged.z_checks() if meas_basis == "Z"
                     else merged.x_checks())
    circuit += _measure_checks_rounds(
        checks=merged_checks,
        reg=reg,
        rounds=rounds,
        p_data=p_data,
        p_meas=p_meas,
        pauli=meas_basis,
        data_qubit_subset=merged.data_qubits(),
        phase_t_offset=rounds,
    )

    data_indices = [reg[q] for q in merged.data_qubits()]
    circuit.append("M", data_indices)

    lz1 = merged.patch1.z_logicals()[0]
    lz2 = merged.patch2.z_logicals()[0]

    n_p1z = len(merged.patch1.z_checks())
    n_seam_z = len(merged.seam_z_checks())
    n_merged_z = len(merged.z_checks())
    seam_zz_targets = [
        stim.target_rec(-(n_total + n_merged_z - (n_p1z + i)))
        for i in range(n_seam_z)
    ]
    circuit.append("OBSERVABLE_INCLUDE", seam_zz_targets, 0)

    idx1 = sorted(reg[q] for q in lz1)
    targets_z1 = [stim.target_rec(-(n_total - i)) for i in idx1]
    circuit.append("OBSERVABLE_INCLUDE", targets_z1, 1)

    idx2 = sorted(reg[q] for q in lz2)
    targets_z2 = [stim.target_rec(-(n_total - i)) for i in idx2]
    circuit.append("OBSERVABLE_INCLUDE", targets_z2, 2)

    return circuit


def _measure_checks_rounds(
    checks: list[Check],
    reg: QubitRegistry,
    rounds: int,
    p_data: float,
    p_meas: float,
    pauli: str,
    data_qubit_subset: list[Qubit],
    phase_t_offset: int,
):
    import stim

    circuit = stim.Circuit()
    n_global = len(reg)
    data_indices = [reg[q] for q in data_qubit_subset]

    anc_start = n_global
    n_chks = len(checks)

    for t in range(rounds):
        if p_data > 0:
            circuit.append("X_ERROR", data_indices, p_data)
            circuit.append("Z_ERROR", data_indices, p_data)

        anc = list(range(anc_start, anc_start + n_chks))
        circuit.append("R", anc)

        for ai, chk in enumerate(checks):
            a = anc[ai]
            q_indices = sorted(reg.check_to_indices(chk))
            if pauli == "X":
                circuit.append("H", [a])
                for q in q_indices:
                    circuit.append("CNOT", [a, q])
                circuit.append("H", [a])
            else:
                for q in q_indices:
                    circuit.append("CNOT", [q, a])

        if p_meas > 0:
            circuit.append("X_ERROR", anc, p_meas)
        circuit.append("M", anc)

        for ai in range(n_chks):
            rec_cur = stim.target_rec(-(n_chks - ai))
            t_abs = phase_t_offset + t
            if t_abs == 0:
                circuit.append("DETECTOR", [rec_cur], [ai, 0, t_abs])
            else:
                rec_prev = stim.target_rec(-(n_chks - ai) - n_chks)
                circuit.append("DETECTOR",
                                [rec_cur, rec_prev], [ai, 0, t_abs])

    return circuit


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
        A        : ancilla code (data qubits 만 가짐)
        f1       : np.ndarray, shape (n, w), F2  — eq.35
        f0       : np.ndarray, shape (nZ, n0Z), F2  — eq.48
                   commutativity: H_Z @ f1 = f0 @ partial1  (mod 2)
        partial1t   : np.ndarray, shape (w, n0Z), F2  — eq.47 + Algorithm 1
        partial0   : np.ndarray, shape (n_cyc, n0Z), F2 or None  — Algorithm 2
    """

    def __init__(
        self,
        C1: CodePatch,
        C2: CodePatch,
        A: CodePatch,
        f1: np.ndarray,
        f0: np.ndarray,
        partial1t: np.ndarray,
        partial0: np.ndarray | None = None,
        name: str = "cone",
    ):
        self._name = name
        self.C1 = C1
        self.C2 = C2
        self.A  = A

        direct_sum = DirectSum(C1, C2)
        c_qubits = direct_sum.data_qubits()

        a_qubits = A.data_qubits()
        self._data_qubits = c_qubits + a_qubits

        self._registry = QubitRegistry()
        self._registry.register(c_qubits)
        self._registry.register(a_qubits)

        w = len(a_qubits)

        # TODO: shape / commutativity 검증
        # assert f1.shape == (len(c_qubits), w)
        # assert (H_Z @ f1 % 2 == f0 @ delta1 % 2).all()
        # assert delta0 is None or (delta0 @ delta1 % 2 == 0).all()

        # X-checks
        # (1) 기존 C1, C2 X-checks
        # TODO: ZZ surgery 에서 seam 쪽 boundary X-check (weight-2) 제거 필요
        existing_x = C1.x_checks() + C2.x_checks()

        # (2) 새 X-checks: partial1t 의 각 row s
        #     C 쪽: f1 @ partial1t[s,:] nonzero 위치
        #     A 쪽: partial1t[s,:] nonzero 위치
        new_x_checks = []
        for s in range(partial1t.shape[0]):
            c_side = (f1 @ partial1t[s, :] % 2).astype(bool)
            a_side = partial1t[s, :].astype(bool)
            qubits = frozenset(
                [c_qubits[i] for i in np.where(c_side)[0]] +
                [a_qubits[k] for k in np.where(a_side)[0]]
            )
            new_x_checks.append(Check(qubits, "X", f"cone_X_{s}"))

        self._x_checks = existing_x + new_x_checks

        # Z-checks
        # (1) 기존 C Z-checks: f0 로 ancilla qubit 추가
        c_z_checks = C1.z_checks() + C2.z_checks()
        existing_z = []
        for i, chk in enumerate(c_z_checks):
            extra = frozenset(a_qubits[k] for k in np.where(f0[i, :].astype(bool))[0])
            existing_z.append(Check(chk.qubits | extra, "Z", chk.label))

        # (2) 새 Z-checks: partial0 의 각 row (A 쪽만)
        new_z_checks = []
        if partial0 is not None:
            for c in range(partial0.shape[0]):
                a_side = partial0[c, :].astype(bool)
                qubits = frozenset(a_qubits[k] for k in np.where(a_side)[0])
                new_z_checks.append(Check(qubits, "Z", f"cone_Z_{c}"))

        self._z_checks = existing_z + new_z_checks

        self._x_logicals = C1.x_logicals() + C2.x_logicals()
        self._z_logicals = C1.z_logicals() + C2.z_logicals()

    @property
    def name(self) -> str: return self._name

    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self)   -> list[Check]:  return self._x_checks
    def z_checks(self)   -> list[Check]:  return self._z_checks
    def x_logicals(self):                 return self._x_logicals
    def z_logicals(self):                 return self._z_logicals
    def make_registry(self) -> QubitRegistry: return self._registry

    def h_x_tilde(self) -> np.ndarray:
        """H̃_X = [H_X | 0 ; f1^T | ∂1^T]"""
        # TODO
        # 굳이 구현 안해도 될 듯. x_checks()에서 이미 implicit하게 표현되어 있음.
        raise NotImplementedError

    def h_z_tilde(self) -> np.ndarray:
        """H̃_Z = [H_Z | f0 ; ∂0 | 0]"""
        # TODO
        raise NotImplementedError


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