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
        seam: CodePatch,      # Qubit("seam", i)
        seam_x_checks: list[Check],
        seam_z_checks: list[Check],
        name: str = "merged",
    ):
        self._name = name
        self.patch1 = patch1
        self.patch2 = patch2
        self._seam_qubits = seam.data_qubits()

        # Merged check lists (patch1 + seam + patch2 순서)

        merged_x_checks_from_patch1 = []
        for chk in patch1.x_checks():
            if chk.label.split("_")[1] == "right":
                continue
            merged_x_checks_from_patch1.append(chk)
        merged_x_checks_from_patch2 = []
        for chk in patch2.x_checks():
            if chk.label.split("_")[1] == "left":
                continue
            merged_x_checks_from_patch2.append(chk)
        self._x_checks = (
            merged_x_checks_from_patch1 +
            seam_x_checks +
            merged_x_checks_from_patch2
        )
        self._z_checks = (
            patch1.z_checks() +
            seam_z_checks +
            patch2.z_checks()
        )

        # Seam check 위치 기록 (index range in merged list)
        self._seam_x_slice = slice(
            len(patch1.x_checks()),
            len(patch1.x_checks()) + len(seam_x_checks)
        )
        self._seam_z_slice = slice(
            len(patch1.z_checks()),
            len(patch1.z_checks()) + len(seam_z_checks)
        )

        # Logicals: 두 패치 각자의 것 유지
        self._x_logicals = patch1.x_logicals() + patch2.x_logicals()
        self._z_logicals = patch1.z_logicals() + patch2.z_logicals()

        # Global registry: patch1 → seam → patch2 순서
        self._registry = QubitRegistry()
        self._registry.register(patch1.data_qubits())
        self._registry.register(seam.data_qubits())
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
        return self._registry  # 외부에서 새로 만들지 않도록 override


# ---------------------------------------------------------------------------
# MergedPatchWithAncilla
# ---------------------------------------------------------------------------

# 잘 정의하면 MergedPatch는 없애도 될 듯? 그냥 이 이름을 그걸로 바꾸자.

class MergedPatchWithAncilla(CodePatch):
    """
    Three-way merge: patch1 ↔ ancilla ↔ patch2
    
    Qubit namespace:
        patch1.data_qubits()         — patch1의 Qubit 객체 그대로
        ancilla.data_qubits()        — ancilla의 Qubit 객체 그대로
        patch2.data_qubits()         — patch2의 Qubit 객체 그대로
    
    Seam consists of:
        - Checks between patch1 and ancilla
        - Checks between ancilla and patch2
    
    Global registry: patch1 → ancilla → patch2 순서
    """
    
    def __init__(
        self,
        patch1: CodePatch,
        ancilla: CodePatch,
        patch2: CodePatch,
        merged_x_checks: list[Check],
        merged_z_checks: list[Check],
        name: str = "merged-ancilla",
    ):
        self._name = name
        self.patch1 = patch1
        self.ancilla = ancilla
        self.patch2 = patch2
        
        # Merged check lists (patch1 + seam + patch2) 순서
        # Note: patch1, ancilla, patch2 각 코드의 checks + seam checks
        self._x_checks = merged_x_checks
        
        self._z_checks = merged_z_checks

        # 이 밑에 다 고쳐야함.
        
        # Seam check tracking
        self._seam_x_checks = seam_x_checks
        self._seam_z_checks = seam_z_checks
        
        # Track seam indices in x_checks
        n_patch1_x = len(patch1.x_checks())
        n_seam_x = len(seam_x_checks)
        
        self._seam_x_slice = slice(
            n_patch1_x,
            n_patch1_x + n_seam_x
        )
        
        # Track seam indices in z_checks
        n_patch1_z = len(patch1.z_checks())
        n_seam_z = len(seam_z_checks)
        
        self._seam_z_slice = slice(
            n_patch1_z,
            n_patch1_z + n_seam_z
        )
        
        # Logicals: 두 패치 각자의 것 유지 (ancilla는 논리 큐빗이 아님)
        self._x_logicals = patch1.x_logicals() + patch2.x_logicals()
        self._z_logicals = patch1.z_logicals() + patch2.z_logicals()
        
        # Global registry: patch1 → ancilla → patch2 순서
        self._registry = QubitRegistry()
        self._registry.register(patch1.data_qubits())
        self._registry.register(ancilla.data_qubits())
        self._registry.register(patch2.data_qubits())
        self.patch_dict = {
            patch1.name: patch1,
            ancilla.name: ancilla,
            patch2.name: patch2
        }

    # qubit을 넣으면 patch 이름을 통해 MergedPatch의 local coord를 계산해 반환하는 함수
    # 각 patch 입장에선 global coord를 반환하는 것임.
    def local_coord(self, q: Qubit) -> tuple[float, float]:
        patch = self.patch_dict[q.patch_name]
        local_x, local_y = q._local_coord()
        global_x = local_x + patch._offset[0]
        global_y = local_y + patch._offset[1]
        return (global_x, global_y)

    @property
    def name(self): return self._name
    
    def data_qubits(self) -> list[Qubit]:
        return (self.patch1.data_qubits() +
                self.ancilla.data_qubits() +
                self.patch2.data_qubits())
    
    def x_checks(self) -> list[Check]: 
        return self._x_checks
    
    def z_checks(self) -> list[Check]: 
        return self._z_checks
    
    def x_logicals(self):              
        return self._x_logicals
    
    def z_logicals(self):              
        return self._z_logicals
    
    def registry(self) -> QubitRegistry:
        return self._registry
    
    def seam_x_checks(self) -> list[Check]:
        """All seam X checks (both seam1 and seam2)"""
        return self._seam_x_checks
    
    def seam_z_checks(self) -> list[Check]:
        """All seam Z checks (both seam1 and seam2)"""
        return self._seam_z_checks
    
    def make_registry(self) -> QubitRegistry:
        return self._registry  # 외부에서 새로 만들지 않도록 override


# ---------------------------------------------------------------------------
# ZZ / XX seam builder
# ---------------------------------------------------------------------------

def build_zz_merged(
    patch1: CodePatch,
    patch2: CodePatch,
    seam: CodePatch
) -> MergedPatch:
    """
    ZZ logical 측정을 위한 merged code.

    각 boundary pair (a, b)에 대해 seam qubit s_i 추가 후
    weight-3 Z check {a, s_i, b} 생성.

    Product of seam Z checks = L_Z1 ⊗ L_Z2 이 되도록
    boundary_pairs는 각 logical string의 qubit을 순서대로 전달.
    """
    assert patch1.d == patch2.d
    left_seam_z_checks = [
    Check(
        frozenset([patch1.qubit_from_coord(patch1.d-1, i),
                    patch1.qubit_from_coord(patch1.d-1, i+1),
                      seam.data_qubits()[i],
                        seam.data_qubits()[i+1]]),
        "Z",
        f"left_seam_Z_{i//2}",
    )
    for i in range(0, patch1.d-1, 2)
    ]
    left_seam_z_checks.append(
        Check(
            frozenset([patch1.qubit_from_coord(patch1.d-1, patch1.d-1),
                        seam.data_qubits()[patch1.d-1]]),
            "Z",
            f"left_seam_Z_{(patch1.d-1)//2}",
        )
    )
    right_seam_z_checks = [
        Check(
            frozenset([patch2.qubit_from_coord(0, i),
                        patch2.qubit_from_coord(0, i+1),
                        seam.data_qubits()[i],
                            seam.data_qubits()[i+1]]),
            "Z",
            f"right_seam_Z_{i//2}",
        )
        for i in range(1, patch2.d-1, 2)
    ]
    right_seam_z_checks.append(
        Check(
            frozenset([patch2.qubit_from_coord(0, 0),
                        seam.data_qubits()[0]]),
            "Z",
            f"right_seam_Z_{(patch2.d-1)//2}",
        )
    )

    # seam_x_check는 left와 right 두 종류로 이뤄져있음.
    # left seam_x_check는 patch1.qubit_from_coord(d-1,i),patch1.qubit_from_coord(d-1,i+1), seam qubit(i), seam qubit(i+1)로 이뤄지고 i=1,3,5,...,d-2
    # right seam_x_check는 patch2.qubit_from_coord(0,i),patch2.qubit_from_coord(0,i+1), seam qubit(i), seam qubit(i+1)로 이뤄짐. i=0,2,4,...,d-3 

    left_seam_x_checks = [
        Check(
            frozenset([patch1.qubit_from_coord(patch1.d-1, i),
                        patch1.qubit_from_coord(patch1.d-1, i+1),
                        seam.data_qubits()[i],
                            seam.data_qubits()[i+1]]),
            "X",
            f"left_seam_X_{(i-1)//2}",
        )
        for i in range(1, patch1.d-1, 2)
    ]
    right_seam_x_checks = [
        Check(
            frozenset([patch2.qubit_from_coord(0, i),
                        patch2.qubit_from_coord(0, i+1),
                        seam.data_qubits()[i],
                            seam.data_qubits()[i+1]]),
            "X",
            f"right_seam_X_{i//2}",
        )
        for i in range(0, patch2.d-1, 2)
    ]

    seam_z_checks = left_seam_z_checks + right_seam_z_checks
    seam_x_checks = left_seam_x_checks + right_seam_x_checks

    merged_z_checks = patch1.z_checks() + seam_z_checks + patch2.z_checks()
    # x_check는 기존 patch의 안쪽 boundary check들은 제외함. 여기서 boundary check라고 함은 weight-2짜리 check를 의미

    merged_x_checks = []
    for chk in patch1.x_checks():
        if chk.label.split("_")[1] == "right":
            continue
        merged_x_checks.append(chk)
    merged_x_checks += seam_x_checks
    for chk in patch2.x_checks():
        if chk.label.split("_")[1] == "left":
            continue
        merged_x_checks.append(chk)
    

    return MergedPatch(
        patch1=patch1,
        patch2=patch2,
        seam=seam,
        merged_x_checks=merged_x_checks,
        merged_z_checks=merged_z_checks,
        name="ZZ-merged",
    )


def build_xx_merged(
    patch1: CodePatch,
    patch2: CodePatch,
    boundary_pairs: list[tuple[Qubit, Qubit]],
    seam_name: str = "seam",
) -> MergedPatch:
    """XX logical 측정을 위한 merged code."""
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
    """
    두 surface code를 left/right으로 붙이는 ZZ surgery seam.

    L_Z = left column이므로, seam은 patch1 left col ↔ patch2 left col.
    seam Z checks의 product = L_Z1 ⊗ L_Z2.
    """
    assert patch1.d == patch2.d
    b1 = patch1.boundary_qubits("left")
    b2 = patch2.boundary_qubits("left")
    return list(zip(b1, b2))


def surface_xx_boundary_pairs(
    patch1: RotatedSurfaceCode,
    patch2: RotatedSurfaceCode,
) -> list[tuple[Qubit, Qubit]]:
    """
    두 surface code를 top/bottom으로 붙이는 XX surgery seam.
    L_X = top row이므로, seam은 patch1 top row ↔ patch2 top row.
    """
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
    # (patch1_boundary_qubit, patch2_boundary_qubit) pairs
    # ancilla qubits와 paired됨
) -> MergedPatch:
    """
    Three-patch merge with ancilla code for qLDPC surgery.

    The merged code consists of:
        - patch1 (data qubits + checks)
        - ancilla (trivial code, [[n, k=n]], data qubits only)
        - patch2 (data qubits + checks)

    Seam checks (Z-type, weight-2):
        - Between patch1 and ancilla:  {patch1_boundary[i], ancilla[i]}
        - Between ancilla and patch2:  {ancilla[i], patch2_boundary[i]}

    Product of all seam Z checks = L_Z1 ⊗ L_Z2.
    
    Args:
        patch1, patch2: Source patches
        ancilla: Ancilla code (typically TrivialCode([[n, k=n]]))
        boundary_pairs_1to2: List of (patch1_q, patch2_q) boundary qubit pairs
        
    Returns:
        MergedPatch with ancilla as data-qubit bridge
    """
    anc_qubits = ancilla.data_qubits()
    n_seam = len(boundary_pairs_1to2)
    
    assert len(anc_qubits) == n_seam, \
        f"Ancilla has {len(anc_qubits)} qubits but need {n_seam} for {len(boundary_pairs_1to2)} boundary pairs"
    
    # No seam checks yet - ancilla qubits initialized to |0>
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
) -> "MergedPatchWithAncilla":
    """
    Three-patch merge with ancilla for XX logical measurement.

    Weight-2 X seam checks:
        - Between patch1 and ancilla:  {patch1_boundary[i], ancilla[i]}
        - Between ancilla and patch2:  {ancilla[i], patch2_boundary[i]}

    Product of all seam X checks = L_X1 ⊗ L_X2.
    """
    anc_qubits = ancilla.data_qubits()
    n_seam = len(boundary_pairs_1to2)
    
    assert len(anc_qubits) == n_seam, \
        f"Ancilla has {len(anc_qubits)} qubits but need {n_seam}"
    
    # No seam checks yet - ancilla qubits initialized to |0>
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
    """
    Boundary qubit pairs for ZZ surgery with ancilla.
    patch1 right col ↔ patch2 left col (via ancilla line).
    """
    assert patch1.d == patch2.d
    b1 = patch1.boundary_qubits("right")
    b2 = patch2.boundary_qubits("left")
    return list(zip(b1, b2))


def surface_xx_boundary_pairs_with_ancilla(
    patch1: RotatedSurfaceCode,
    patch2: RotatedSurfaceCode,
) -> list[tuple[Qubit, Qubit]]:
    """
    Boundary qubit pairs for XX surgery with ancilla.
    patch1 top row ↔ patch2 top row (via ancilla line).
    """
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
    """
    Full surgery circuit:
        Phase 1 (pre):    rounds on patch1, patch2 independently
        Phase 2 (merge):  rounds on merged patch
        Phase 3 (final):  data measurement + OBSERVABLE_INCLUDE
    """
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

    # Phase 1: pre-surgery rounds (patch1, patch2 각자 stabilizer 측정)
    # patch1만의 registry (local)
    reg1 = patch1.make_registry()
    reg2 = patch2.make_registry()

    # Phase 1은 merged registry 기준으로 각 patch의 checks만 측정
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

    # Phase 2: merged rounds
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

    # Phase 3: final measurement
    data_indices = [reg[q] for q in merged.data_qubits()]
    circuit.append("M", data_indices)

    # OBSERVABLE_INCLUDE
    lz1 = merged.patch1.z_logicals()[0]
    lz2 = merged.patch2.z_logicals()[0]

    # Observable 0: ZZ surgery 결과 — Phase 2 마지막 라운드의 seam Z-check 측정들의 XOR
    # 측정 구조: Phase2 last round = [patch1_z | seam_z | patch2_z] (n_merged_z 개)
    # Phase 3 M(n_total) 이후 기준: rec[-(n_total + n_merged_z - si)] (si: check 인덱스)
    n_p1z = len(merged.patch1.z_checks())
    n_seam_z = len(merged.seam_z_checks())
    n_merged_z = len(merged.z_checks())  # = n_p1z + n_seam_z + n_p2z
    seam_zz_targets = [
        stim.target_rec(-(n_total + n_merged_z - (n_p1z + i)))
        for i in range(n_seam_z)
    ]
    circuit.append("OBSERVABLE_INCLUDE", seam_zz_targets, 0)

    # Observable 1: patch1 개별 Z logical — surgery 이후 +1인지 (최종 데이터 측정 기반)
    idx1 = sorted(reg[q] for q in lz1)
    targets_z1 = [stim.target_rec(-(n_total - i)) for i in idx1]
    circuit.append("OBSERVABLE_INCLUDE", targets_z1, 1)

    # Observable 2: patch2 개별 Z logical — surgery 이후 +1인지 (최종 데이터 측정 기반)
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
    """
    주어진 check list에 대해 rounds번 syndrome 측정.
    ancilla는 임시로 data qubit 뒤에 배치 (per-call fresh ancilla).
    """
    import stim

    circuit = stim.Circuit()
    n_global = len(reg)
    data_indices = [reg[q] for q in data_qubit_subset]

    # ancilla: n_global 이후에 임시 배치
    # (실제 회로에서는 ancilla를 재사용하므로, 여기선 개념적으로만)
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
