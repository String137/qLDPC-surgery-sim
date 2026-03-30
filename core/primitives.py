"""
primitives.py — Qubit, Check, QubitRegistry

모든 코드의 기반. Qubit은 (patch, local_idx)로 identity가 결정됨.
stim circuit 생성시에만 QubitRegistry를 통해 int index로 변환.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np


# ---------------------------------------------------------------------------
# Qubit
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Qubit:
    """
    patch: 어느 코드 패치에 속하는지 (e.g. "patch1", "seam", "patch2")
    idx:   그 패치 내의 local index
    """
    patch_name: str
    idx: int
    local_coord: tuple[float, float]

    def __repr__(self):
        return f"Q({self.patch_name},{self.idx})"
    def _local_coord(self):
        return self.local_coord
    


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    """
    Pauli type ("X" or "Z") + qubit support.
    label은 디버깅/시각화용.
    """
    qubits: frozenset[Qubit]
    pauli: str          # "X" or "Z"
    label: str = ""

    def __post_init__(self):
        assert self.pauli in ("X", "Z")

    def __repr__(self):
        qs = sorted(self.qubits)
        return f"{self.pauli}({qs})"

    def weight(self) -> int:
        return len(self.qubits)

    def __add__(self, other: Check) -> Check:
        """두 check의 합 (mod 2)."""
        if self.pauli != other.pauli:
            # CSS code가 아닌 경우를 지원하려면 Check 클래스 구조를 바꿔야함.
            raise NotImplementedError("Cannot add checks of different Pauli types.")
        new_qubits = (self.qubits - other.qubits) | (other.qubits - self.qubits)
        new_label = f"({self.label}+{other.label})"
        return Check(frozenset(new_qubits), self.pauli, new_label)


# ---------------------------------------------------------------------------
# QubitRegistry
# ---------------------------------------------------------------------------

class QubitRegistry:
    """
    Qubit 객체 → int stim index 매핑.
    register() 순서대로 index가 결정됨.

    사용법:
        reg = QubitRegistry()
        reg.register(patch1.all_qubits(), prefix="patch1")
        reg.register(seam_qubits)
        reg.register(patch2.all_qubits(), prefix="patch2")

        idx = reg[qubit]          # Qubit → int
        q   = reg.qubit(idx)      # int → Qubit
    """

    def __init__(self):
        self._q2i: dict[Qubit, int] = {}
        self._i2q: list[Qubit] = []

    def register(self, qubits: Iterable[Qubit]) -> None:
        for q in qubits:
            if q not in self._q2i:
                self._q2i[q] = len(self._i2q)
                self._i2q.append(q)

    def __getitem__(self, q: Qubit) -> int:
        return self._q2i[q]

    def qubit(self, idx: int) -> Qubit:
        return self._i2q[idx]

    def __len__(self) -> int:
        return len(self._i2q)

    def all_qubits(self) -> list[Qubit]:
        return list(self._i2q)

    def indices(self, qubits: Iterable[Qubit]) -> list[int]:
        return [self._q2i[q] for q in qubits]

    def check_to_indices(self, check: Check) -> list[int]:
        return sorted(self._q2i[q] for q in check.qubits)

    def support_to_bitvec(self, qubits: Iterable[Qubit]) -> np.ndarray:
        """주어진 qubit set을 길이 n의 binary vector로 변환."""
        vec = np.zeros(len(self), dtype=np.uint8)
        for q in qubits:
            vec[self._q2i[q]] = 1
        return vec

    def checks_to_matrix(self, checks: list[Check]) -> np.ndarray:
        """Check list → (r, n) binary parity-check matrix."""
        n = len(self)
        H = np.zeros((len(checks), n), dtype=np.uint8)
        for i, chk in enumerate(checks):
            for q in chk.qubits:
                H[i, self._q2i[q]] = 1
        return H
