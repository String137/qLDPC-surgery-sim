"""
codes.py — CodePatch interface + RotatedSurfaceCode + BBCode adapter

CodePatch는 Qubit/Check 레벨에서 코드를 기술.
stim circuit 생성, surgery, decoding은 모두 이 인터페이스만 사용.

Gauge code 확장:
    gauge_x_checks() / gauge_z_checks() override.
    기본값은 stabilizer checks와 동일 (pure stabilizer code).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from core.primitives import Qubit, Check, QubitRegistry


# ---------------------------------------------------------------------------
# CodePatch interface
# ---------------------------------------------------------------------------

class CodePatch(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def data_qubits(self) -> list[Qubit]: ...

    @abstractmethod
    def x_checks(self) -> list[Check]: ...

    @abstractmethod
    def z_checks(self) -> list[Check]: ...

    @abstractmethod
    def x_logicals(self) -> list[frozenset[Qubit]]: ...

    @abstractmethod
    def z_logicals(self) -> list[frozenset[Qubit]]: ...

    # -- gauge (default: same as stabilizer) ---------------------------------
    def gauge_x_checks(self) -> list[Check]:
        return self.x_checks()

    def gauge_z_checks(self) -> list[Check]:
        return self.z_checks()

    # -- derived helpers -----------------------------------------------------
    def n_qubits(self) -> int:
        return len(self.data_qubits())

    def make_registry(self) -> QubitRegistry:
        reg = QubitRegistry()
        reg.register(self.data_qubits())
        return reg

    def h_x(self, reg: QubitRegistry = None) -> np.ndarray:
        if reg is None:
            reg = self.make_registry()
        return reg.checks_to_matrix(self.x_checks())

    def h_z(self, reg: QubitRegistry = None) -> np.ndarray:
        if reg is None:
            reg = self.make_registry()
        return reg.checks_to_matrix(self.z_checks())

    def check_commutativity(self) -> bool:
        reg = self.make_registry()
        return np.all((self.h_x(reg) @ self.h_z(reg).T) % 2 == 0)

    def to_stim_circuit(
        self,
        *,
        reg: QubitRegistry,
        rounds: int,
        p_data: float,
        p_meas: float,
        basis: str = "Z",           # "X", "Z", "both"
        first_round: bool = True,
        last_round: bool = False,
    ):
        """
        주어진 QubitRegistry 기준으로 stim circuit 생성.
        reg는 외부에서 주입 — merged code일 때 global registry 공유.
        """
        import stim

        do_x = basis in ("X", "both")
        do_z = basis in ("Z", "both")

        x_chks = self.x_checks() if do_x else []
        z_chks = self.z_checks() if do_z else []

        n_data = len(reg)
        # ancilla indices: data qubits 이후에 배치
        # X ancillas: n_data .. n_data + len(x_chks) - 1
        # Z ancillas: n_data + len(x_chks) .. end
        x_anc = list(range(n_data, n_data + len(x_chks)))
        z_anc = list(range(n_data + len(x_chks),
                           n_data + len(x_chks) + len(z_chks)))

        circuit = stim.Circuit()

        if first_round:
            all_qubits = list(range(n_data + len(x_chks) + len(z_chks)))
            circuit.append("R", all_qubits)

        prev_x_meas_count = 0
        prev_z_meas_count = 0

        for t in range(rounds):
            # Data errors
            if p_data > 0:
                data_indices = list(range(n_data))
                circuit.append("X_ERROR", data_indices, p_data)
                circuit.append("Z_ERROR", data_indices, p_data)

            # -- X stabilizers -----------------------------------------------
            if do_x:
                circuit.append("R", x_anc)
                for ai, chk in enumerate(x_chks):
                    anc = x_anc[ai]
                    circuit.append("H", [anc])
                    for q in sorted(reg.check_to_indices(chk)):
                        circuit.append("CNOT", [anc, q])
                    circuit.append("H", [anc])
                if p_meas > 0:
                    circuit.append("X_ERROR", x_anc, p_meas)
                circuit.append("M", x_anc)

                for ai in range(len(x_chks)):
                    rec_cur = stim.target_rec(-(len(x_chks) - ai))
                    if t == 0:
                        circuit.append("DETECTOR", [rec_cur], [ai, 0, t])
                    else:
                        rec_prev = stim.target_rec(
                            -(len(x_chks) - ai) - len(x_chks))
                        circuit.append("DETECTOR",
                                       [rec_cur, rec_prev], [ai, 0, t])

            # -- Z stabilizers -----------------------------------------------
            if do_z:
                circuit.append("R", z_anc)
                for ai, chk in enumerate(z_chks):
                    anc = z_anc[ai]
                    for q in sorted(reg.check_to_indices(chk)):
                        circuit.append("CNOT", [q, anc])
                if p_meas > 0:
                    circuit.append("X_ERROR", z_anc, p_meas)
                circuit.append("M", z_anc)

                for ai in range(len(z_chks)):
                    rec_cur = stim.target_rec(-(len(z_chks) - ai))
                    if t == 0:
                        circuit.append("DETECTOR", [rec_cur], [ai, 1, t])
                    else:
                        rec_prev = stim.target_rec(
                            -(len(z_chks) - ai) - len(z_chks))
                        circuit.append("DETECTOR",
                                       [rec_cur, rec_prev], [ai, 1, t])

        if last_round:
            data_indices = list(range(n_data))
            circuit.append("M", data_indices)
            for li, support in enumerate(self.z_logicals()):
                indices = sorted(reg[q] for q in support)
                targets = [stim.target_rec(-(n_data - i)) for i in indices]
                circuit.append("OBSERVABLE_INCLUDE", targets, li)

        return circuit


# ---------------------------------------------------------------------------
# Rotated Surface Code
# ---------------------------------------------------------------------------

class RotatedSurfaceCode(CodePatch):
    """
    Rotated surface code distance d. (Fowler et al. 2012 convention)

    Qubit index: row*d + col

    Plaquette at center (r+0.5, c+0.5), r,c in 0..d-2:
        (r+c) even  →  Z plaquette
        (r+c) odd   →  X plaquette

    Bulk plaquettes are weight-4.
    Boundary half-plaquettes are weight-2:
        Z-smooth boundaries (top/bottom):
            top:    Z weight-2 at columns c where (0+c) % 2 == 1, i.e. c=1,3,...
            bottom: Z weight-2 at columns c where (d-2+c) % 2 == 1
        X-smooth boundaries (left/right):
            left:   X weight-2 at rows r where (r+0) % 2 == 0, i.e. r=0,2,...
            right:  X weight-2 at rows r where (r+d-2) % 2 == 0

    Logical operators (Fowler convention):
        L_Z: left column  (0,0)..(d-1, 0)   ← Z-rough boundary = top/bottom
        L_X: top row      (0,0)..(0, d-1)   ← X-rough boundary = left/right

    d=3 checks:
        Z: Z_bulk_0_0={0,1,3,4}, Z_bulk_1_1={4,5,7,8}, Z_top_1={1,2}, Z_bot_0={6,7}
        X: X_bulk_0_1={1,2,4,5}, X_bulk_1_0={3,4,6,7}, X_left_0={0,3}, X_right_1={5,8}
        L_Z={0,3,6}  L_X={0,1,2}
    """

    def __init__(self, d: int, patch_name: str = "patch"):
        self.d = d
        self._name = patch_name
        self._build()

    @property
    def name(self) -> str:
        return self._name

    def _q(self, r: int, c: int) -> Qubit:
        return Qubit(self._name, r * self.d + c)

    def _build(self):
        d = self.d
        self._data_qubits = [self._q(r, c)
                             for r in range(d) for c in range(d)]

        self._x_checks: list[Check] = []
        self._z_checks: list[Check] = []

        # ------------------------------------------------------------------
        # Bulk plaquettes: center at (r+0.5, c+0.5), r,c in 0..d-2
        # ------------------------------------------------------------------
        for r in range(d - 1):
            for c in range(d - 1):
                support = frozenset([
                    self._q(r,   c),   self._q(r,   c+1),
                    self._q(r+1, c),   self._q(r+1, c+1),
                ])
                if (r + c) % 2 == 0:
                    self._z_checks.append(Check(support, "Z", f"Z_bulk_{r}_{c}"))
                else:
                    self._x_checks.append(Check(support, "X", f"X_bulk_{r}_{c}"))

        # ------------------------------------------------------------------
        # Boundary half-plaquettes
        #
        # Key insight: top/bottom boundaries are Z-type, left/right are X-type.
        # A boundary check is the exposed edge of a bulk plaquette of the
        # SAME type that touches the boundary.
        #
        # Top boundary Z: exposed top edge of bulk Z plaquettes in row r=0.
        #   Bulk Z plaquettes in row r=0 are at columns c where (0+c)%2==0,
        #   i.e. c=0,2,4,... Their top edge is {(0,c),(0,c+1)}.
        #
        # Bottom boundary Z: exposed bottom edge of bulk Z plaquettes in row r=d-2.
        #   Bulk Z at (r=d-2, c) where (d-2+c)%2==0.
        #   Their bottom edge is {(d-1,c),(d-1,c+1)}.
        #
        # Left boundary X: exposed left edge of bulk X plaquettes in col c=0.
        #   Bulk X at (r,c=0) where (r+0)%2==1, i.e. r=1,3,...
        #   Their left edge is {(r,0),(r+1,0)}.
        #
        # Right boundary X: exposed right edge of bulk X plaquettes in col c=d-2.
        #   Bulk X at (r,c=d-2) where (r+d-2)%2==1.
        #   Their right edge is {(r,d-1),(r+1,d-1)}.
        #
        # This guarantees every boundary check shares exactly 2 qubits with
        # the adjacent bulk check of the same type → commutes with all others.
        # ------------------------------------------------------------------

        # Top boundary Z
        # Z_top at column c sits on the top edge of bulk plaquette (r=0, c).
        # For commutativity with X-bulk, the plaquette to the LEFT (r=0, c-1)
        # must be Z-type, i.e. (0 + c-1) % 2 == 0  ↔  c % 2 == 1.
        for c in range(d - 1):
            if (0 + c) % 2 == 1:        # bulk Z plaquette at (r=0, c)
                self._z_checks.append(Check(
                    frozenset([self._q(0, c), self._q(0, c+1)]),
                    "Z", f"Z_top_{c}"))

        # Bottom boundary Z
        for c in range(d - 1):
            if (d - 2 + c) % 2 == 1:    # bulk Z plaquette at (r=d-2, c)
                self._z_checks.append(Check(
                    frozenset([self._q(d-1, c), self._q(d-1, c+1)]),
                    "Z", f"Z_bot_{c}"))

        # Left boundary X
        # X_left at row r sits on the left edge of bulk plaquette (r, c=0).
        # For commutativity with Z-bulk, the plaquette ABOVE (r-1, c=0)
        # must be X-type, i.e. (r-1 + 0) % 2 == 1  ↔  r % 2 == 0.
        for r in range(d - 1):
            if (r + 0) % 2 == 0:        # bulk X plaquette at (r, c=0)
                self._x_checks.append(Check(
                    frozenset([self._q(r, 0), self._q(r+1, 0)]),
                    "X", f"X_left_{r}"))

        # Right boundary X
        for r in range(d - 1):
            if (r + d - 2) % 2 == 0:    # bulk X plaquette at (r, c=d-2)
                self._x_checks.append(Check(
                    frozenset([self._q(r, d-1), self._q(r+1, d-1)]),
                    "X", f"X_right_{r}"))

        # ------------------------------------------------------------------
        # Logical operators  (Fowler convention)
        # L_Z: right column  — crosses Z-rough boundary (top/bottom)
        # L_X: top row      — crosses X-rough boundary (left/right)
        # ------------------------------------------------------------------
        self._z_logicals = [
            frozenset(self._q(r, d-1) for r in range(d))
        ]
        self._x_logicals = [
            frozenset(self._q(0, c) for c in range(d))
        ]

    # -- CodePatch interface -------------------------------------------------
    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self) -> list[Check]:    return self._x_checks
    def z_checks(self) -> list[Check]:    return self._z_checks
    def x_logicals(self):                 return self._x_logicals
    def z_logicals(self):                 return self._z_logicals

    # -- geometry ------------------------------------------------------------
    def qubit_coords(self) -> dict[Qubit, tuple[int, int]]:
        return {self._q(r, c): (r, c)
                for r in range(self.d) for c in range(self.d)}

    def boundary_qubits(self, side: str) -> list[Qubit]:
        """side: 'top', 'bottom', 'left', 'right'"""
        d = self.d
        return {
            "top":    [self._q(0, c)   for c in range(d)],
            "bottom": [self._q(d-1, c) for c in range(d)],
            "left":   [self._q(r, 0)   for r in range(d)],
            "right":  [self._q(r, d-1) for r in range(d)],
        }[side]


# ---------------------------------------------------------------------------
# Trivial Code: [[n, k=n]]
# ---------------------------------------------------------------------------

class TrivialCode(CodePatch):
    """
    Trivial code with n data qubits and NO stabilizer checks (k=n).
    
    Used to represent ancilla lines in qLDPC surgery.
    All qubits are logical — no stabilizers.
    
    Example: 1-row ancilla for merging two surface codes.
    """
    
    def __init__(self, n: int, code_name: str = "trivial"):
        self.n = n
        self._name = code_name
        self._data_qubits = [Qubit(code_name, i) for i in range(n)]
    
    @property
    def name(self) -> str:
        return self._name
    
    def data_qubits(self) -> list[Qubit]:
        return self._data_qubits
    
    def x_checks(self) -> list[Check]:
        return []  # No stabilizers
    
    def z_checks(self) -> list[Check]:
        return []  # No stabilizers
    
    def x_logicals(self) -> list[frozenset[Qubit]]:
        # Each data qubit is a logical X
        return [frozenset([q]) for q in self._data_qubits]
    
    def z_logicals(self) -> list[frozenset[Qubit]]:
        # Each data qubit is a logical Z
        return [frozenset([q]) for q in self._data_qubits]


# ---------------------------------------------------------------------------
# BBCode adapter (네 기존 코드에서 가져옴)
# ---------------------------------------------------------------------------

class BBCodePatch(CodePatch):
    """
    기존 BBCode 클래스를 CodePatch로 감싸는 adapter.
    BBCode 인스턴스를 받아서 Hx/Hz → Check list로 변환.
    """

    def __init__(self, bb, patch_name: str = "bb"):
        """
        bb: 기존 BBCode 인스턴스
        """
        self._bb = bb
        self._name = patch_name
        n = bb.n
        self._data_qubits = [Qubit(patch_name, i) for i in range(n)]
        self._x_checks = self._matrix_to_checks(bb.Hx, "X")
        self._z_checks = self._matrix_to_checks(bb.Hz, "Z")
        # logicals은 get_logical_representatives() 이후에 채워짐
        self._x_logicals: list[frozenset[Qubit]] = []
        self._z_logicals: list[frozenset[Qubit]] = []

    def _matrix_to_checks(self, H, pauli: str) -> list[Check]:
        checks = []
        H_np = np.array(H, dtype=np.uint8)
        for i, row in enumerate(H_np):
            support = frozenset(
                Qubit(self._name, j) for j in np.where(row)[0]
            )
            checks.append(Check(support, pauli, f"{pauli}_{i}"))
        return checks

    def load_logicals(self):
        """BBCode.get_logical_representatives() 호출 후 실행."""
        self._bb.get_logical_representatives()
        lx = np.array(self._bb.logical_x_repr, dtype=np.uint8)
        lz = np.array(self._bb.logical_z_repr, dtype=np.uint8)
        self._x_logicals = [
            frozenset(Qubit(self._name, j) for j in np.where(row)[0])
            for row in lx
        ]
        self._z_logicals = [
            frozenset(Qubit(self._name, j) for j in np.where(row)[0])
            for row in lz
        ]

    @property
    def name(self): return self._name
    def data_qubits(self): return self._data_qubits
    def x_checks(self): return self._x_checks
    def z_checks(self): return self._z_checks
    def x_logicals(self): return self._x_logicals
    def z_logicals(self): return self._z_logicals
