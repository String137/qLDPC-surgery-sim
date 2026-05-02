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
from docplex.mp.model import Model
from docplex.mp.progress import ProgressClock
from core.CSS import CSS
import galois
# ---------------------------------------------------------------------------
# CodePatch interface
# ---------------------------------------------------------------------------

class CodePatch(ABC):
    def __init__(self, offset=(0.0, 0.0)):
        self._offset = offset    # (float, float)
        self.css = None
        self._data_qubits = None
        self.check2coord = None
        
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
    def local_coord(self, q: Qubit) -> tuple[float, float]: ...

    # @abstractmethod
    # def x_logicals(self) -> list[frozenset[Qubit]]: ...

    # @abstractmethod
    # def z_logicals(self) -> list[frozenset[Qubit]]: ...

    # logical operators는 직접 구현하자.
    # check 만족하는 operator 찾으면 되고, 원하는 logical operator를 고정시키고, 남은 operator를 찾는 부분도 만들자.


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

    def make_css(self):
        reg = self.make_registry()
        Gx = galois.GF(2)(self.h_x(reg) )
        Gz = galois.GF(2)(self.h_z(reg) )
        self.css = CSS(Gx, Gz)
        return self
    
    def get_distance(self) -> tuple[int, int]:
        if self.css is None:
            self.make_css()
        self.dx, self.dz = self.css.get_distance()
        return self.dx, self.dz
    def get_k(self) -> int:
        if self.css is None:
            self.make_css()
        return self.css.k
    
    def add_qubit(self, x, y) -> Qubit:
        # qubit 추가 (data qubits에 추가)
        # check 업데이트는 외부에서 직접 해줘야 함.
        self._data_qubits.append(Qubit(self.name, len(self._data_qubits), (x, y)))
        return self._data_qubits[-1]
    
    def add_check(self, qubits: list[Qubit], pauli: str, label: str = "") -> Check:
        chk = Check(frozenset(qubits), pauli, label)
        if pauli == "X":
            self._x_checks.append(chk)
        elif pauli == "Z":
            self._z_checks.append(chk)
        else:
            raise ValueError("pauli must be 'X' or 'Z'")
        return chk
    
    def get_idx(self, qubit: Qubit) -> int:
        # qubit index 반환 (data qubits 기준)
        return self.data_qubits().index(qubit)
    
    def set_check2coord(self, check2coord: dict):
        # check2coord는 check → (x,y) 좌표 매핑. surgery에서 ancilla 배치할 때 필요.
        self.check2coord = check2coord
    
    # def to_stim_circuit(
    #     self,
    #     *,
    #     reg: QubitRegistry,
    #     rounds: int,
    #     p_data: float,
    #     p_meas: float,
    #     basis: str = "Z",           # "X", "Z", "both"
    #     first_round: bool = True,
    #     last_round: bool = False,
    # ):
    #     """
    #     주어진 QubitRegistry 기준으로 stim circuit 생성.
    #     reg는 외부에서 주입 — merged code일 때 global registry 공유.
    #     """
    #     import stim

    #     do_x = basis in ("X", "both")
    #     do_z = basis in ("Z", "both")

    #     x_chks = self.x_checks() if do_x else []
    #     z_chks = self.z_checks() if do_z else []

    #     n_data = len(reg)
    #     # ancilla indices: data qubits 이후에 배치
    #     # X ancillas: n_data .. n_data + len(x_chks) - 1
    #     # Z ancillas: n_data + len(x_chks) .. end
    #     x_anc = list(range(n_data, n_data + len(x_chks)))
    #     z_anc = list(range(n_data + len(x_chks),
    #                        n_data + len(x_chks) + len(z_chks)))

    #     circuit = stim.Circuit()

    #     if first_round:
    #         all_qubits = list(range(n_data + len(x_chks) + len(z_chks)))
    #         circuit.append("R", all_qubits)

    #     prev_x_meas_count = 0
    #     prev_z_meas_count = 0

    #     for t in range(rounds):
    #         # Data errors
    #         if p_data > 0:
    #             data_indices = list(range(n_data))
    #             circuit.append("X_ERROR", data_indices, p_data)
    #             circuit.append("Z_ERROR", data_indices, p_data)

    #         # -- X stabilizers -----------------------------------------------
    #         if do_x:
    #             circuit.append("R", x_anc)
    #             for ai, chk in enumerate(x_chks):
    #                 anc = x_anc[ai]
    #                 circuit.append("H", [anc])
    #                 for q in sorted(reg.check_to_indices(chk)):
    #                     circuit.append("CNOT", [anc, q])
    #                 circuit.append("H", [anc])
    #             if p_meas > 0:
    #                 circuit.append("X_ERROR", x_anc, p_meas)
    #             circuit.append("M", x_anc)

    #             for ai in range(len(x_chks)):
    #                 rec_cur = stim.target_rec(-(len(x_chks) - ai))
    #                 if t == 0:
    #                     circuit.append("DETECTOR", [rec_cur], [ai, 0, t])
    #                 else:
    #                     rec_prev = stim.target_rec(
    #                         -(len(x_chks) - ai) - len(x_chks))
    #                     circuit.append("DETECTOR",
    #                                    [rec_cur, rec_prev], [ai, 0, t])

    #         # -- Z stabilizers -----------------------------------------------
    #         if do_z:
    #             circuit.append("R", z_anc)
    #             for ai, chk in enumerate(z_chks):
    #                 anc = z_anc[ai]
    #                 for q in sorted(reg.check_to_indices(chk)):
    #                     circuit.append("CNOT", [q, anc])
    #             if p_meas > 0:
    #                 circuit.append("X_ERROR", z_anc, p_meas)
    #             circuit.append("M", z_anc)

    #             for ai in range(len(z_chks)):
    #                 rec_cur = stim.target_rec(-(len(z_chks) - ai))
    #                 if t == 0:
    #                     circuit.append("DETECTOR", [rec_cur], [ai, 1, t])
    #                 else:
    #                     rec_prev = stim.target_rec(
    #                         -(len(z_chks) - ai) - len(z_chks))
    #                     circuit.append("DETECTOR",
    #                                    [rec_cur, rec_prev], [ai, 1, t])

    #     if last_round:
    #         data_indices = list(range(n_data))
    #         circuit.append("M", data_indices)
    #         for li, support in enumerate(self.z_logicals()):
    #             indices = sorted(reg[q] for q in support)
    #             targets = [stim.target_rec(-(n_data - i)) for i in indices]
    #             circuit.append("OBSERVABLE_INCLUDE", targets, li)

    #     return circuit


class EmptyPatch(CodePatch):
    def __init__(self, patch_name: str = "empty", offset=(0.0, 0.0)):
        super().__init__(offset)
        self._name = patch_name
        self._data_qubits = []
        self._x_checks = []
        self._z_checks = []

    @property
    def name(self) -> str:
        return self._name

    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self) -> list[Check]:    return self._x_checks
    def z_checks(self) -> list[Check]:    return self._z_checks


# ---------------------------------------------------------------------------
# Surface Code
# ---------------------------------------------------------------------------


class SurfaceCode(CodePatch):
    # See surface_lattice.jpg
    # 총 d^2+(d-1)^2= 2d^2-2d+1 qubits
    # (x,y) = (n,m), where n,m is half integer, and n+m is an integer, and 0 <= n,m <= d-1

    def __init__(self, d: int, patch_name: str = "surf", offset=(0.0, 0.0)):
        self.d = d
        self._name = patch_name
        self._offset = offset
        self._build()
    
    @property
    def name(self) -> str:
        return self._name
    
    def _coord_to_idx(self, x: float, y: float) -> int:
        # (x,y) 좌표 → qubit index
        assert (x + y) % 1 == 0, "x+y must be an integer"
        assert 0 <= x <= self.d - 1 and 0 <= y <= self.d - 1, "x and y must be in [0, d-1]"
        assert (2 * x) % 1 == 0 and (2 * y) % 1 == 0, "x and y must be half-integers"
        # x: 오른쪽으로 증가, y: 아래로 증가
        # index는 y=0부터 x=0..d-1, y=0.5부터 x=0.5..d-1.5, y=1부터 x=0..d-1, ... 순서로 매겨짐.
        if y % 1 == 0:
            idx = int(y * (2 * self.d-1) + x)
        else:
            idx = int((y-0.5) * (2 * self.d-1) + self.d + x - 0.5)
        return idx
    
    def _q(self, x: float, y: float) -> Qubit:
        # (x,y) local 좌표에 있는 qubit을 생성.
        idx = self._coord_to_idx(x, y)
        return Qubit(self._name, idx, (x, y))
    
    def _idx_to_coord(self, idx: int) -> tuple[float, float]:
        # idx로부터 (x,y) 좌표 계산
        y = (idx // (2 * self.d - 1))
        x = idx % (2 * self.d - 1)
        if x >= self.d:
            x -= self.d
            x += 0.5
            y += 0.5
        return (x, y)
    
    def local_coord(self, q):
        return q.local_coord

    def _get_qubit_from_idx(self, idx: int) -> Qubit:
        # idx로부터 Qubit 객체 반환
        return self._data_qubits[idx]  # 이미 생성된 qubit 객체 반환
    def _get_qubit_from_coord(self, x: float, y: float) -> Qubit:
        idx = self._coord_to_idx(x, y)
        return self._get_qubit_from_idx(idx)
    
    def _build(self):
        # x: 오른쪽, y: 아래로 증가
        total_num_qubits = 2 * self.d**2 - 2 * self.d + 1
        self._data_qubits = [Qubit(self._name, idx, self._idx_to_coord(idx))
                             for idx in range(total_num_qubits)]

        self._x_checks: list[Check] = []
        self._z_checks: list[Check] = []

        # check의 좌표는 (x,y), where x+y = integer + 0.5꼴
        # y가 정수이면 X-check, y가 반정수이면 Z-check

        # check 좌표 (x,y)에 대해 상하좌우로 0.5씩 떨어진 qubit들이 (있다면) check의 support가 됨.
        check2coord = {} # check → (x,y) 좌표 매핑. surgery에서 ancilla 배치할 때 필요.
        for yy in range(0, 2*self.d-1):
            for xx in range(0, 2*self.d-1):
                x = float(xx)/2
                y = float(yy)/2
                if (x + y) % 1 == 0.5:
                    chk_qubits = []
                    for dx, dy in [[-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]]:
                        qx = x + dx
                        qy = y + dy
                        if 0 <= qx <= self.d - 1 and 0 <= qy <= self.d - 1:
                            chk_qubits.append(self._get_qubit_from_coord(qx, qy))
                    chk_qubits = frozenset(chk_qubits)
                    if y % 1 == 0:   # y가 정수 → X-check
                        self._x_checks.append(Check(chk_qubits, "X", f"X_{x}_{y}"))
                    else:            # y가 반정수 → Z-check
                        self._z_checks.append(Check(chk_qubits, "Z", f"Z_{x}_{y}"))
                    check2coord[self._x_checks[-1] if y % 1 == 0 else self._z_checks[-1]] = (x, y)
        self.set_check2coord(check2coord)
    # -- CodePatch interface -------------------------------------------------
    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self) -> list[Check]:    return self._x_checks
    def z_checks(self) -> list[Check]:    return self._z_checks


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

    def __init__(self, d: int, patch_name: str = "patch", offset=(0.0, 0.0)):
        self.d = d
        self._name = patch_name
        self._offset = offset
        self._build()

    @property
    def name(self) -> str:
        return self._name

    def _q(self, x: int, y: int) -> Qubit:
        # x: 오른쪽으로 증가, y: 아래로 증가
        return Qubit(self._name, y * self.d + x, (x, y))

    def _build(self):
        d = self.d
        # x: 오른쪽, y: 아래로 증가
        self._data_qubits = [self._q(x, y)
                                for y in range(d) for x in range(d)]

        self._x_checks: list[Check] = []
        self._z_checks: list[Check] = []

        # Bulk plaquettes: center at (x+0.5, y+0.5), x,y in 0..d-2
        for y in range(d - 1):
            for x in range(d - 1):
                support = frozenset([
                    self._q(x,   y),   self._q(x+1, y),
                    self._q(x, y+1),   self._q(x+1, y+1),
                ])
                if (x + y) % 2 == 0:
                    self._z_checks.append(Check(support, "Z", f"Z_bulk_{x}_{y}"))
                else:
                    self._x_checks.append(Check(support, "X", f"X_bulk_{x}_{y}"))

        # Top boundary Z (y=0)
        for x in range(d - 1):
            if (x + 0) % 2 == 1:
                self._z_checks.append(Check(
                    frozenset([self._q(x, 0), self._q(x+1, 0)]),
                    "Z", f"Z_top_{x}"))

        # Bottom boundary Z (y=d-1)
        for x in range(d - 1):
            if (x + d - 2) % 2 == 1:
                self._z_checks.append(Check(
                    frozenset([self._q(x, d-1), self._q(x+1, d-1)]),
                    "Z", f"Z_bot_{x}"))

        # Left boundary X (x=0)
        for y in range(d - 1):
            if (0 + y) % 2 == 0:
                self._x_checks.append(Check(
                    frozenset([self._q(0, y), self._q(0, y+1)]),
                    "X", f"X_left_{y}"))

        # Right boundary X (x=d-1)
        for y in range(d - 1):
            if (d - 1 + y) % 2 == 0:
                self._x_checks.append(Check(
                    frozenset([self._q(d-1, y), self._q(d-1, y+1)]),
                    "X", f"X_right_{y}"))

        # Logical operators (Fowler convention)
        # L_Z: 오른쪽 끝 x=d-1 column
        # L_X: 맨 위 y=0 row
        self._z_logicals = [
            frozenset(self._q(d-1, y) for y in range(d))
        ]
        self._x_logicals = [
            frozenset(self._q(x, 0) for x in range(d))
        ]
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
    # def qubit_coords(self) -> dict[Qubit, tuple[int, int]]:
    #     return {self._q(r, c): (r, c)
    #             for r in range(self.d) for c in range(self.d)}
    def qubit_from_coord(self, x: int, y: int) -> Qubit:
        idx = y * self.d + x
        return self._data_qubits[idx]
    def boundary_qubits(self, side: str) -> list[Qubit]:
        """side: 'top', 'bottom', 'left', 'right'"""
        d = self.d
        return {
            "top":    [self._q(x, 0)   for x in range(d)],
            "bottom": [self._q(x, d-1) for x in range(d)],
            "left":   [self._q(0, y)   for y in range(d)],
            "right":  [self._q(d-1, y) for y in range(d)],
        }[side]


# ---------------------------------------------------------------------------
# Trivial Code: [[n, k=n]]
# ---------------------------------------------------------------------------

class TrivialCode(CodePatch):
    """
    Trivial code with n data qubits and NO stabilizer checks (k=n).
    
    Used to represent ancilla lines in qLDPC surgery.
    All qubits are logical — no stabilizers.
    
    Example: 1-column ancilla for merging two surface codes.
    """
    
    def __init__(self, n: int, patch_name: str = "trivial", offset=(0.0, 0.0), direction: str = "vertical"):
        self.n = n
        self._name = patch_name
        self._offset = offset
        self._direction = direction
        if direction == "vertical":
            self._data_qubits = [Qubit(patch_name, i, (0, i)) for i in range(n)]
        elif direction == "horizontal":
            self._data_qubits = [Qubit(patch_name, i, (i, 0)) for i in range(n)]
        else:
            raise ValueError("direction must be 'vertical' or 'horizontal'")
    
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


class LPCodePatch(CodePatch):
    
    pass


class CCCode(CodePatch):
    # https://arxiv.org/pdf/2603.05398
    pass

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
