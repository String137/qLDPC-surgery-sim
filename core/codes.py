"""
codes.py — CodePatch interface + 구현체 (Surface, RotatedSurface, HGP, LP, BB, Trivial, Empty, AncillaGraph)

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
from core.CSS import CSS, LPCode, HGPCode
import galois
import networkx as nx
from collections import deque
# ---------------------------------------------------------------------------
# CodePatch interface
# ---------------------------------------------------------------------------

class CodePatch(ABC):
    def __init__(self, offset=(0.0, 0.0)):
        self._offset = offset    # (float, float)
        self.css = None
        self._data_qubits = None
        self.check2coord = None
        
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def data_qubits(self) -> list[Qubit]: ...

    @abstractmethod
    def x_checks(self) -> list[Check]: ...

    @abstractmethod
    def z_checks(self) -> list[Check]: ...

    def local_coord(self, q: Qubit) -> tuple[float, float] | None:
        return q.local_coord

    def set_css(self, css: CSS):
        self.css = css
        self._data_qubits = [Qubit(self.name(), idx, None) for idx in range(css.n)]
        self._x_checks: list[Check] = []
        self._z_checks: list[Check] = []
        for i in range(css.Hx.shape[0]):
            support = frozenset(self._data_qubits[j] for j in range(css.n) if css.Hx[i, j] == 1)
            self._x_checks.append(Check(support, "X", f"X_{i}"))
        for i in range(css.Hz.shape[0]):
            support = frozenset(self._data_qubits[j] for j in range(css.n) if css.Hz[i, j] == 1)
            self._z_checks.append(Check(support, "Z", f"Z_{i}"))
        if all(l is not None for l in css.logical_x_repr + css.logical_z_repr):
            self.set_logicals(css.logical_x_repr, css.logical_z_repr)


    def set_logicals(self, lx: list[list[int]], lz: list[list[int]]):
        self._x_logicals = [frozenset(self._data_qubits[j] for j in range(self.css.n) if lx[i][j] == 1)
                            for i in range(self.css.k)]
        self._z_logicals = [frozenset(self._data_qubits[j] for j in range(self.css.n) if lz[i][j] == 1)
                            for i in range(self.css.k)]
    # logical operators는 직접 구현하자.
    # check 만족하는 operator 찾으면 되고, 원하는 logical operator를 고정시키고, 남은 operator를 찾는 부분도 만들자.
    def get_logical_representative(self, type: str = "X", idx: int = 0) -> list[Qubit]:
        if self.css is None:
            self.make_css()
        if any(l is None for l in self.css.logical_x_repr + self.css.logical_z_repr):
            self.css.set_missing_logicals()
        # self.logical_x_repr는 list[Qubit] type이 아니라 0,1들 list임.
        # list[Qubit]을 리턴해야함!!
        if type == "X":
            support = []
            for j in range(self.css.n):
                if self.css.logical_x_repr[idx][j] == 1:
                    support.append(self._data_qubits[j])
            return support
        else:
            support = []
            for j in range(self.css.n):
                if self.css.logical_z_repr[idx][j] == 1:
                    support.append(self._data_qubits[j])
            return support

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
        self._data_qubits.append(Qubit(self.name(), len(self._data_qubits), (x, y)))
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


class EmptyPatch(CodePatch):
    def __init__(self, patch_name: str = "empty", offset=(0.0, 0.0)):
        super().__init__(offset)
        self._name = patch_name
        self._data_qubits = []
        self._x_checks = []
        self._z_checks = []

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
        super().__init__(offset)
        self.d = d
        self._name = patch_name
        self._offset = offset
        self._build()
    
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

class RotatedSurfaceCodePatch(CodePatch):
    def __init__(self, css: CSS, patch_name: str = "rot_surf", offset=(0.0, 0.0), scale=(1,1)):
        super().__init__(offset)
        self._name = patch_name
        self.d = css.d
        self.set_css(css)
        self.scale = scale

    def name(self) -> str:
        return self._name
    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self) -> list[Check]:    return self._x_checks
    def z_checks(self) -> list[Check]:    return self._z_checks
    def local_coord(self, q: Qubit):
        idx = self.get_idx(q)
        y = idx//self.d
        x = idx%self.d
        scale_x, scale_y = self.scale
        return (x*scale_x, y*scale_y)
    def local_coord_chk(self, typ:str, idx:int):
        raise NotImplementedError


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
        super().__init__(offset)
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


class HGPCodePatch(CodePatch):
    def __init__(self, css: HGPCode, patch_name: str = "hgp", offset=(0.0,0.0), scale=(1,1)):
        super().__init__(offset)
        self._name = patch_name
        self.set_css(css)
        self.scale = scale
        self.Ca = self.css.Ca
        self.Cb = self.css.Cb
    def name(self) -> str:
        return self._name
    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self) -> list[Check]: return self._x_checks
    def z_checks(self) -> list[Check]: return self._z_checks
    def local_coord(self, q: Qubit) -> tuple[float, float]:
        """Cb 축(자기 qubit/check 좌표)이 세로(y, "열"), Ca 축이 가로(x, "행").
        block1 은 오른쪽 아래, block2 는 왼쪽 위 사분면 (이전 버전에서 x<->y 를
        뒤집은 것 -- Cb_A 가 열, Ca_A 가 행이 되도록)."""
        idx = self.get_idx(q)
        css = self.css
        ra, na, nb = css.ra, css.na, css.nb
        scale_x, scale_y = self.scale
        if idx < ra * nb:                      # block1 : Ca0 (x) Cb1
            b, j = divmod(idx, nb)
            x, y = na + b, j
        else:                                   # block2 : Ca1 (x) Cb0
            i, s = divmod(idx - ra * nb, css.rb)
            x, y = i, nb + s
        return (float(x) * scale_x, float(y) * scale_y)

    def local_coord_chk(self, typ: str, idx: int) -> tuple[float, float]:
        """typ 은 'X' 또는 'Z'. idx 는 css.Hx / css.Hz 의 행 번호.
        local_coord() 와 같은 x<->y 축 (Cb=세로, Ca=가로)."""
        css = self.css
        na, nb, rb = css.na, css.nb, css.rb
        scale_x, scale_y = self.scale
        if typ == "X":                          # Ca1 (x) Cb1
            i, j = divmod(idx, nb)
            x, y = i, j
        elif typ == "Z":                        # Ca0 (x) Cb0
            b, s = divmod(idx, rb)
            x, y = na + b, nb + s
        else:
            raise ValueError("typ must be 'X' or 'Z'")
        return (float(x) * scale_x, float(y) * scale_y)

class LPCodePatch(CodePatch):
    def __init__(self, css: LPCode, patch_name: str = "lp", offset=(0.0, 0.0), scale=(1,1)):
        super().__init__(offset)
        self._name = patch_name
        self.set_css(css)
        self.scale = scale
        
    
    def name(self) -> str:
        return self._name
    def data_qubits(self) -> list[Qubit]: return self._data_qubits
    def x_checks(self) -> list[Check]:    return self._x_checks
    def z_checks(self) -> list[Check]:    return self._z_checks
    def local_coord(self, q: Qubit):
        ra = len(self.css.Ha)
        ca = len(self.css.Ha[0])
        rb = len(self.css.Hb)
        cb = len(self.css.Hb[0])
        p = self.css.l  # ring size
        scale_x, scale_y = self.scale
        idx = self.get_idx(q)
        if idx < ca*rb*p:
            # left sector
            x = idx // (rb*p)
            y = (idx % (rb*p))
            return (x*scale_x, y*scale_y)
        else:
            # right sector
            idx -= ca*rb*p
            x = idx // (cb*p)
            y = (idx % (cb*p))
            return ((x + ca)*scale_x, (y + rb*p)*scale_y)


    
    # check의 좌표
    def local_coord_chk(self, typ:str, idx:int):
        ra = len(self.css.Ha)
        ca = len(self.css.Ha[0])
        rb = len(self.css.Hb)
        cb = len(self.css.Hb[0])
        p = self.css.l  # ring size
        scale_x, scale_y = self.scale
        if typ == "X":
            assert idx < ra*rb*p, "X check index out of range"
            x = idx // (rb*p)
            y = (idx % (rb*p))
            return ((y + ca)*scale_x, x*scale_y)
        elif typ == "Z":
            assert idx < ca*cb*p, "Z check index out of range"
            x = idx // (cb*p)
            y = (idx % (cb*p))
            return (x*scale_x, (y + rb*p)*scale_y)
        else:
            raise ValueError("typ must be 'X' or 'Z'")


class CCCode(CodePatch):
    # https://arxiv.org/pdf/2603.05398
    pass

# ---------------------------------------------------------------------------
# BBCode adapter
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
        super().__init__()
        self._bb = bb
        self._name = patch_name
        n = bb.n
        self._data_qubits = [Qubit(patch_name, i) for i in range(n)]
        self._x_checks = self._matrix_to_checks(bb.Hx, "X")
        self._z_checks = self._matrix_to_checks(bb.Hz, "Z")
        # logicals은 load_logicals() 이후에 채워짐
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
        """BBCode.set_missing_logicals()로 logical을 채운 뒤 Qubit 집합으로 변환."""
        self._bb.set_missing_logicals()
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

    def name(self): return self._name
    def data_qubits(self): return self._data_qubits
    def x_checks(self): return self._x_checks
    def z_checks(self): return self._z_checks
    def x_logicals(self): return self._x_logicals
    def z_logicals(self): return self._z_logicals


class AncillaGraph(CodePatch):
    def __init__(self, graph: nx.Graph, patch_name: str = "ancilla_graph", type: str = "X", offset=(0.0, 0.0), scale=(1,1)):
        # scale = (scale_x, scale_y)은 graph의 vertex들의 min max x좌표 차이가 scale_x이 되도록,
        # min max y좌표 차이도 scale_y이 되도록
        
        super().__init__(offset)
        self._name = patch_name
        self._graph = graph
        self.scale = scale
        self.edge_index = {edge: i for i, edge in enumerate(graph.edges())}
        # 역방향도 처리
        self.edge_index.update({(v, u): i for (u, v), i in self.edge_index.items()})
        self.vertex_local_coords = {}
        self.edge_local_coords = {}
        self.assign_vertex_local_coords()
        # edge는 vertex 두개의 중점
        for u, v in graph.edges():
            x = (self.vertex_local_coords[u][0] + self.vertex_local_coords[v][0]) / 2
            y = (self.vertex_local_coords[u][1] + self.vertex_local_coords[v][1]) / 2
            self.edge_local_coords[self.edge_index[(u, v)]] = (x, y)
        assert type in ("X", "Z"), "type must be 'X' or 'Z'"
        # vertex가 check에 해당, edge는 qubit
        self._data_qubits = [Qubit(patch_name, i, self.edge_local_coords[i]) for i in range(graph.number_of_edges())]
        if type == "X":
            self._x_checks = []
            self._z_checks = []
            for i, v in enumerate(graph.nodes()):
                # node v를 포함하는 edge들
                support = frozenset([self._data_qubits[self.edge_index[edge]] for edge in graph.edges(v)])

                self._x_checks.append(Check(support, "X", f"X_{v}"))

            # cycle basis에 해당하는게 z check
            cb = self.cycle_edge_basis()
            for cycle in cb:
                support = frozenset([self._data_qubits[edge] for edge in cycle])
                self._z_checks.append(Check(support, "Z", f"Z_cycle_{cycle}"))
        else:
            self._x_checks = []
            self._z_checks = []
            raise NotImplementedError("AncillaGraph with Z checks is not implemented yet")
    
    def assign_vertex_local_coords(self,):
        pos = nx.spring_layout(self._graph, seed=42)  # spring layout으로 vertex 좌표 계산
        # scale 조절
        minx, maxx = min(x for x, y in pos.values()), max(x for x, y in pos.values())
        miny, maxy = min(y for x, y in pos.values()), max(y for x, y in pos.values())
        scale_x, scale_y = self.scale
        if minx==maxx:
            maxx = minx + 1
        if miny==maxy:
            maxy = miny + 1
        transform = lambda x, y: (
            float((x - minx) / (maxx - minx) * scale_x),
            float((y - miny) / (maxy - miny) * scale_y)
        )
        for v, (x, y) in pos.items():
            self.vertex_local_coords[v] = transform(x, y)

    def cycle_edge_basis(self,):
        cycles_v = nx.cycle_basis(self._graph)
        
        cycles_e = []
        for cycle in cycles_v:
            edges = []
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                edges.append(self.edge_index[(u, v)])
            cycles_e.append(edges)
        
        return cycles_e
    
    def get_perfect_matching(self, V_s):
        # self._graph의 vertex subset V_s에 대해 perfect matching을 찾음.
        # 각 matching의 edge weight들의 최댓값을 minimize
        k = len(V_s)

        def bfs(src):
            dist = {src: 0}
            prev = {src: None}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self._graph.neighbors(u):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
            return dist, prev

        def get_path(prev, src, dst):
            path = []
            u = dst
            while u != src:
                path.append(u)
                u = prev[u]
            path.append(src)
            path.reverse()
            return path

        D = []
        PREV = []
        for u in V_s:
            d, p = bfs(u)
            D.append([d[v] for v in V_s])
            PREV.append(p)

        INF = float('inf')
        dp = [INF] * (1 << k)
        dp[0] = 0
        parent = [None] * (1 << k)

        for mask in range((1 << k) - 1):
            if dp[mask] == INF:
                continue
            i = next(b for b in range(k) if not (mask >> b & 1))
            for j in range(i+1, k):
                if not (mask >> j & 1):
                    new_mask = mask | (1 << i) | (1 << j)
                    val = max(dp[mask], D[i][j])
                    if val < dp[new_mask]:
                        dp[new_mask] = val
                        parent[new_mask] = (i, j)

        matching = []
        mask = (1 << k) - 1
        while mask:
            i, j = parent[mask]
            path = get_path(PREV[i], V_s[i], V_s[j])
            edges = list(zip(path, path[1:]))
            matching.append((V_s[i], V_s[j], edges))
            mask ^= (1 << i) | (1 << j)

        return matching
    
    def name(self): return self._name
    def data_qubits(self): return self._data_qubits
    def x_checks(self): return self._x_checks
    def z_checks(self): return self._z_checks
    def local_coord(self, q):
        x, y = q.local_coord
        return (x * self.scale[0], y * self.scale[1])
        