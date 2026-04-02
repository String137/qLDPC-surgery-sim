"""
surgery_circuit_cl.py -- ZZ surgery circuit-level noise (LRC scheduling)
LRC order per Strikis, Browne, Beverland (arXiv:2603.05481):
  One round: LZ -> RZ+LX -> RX -> H -> M

Checkerboard partition (global coords):
  patch1: col 0..d-1
  patch2: col d+1..2d
  ancilla: col d
  left if (global_col + row) % 2 == 0 else right
"""

from __future__ import annotations
import stim
import tqecd
import argparse
from collections import defaultdict
from core.primitives import Qubit, Check, QubitRegistry
from core.codes import CodePatch, EmptyPatch, SurfaceCode
import numpy as np
from collections import defaultdict
# from surgery.surgery_circuit import SurgeryCircuitBuilder

# surgery_circuit.py는 안 쓰는 방향으로..

# Single-code memory experiments 예시부터 들어보자.
# LRC 방식으로 구현할 거임.

class LRCCircuitBuilder:
    def __init__(self, patch: CodePatch, rounds: int, p_data: float, p_meas: float):
        self.patch = patch
        self.rounds = rounds
        self.p_data = p_data
        self.p_meas = p_meas
        self.left = None
        self.right = None
        self.check_idx = {chk: i for i, chk in enumerate(patch.z_checks() + patch.x_checks())}
        self.ancilla_coord = None # syndrome meas를 위한 ancilla들의 좌표. List의 idx번째 원소가 좌표.

    def find_partition(self,):
        raise NotImplementedError
    
    def set_partition(self, left: list[int]):
        self.left = left
        self.right = [i for i in range(self.patch.n_qubits()) if i not in left]

    def set_check_idx(self, check_idx: dict):
        self.check_idx = check_idx
    
    def set_ancilla_coord(self, ancilla_coord: list[tuple[float, float]]):
        self.ancilla_coord = ancilla_coord
    def set_default_ancilla_coord(self,):
        # check_idx에 따라 ancilla_coord 자동으로 설정
        ancilla_coord = [None] * len(self.check_idx)
        for chk, idx in self.check_idx.items():
            if self.patch.check2coord is None:
                raise ValueError("patch.check2coord is not set. Cannot set default ancilla coordinates. Please set patch.check2coord first.")
            coord = self.patch.check2coord[chk]
            ancilla_coord[idx] = coord
        self.ancilla_coord = ancilla_coord

    def edge_coloring(self, support: list[Qubit]|list[int], pauli: str, existing_coloring: dict | None = None) -> dict:
        # only for bipartite graph
        assert pauli in ("X", "Z")
        chk_mat = self.patch.h_x() if pauli == "X" else self.patch.h_z()
        support_idx = (
            [self.patch.get_idx(qubit=q) for q in support]
            if isinstance(support[0], Qubit) else list(support)
        )
        chk_mat_restricted = chk_mat[:, support_idx]
        print(chk_mat_restricted)
        n_checks, n_qubits = chk_mat_restricted.shape

        edges = [
            (i, j)
            for i in range(n_checks)
            for j in range(n_qubits)
            if chk_mat_restricted[i, j]
        ]
        edge_index = {(i, j): idx for idx, (i, j) in enumerate(edges)}

        color = [-1] * len(edges)
        check_colors = [dict() for _ in range(n_checks)]  # check_colors[i] = {c: local_j}
        qubit_colors  = [dict() for _ in range(n_qubits)]  # qubit_colors[j] = {c: check_i}

        # 기존 coloring 로드
        support_inv = {orig: local for local, orig in enumerate(support_idx)}
        if existing_coloring:
            for c, edge_list in existing_coloring.items():
                for (i, orig_j) in edge_list:
                    if orig_j not in support_inv:
                        continue
                    j = support_inv[orig_j]
                    if (i, j) not in edge_index:
                        continue
                    eidx = edge_index[(i, j)]
                    color[eidx] = c
                    check_colors[i][c] = j
                    qubit_colors[j][c] = i

        existing_max = max(existing_coloring.keys()) + 1 if existing_coloring else 0
        raw_delta = max(
            int(chk_mat_restricted.sum(axis=1).max()) if n_checks else 0,
            int(chk_mat_restricted.sum(axis=0).max()) if n_qubits else 0,
        )
        delta = max(raw_delta, existing_max) + 1

        def find_free_color(used: dict) -> int:
            for c in range(delta + 1):
                if c not in used:
                    return c

        def augment(start_check, c1, c2):
            # c1-c2 augmenting path를 먼저 수집
            path = []  # (check_i, qubit_j, current_color) 순서로
            u = start_check
            visited = set()
            while True:
                if u in visited:
                    break
                visited.add(u)
                if c2 not in check_colors[u]:
                    break
                j = check_colors[u][c2]
                path.append((u, j, c2))  # edge (u,j)의 색을 c2 → c1으로
                if c1 not in qubit_colors[j]:
                    break
                i = qubit_colors[j][c1]
                path.append((i, j, c1))  # edge (i,j)의 색을 c1 → c2로
                u = i

            # 수집한 path를 한꺼번에 swap
            for u, j, old_c in path:
                new_c = c1 if old_c == c2 else c2
                eidx = edge_index[(u, j)]
                color[eidx] = new_c
                del check_colors[u][old_c]
                check_colors[u][new_c] = j
                del qubit_colors[j][old_c]
                qubit_colors[j][new_c] = u

        for idx, (i, j) in enumerate(edges):
            if color[idx] != -1:
                continue
            c1 = find_free_color(check_colors[i])
            c2 = find_free_color(qubit_colors[j])
            if c1 == c2:
                color[idx] = c1
                check_colors[i][c1] = j
                qubit_colors[j][c1] = i
            else:
                augment(i, c1, c2)
                # augment 후 i에서 c2가 비어있음이 보장됨
                color[idx] = c2
                check_colors[i][c2] = j
                qubit_colors[j][c2] = i

        result = defaultdict(list)
        for idx, (i, j) in enumerate(edges):
            result[color[idx]].append((i, support_idx[j]))
        return dict(result)
            
        
    def build(self):

        # TODO: 아래는 build_round만 짠거임.
        # 1. build_init(): 데이터 qubit |0> 준비 후 LRC cycle 1회 실행
        #    - 첫 syndrome을 reference syndrome으로 사용 (모두 +1 아닐 수 있음)
        #    - logical |0> 초기화 시 logical Z measurement 포함
        # 2. build_round(): 한 라운드 LRC EC 회로 (Z/X ancilla 엇갈린 스케줄)
        #    - rounds번 수동으로 이어붙임 (파이프라인 구조라 단순 *n 불가)
        # 3. build_final(): 마지막 라운드 마무리 + logical observable 설정
        # 4. 셋 이어붙여서 반환

        if self.left is None or self.right is None:
            raise ValueError("Please set partition")
        if self.ancilla_coord is None:
            try:
                self.set_default_ancilla_coord()
            except ValueError as e:
                raise ValueError("Ancilla coordinates are not set. Please set them manually or ensure patch.check2coord is set.") from e

        left = self.left
        right = self.right

        n_data = self.patch.n_qubits()
        n_z = len(self.patch.z_checks())
        n_x = len(self.patch.x_checks())

        # qubit 번호 계산
        def z_ancilla(check_i): return n_data + check_i
        def x_ancilla(check_i): return n_data + n_z + check_i
        
        z_ancilla_qubits = list(range(n_data, n_data + n_z))
        x_ancilla_qubits = list(range(n_data + n_z, n_data + n_z + n_x))

        # edge coloring
        lz_coloring = self.edge_coloring(support=left, pauli="Z")
        lx_coloring = self.edge_coloring(support=left, pauli="X")
        rz_coloring = self.edge_coloring(support=right, pauli="Z")
        rx_coloring = self.edge_coloring(support=right, pauli="X")

        tlz = max(lz_coloring.keys()) + 1 if lz_coloring else 0
        tlx = max(lx_coloring.keys()) + 1 if lx_coloring else 0
        trz = max(rz_coloring.keys()) + 1 if rz_coloring else 0
        trx = max(rx_coloring.keys()) + 1 if rx_coloring else 0
        t1 = max(tlx, trz)
        t2 = max(tlz, trx)

        t_per_round = t1 + t2 + 2

        # t=0: Z ancilla reset
        # t=1~t2: LZ CNOT (data → Z ancilla)
        # t=t2: X ancilla reset (X basis)
        # t=t2+1...t2+t1에는 LX (ancilla → data) + RZ (data → ancilla) CNOT
        # t=t2+t1+1에는 Z ancilla measure
        # t=t2+t1+1...t2+t1+t2에는 RX (ancilla → data) CNOT
        # t=t2+t1+t2+1에는 X ancilla measure

        circuit = stim.Circuit()

        for i in range(n_data):
            coord = self.patch.local_coord(self.patch._get_qubit_from_idx(i))
            circuit.append("QUBIT_COORDS", [i], coord)
        for check_i in range(n_z):
            coord = self.ancilla_coord[check_i]
            circuit.append("QUBIT_COORDS", [z_ancilla(check_i)], coord)
        for check_i in range(n_x):
            coord = self.ancilla_coord[n_z + check_i]
            circuit.append("QUBIT_COORDS", [x_ancilla(check_i)], coord)
        schedule = {}

        def add(t, gate, targets):
            schedule.setdefault(t, []).append((gate, targets))

        # t=0: Z ancilla reset
        add(0, "R", z_ancilla_qubits)

        # t=1~t2: LZ CNOT (data → Z ancilla), 
        for t, edges in lz_coloring.items():
            for (check_i, qubit_j) in edges:
                add(t + 1, "CX", [qubit_j, z_ancilla(check_i)])

        # X ancilla reset at t=t2+1
        add(t2 + 1, "RX", x_ancilla_qubits)

        for round in range(self.rounds-1):
            # Lx+Rz CNOT부터 ... Lz CNOT, X ancilla reset까지 반복
            # LX CNOT (X ancilla → data), t=t2+2 ~ t2+1+t1
            for t, edges in lx_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(round * t_per_round + t2 + 2 + t, "CX", [x_ancilla(check_i), qubit_j])

            # RZ CNOT (data → Z ancilla), 같은 구간
            for t, edges in rz_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(round * t_per_round + t2 + 2 + t, "CX", [qubit_j, z_ancilla(check_i)])
            # RX CNOT (X ancilla → data)
            for t, edges in rx_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(round * t_per_round + t2 + t1 + 3 + t, "CX", [x_ancilla(check_i), qubit_j])
            # Z ancilla measure & reset
            add(round * t_per_round + t2 + t1 + 3, "M", z_ancilla_qubits)
            add(round * t_per_round + t2 + t1 + 4, "R", z_ancilla_qubits)
            # LZ CNOT (data → Z ancilla)
            for t, edges in lz_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(round * t_per_round + t2 + t1 + 5 + t, "CX", [qubit_j, z_ancilla(check_i)])
            # X ancilla measure & reset
            add(round * t_per_round + 2*t2 + t1 + 3, "M", x_ancilla_qubits)
            add(round * t_per_round + 2*t2 + t1 + 4, "RX", x_ancilla_qubits)

        # LX CNOT (X ancilla → data), 
        for t, edges in lx_coloring.items():
            for (check_i, qubit_j) in edges:
                add(self.rounds * t_per_round + t2 + 2 + t, "CX", [x_ancilla(check_i), qubit_j])

        # RZ CNOT (data → Z ancilla), 같은 구간
        for t, edges in rz_coloring.items():
            for (check_i, qubit_j) in edges:
                add(self.rounds * t_per_round + t2 + 2 + t, "CX", [qubit_j, z_ancilla(check_i)])

        # Z ancilla measure
        add(self.rounds * t_per_round + t2 + t1 + 2, "M", z_ancilla_qubits)

        # RX CNOT (X ancilla → data)
        for t, edges in rx_coloring.items():
            for (check_i, qubit_j) in edges:
                add(self.rounds * t_per_round + t2 + t1 + 2 + t, "CX", [x_ancilla(check_i), qubit_j])

        # X ancilla measure
        add(self.rounds * t_per_round + 2*t2 + t1 + 2, "MX", x_ancilla_qubits)

        # apply
        for t in sorted(schedule.keys()):
            for (gate, targets) in schedule[t]:
                circuit.append(gate, targets)
            circuit.append("TICK", [])
        print(circuit)
        with open("debug_circuit.svg", "w") as f:
            f.write(str(circuit.diagram("timeslice-svg")))
        # with open("debug_circuit.html", "w") as f:
        #     f.write(circuit.diagram("timeline-3d")._repr_html_())
        return circuit

                 
class SurgeryCircuitCL:
    def __init__(self, d: int, rounds: int, p: float):
        self.d = d
        self.rounds = rounds
        self.p = p

        patch1 = RotatedSurfaceCode(d, patch_name="patch1")
        patch2 = RotatedSurfaceCode(d, patch_name="patch2")
        self.patch1 = patch1
        self.patch2 = patch2

        b = SurgeryCircuitBuilder(patch1, patch2, rounds=rounds, p_data=0, p_meas=0)
        self.b = b
        self.reg = b.reg

        # checkerboard partition
        p1_left  = {q for q in patch1.data_qubits() if cb_left(q, d, 0)}
        p1_right = {q for q in patch1.data_qubits() if not cb_left(q, d, 0)}
        p2_left  = {q for q in patch2.data_qubits() if cb_left(q, d, d+1)}
        p2_right = {q for q in patch2.data_qubits() if not cb_left(q, d, d+1)}
        anc_col  = d
        anc_left  = {q for q in b.ancilla.data_qubits() if (q.idx + anc_col) % 2 == 0}
        anc_right = {q for q in b.ancilla.data_qubits() if (q.idx + anc_col) % 2 == 1}

        self.left_pp  = p1_left  | p2_left
        self.right_pp = p1_right | p2_right
        self.left_m   = p1_left  | p2_left  | anc_left
        self.right_m  = p1_right | p2_right | anc_right

        # syndrome ancilla indices
        zpp = b.zchk_p1 + b.zchk_p2
        xpp = b.xchk_p1 + b.xchk_p2
        zm  = b.zchk_merge
        xm  = b.xchk_merge

        self.syn_z_pp = b.syn_anc_z[:len(zpp)]
        self.syn_x_pp = b.syn_anc_x[:len(xpp)]
        self.syn_z_m  = b.syn_anc_z[:len(zm)]
        self.syn_x_m  = b.syn_anc_x[:len(xm)]

        # --- Unified 3-step mapping: check -> coord -> ancilla stim idx ---
        checks = zpp + xpp + zm + xm
        syns = self.syn_z_pp + self.syn_x_pp + self.syn_z_m + self.syn_x_m
        self.check_to_coord = {}
        ############## DEBUG ##############
        for chk in checks:
            # qubit coords in chk
            print([b._data_coord(q) for q in chk.qubits], chk)
        ####################################
        for chk in checks:
            if len(chk.qubits) == 4:
                coords = [b._data_coord(q) for q in chk.qubits]
                cx = sum(x for x, y in coords) / 4
                cy = sum(y for x, y in coords) / 4
                self.check_to_coord[chk] = (cx, cy)
            elif len(chk.qubits) == 2:
                coords = [b._data_coord(q) for q in chk.qubits]
                (x0, y0), (x1, y1) = coords
                # left boundary: x0 == x1 == 0
                if x0 == 0 and x1 == 0:
                    cx = -0.5
                    cy = (y0 + y1) / 2
                # right boundary: x0 == x1 == 2d (patch2 rightmost)
                elif x0 == x1 == 2 * d:
                    cx = x0 + 0.5
                    cy = (y0 + y1) / 2
                # top boundary: y0 == y1 == 0
                elif y0 == y1 == 0:
                    cx = (x0 + x1) / 2
                    cy = -0.5
                # bottom boundary: y0 == y1 == d-1
                elif y0 == y1 == d - 1:
                    cx = (x0 + x1) / 2
                    cy = y0 + 0.5
                # 안쪽 seam: x0 == x1 == d-1
                elif x0 == x1 == d - 1:
                    cx = x0 + 0.5
                    cy = (y0 + y1) / 2
                # 안쪽 seam: x0 == x1 == d+1
                elif x0 == x1 == d + 1:
                    cx = x0 - 0.5
                    cy = (y0 + y1) / 2
                else:
                    raise ValueError(f"Unsupported check weight: {len(chk.qubits)} (qubits={[f'({q.patch},{q.idx})' for q in chk.qubits]}), coords: {coords}")
                self.check_to_coord[chk] = (cx, cy)
            else:
                raise ValueError(f"Unsupported check weight: {len(chk.qubits)} (qubits={[f'({q.patch},{q.idx})' for q in chk.qubits]})")
        ############## DEBUG ##############
        for chk in checks:
            print(self.check_to_coord[chk], [f"({q.patch},{q.idx})" for q in chk.qubits])
        ####################################
        self.coord_to_ancilla = {}
        for chk, idx in zip(checks, syns):
            coord = self.check_to_coord[chk]
            if coord not in self.coord_to_ancilla:
                self.coord_to_ancilla[coord] = idx

        self.check_to_ancilla = {chk: self.coord_to_ancilla[self.check_to_coord[chk]] for chk in self.check_to_coord}

        # edge colorings pre/post
        self.col_LZ_pp = edge_color_restricted(zpp, self.left_pp)
        self.col_RZ_pp = edge_color_restricted(zpp, self.right_pp)
        self.col_LX_pp = edge_color_restricted(xpp, self.left_pp)
        self.col_RX_pp = edge_color_restricted(xpp, self.right_pp)

        # edge colorings merge
        self.col_LZ_m  = edge_color_restricted(zm, self.left_m)
        self.col_RZ_m  = edge_color_restricted(zm, self.right_m)
        self.col_LX_m  = edge_color_restricted(xm, self.left_m)
        self.col_RX_m  = edge_color_restricted(xm, self.right_m)

        self._seam_z_last_meas: list[int] = []
        self.circuit = stim.Circuit()

    # ------------------------------------------------------------------

    def _all_data(self):
        return list(self.patch1.data_qubits()) + list(self.patch2.data_qubits())

    def _all_data_idx(self):
        return [self.reg[q] for q in self._all_data()]

    def _add_coords(self):
        b = self.b; d = self.d; c = self.circuit
        x_anc = float(d)
        for q in self.patch1.data_qubits():
            r, col = divmod(q.idx, d)
            c.append("QUBIT_COORDS", [self.reg[q]], [float(col + 1), float(r + 1)])
        for q in b.ancilla.data_qubits():
            c.append("QUBIT_COORDS", [self.reg[q]], [x_anc + 1, float(q.idx + 1)])
        for q in self.patch2.data_qubits():
            r, col = divmod(q.idx, d)
            c.append("QUBIT_COORDS", [self.reg[q]], [float(col + d + 2), float(r + 1)])
        # syndrome ancilla coords (robust: emit for all unique ancilla indices)
        seen = set()
        # Z ancilla
        for chk, idx in self.check_to_ancilla.items():
            if idx in seen:
                continue
            seen.add(idx)
            cx, cy = self.check_to_coord[chk]
            c.append("QUBIT_COORDS", [idx], [cx + 1, cy + 1])

    def _emit_round(self, c: stim.Circuit,
                    zchk, xchk, syn_z, syn_x,
                    col_LZ, col_RZ, col_LX, col_RX,
                    p: float, add_data_noise: bool = True):
        """LZ -> RZ+LX -> RX -> H -> M"""
        c.append("R",  syn_z + syn_x)
        c.append("H",  syn_x)
        c.append("TICK")

        for (cz, cx) in [(col_LZ, {}), (col_RZ, col_LX), ({}, col_RX)]:
            nsteps = max(n_colors(cz), n_colors(cx))
            for color in range(nsteps):
                pairs = ([(self.reg[q], syn_z[ci])
                          for ci, chk in enumerate(zchk)
                          for q in sorted(chk.qubits, key=lambda x: x.idx)
                          if cz.get((ci, q)) == color] +
                         [(syn_x[ci], self.reg[q])
                          for ci, chk in enumerate(xchk)
                          for q in sorted(chk.qubits, key=lambda x: x.idx)
                          if cx.get((ci, q)) == color])
                for a, b in pairs:
                    c.append("CX", [a, b])
                c.append("TICK")

        c.append("H", syn_x)
        c.append("TICK")
        if p > 0:
            c.append("X_ERROR", syn_z + syn_x, p)
        c.append("M", syn_z + syn_x)
        c.append("TICK")
        if p > 0 and add_data_noise:
            all_data_idx = [self.reg[q] for q in self._all_data()]
            c.append("DEPOLARIZE1", all_data_idx, p)
            c.append("TICK")

    def _initialize(self):
        self.circuit.append("R", self._all_data_idx())
        self.circuit.append("TICK")
        # noiseless init round
        self._emit_round(
            self.circuit,
            self.b.zchk_p1 + self.b.zchk_p2,
            self.b.xchk_p1 + self.b.xchk_p2,
            self.syn_z_pp, self.syn_x_pp,
            self.col_LZ_pp, self.col_RZ_pp,
            self.col_LX_pp, self.col_RX_pp,
            p=0.0,
        )

    def _phase_pre(self):
        for i in range(self.rounds):
            self._emit_round(
                self.circuit,
                self.b.zchk_p1 + self.b.zchk_p2,
                self.b.xchk_p1 + self.b.xchk_p2,
                self.syn_z_pp, self.syn_x_pp,
                self.col_LZ_pp, self.col_RZ_pp,
                self.col_LX_pp, self.col_RX_pp,
                p=self.p,
                add_data_noise=(i < self.rounds - 1),
            )

    def _phase_merge(self):
        b = self.b

        anc_data_idx = [self.reg[q] for q in b.ancilla.data_qubits()]
        self.circuit.append("RX", anc_data_idx)
        self.circuit.append("TICK")

        extra_z = [i for i in self.syn_z_m if i not in set(self.syn_z_pp)]
        extra_x = [i for i in self.syn_x_m if i not in set(self.syn_x_pp)]
        print("#"*50)
        print(self.syn_x_m)
        print(self.syn_z_m)
        print(extra_x)
        print("#"*50)
        if extra_z:
            self.circuit.append("R", extra_z)
            self.circuit.append("TICK")
        if extra_x:
            self.circuit.append("R", extra_x)
            self.circuit.append("TICK")

        n_zpp  = len(self.b.zchk_p1 + self.b.zchk_p2)
        n_seam = len(self.b.seam_z_checks)
        seam_anc_set = frozenset(self.syn_z_m[n_zpp: n_zpp + n_seam])

        all_seam: list[list[int]] = []
        for _ in range(self.rounds):
            self._emit_round(
                self.circuit,
                b.zchk_merge, b.xchk_merge,
                self.syn_z_m, self.syn_x_m,
                self.col_LZ_m, self.col_RZ_m,
                self.col_LX_m, self.col_RX_m,
                p=self.p,
                add_data_noise=True,
            )
            mc = 0
            cur = []
            for inst in self.circuit.flattened():
                if inst.name == 'M':
                    tgts = inst.targets_copy()
                    for i, t in enumerate(tgts):
                        if t.qubit_value in seam_anc_set:
                            cur.append(mc + i)
                    mc += len(tgts)
            if cur:
                all_seam.append(cur[-n_seam:])

        self._seam_z_last_meas = all_seam[-1] if all_seam else []

    def _phase_split_post(self):
        for _ in range(self.rounds):
            self._emit_round(
                self.circuit,
                self.b.zchk_p1 + self.b.zchk_p2,
                self.b.xchk_p1 + self.b.xchk_p2,
                self.syn_z_pp, self.syn_x_pp,
                self.col_LZ_pp, self.col_RZ_pp,
                self.col_LX_pp, self.col_RX_pp,
                p=self.p,
                add_data_noise=True,
            )
        # noiseless final round
        self._emit_round(
            self.circuit,
            self.b.zchk_p1 + self.b.zchk_p2,
            self.b.xchk_p1 + self.b.xchk_p2,
            self.syn_z_pp, self.syn_x_pp,
            self.col_LZ_pp, self.col_RZ_pp,
            self.col_LX_pp, self.col_RX_pp,
            p=0.0,
            add_data_noise=False,
        )

    def _phase_final_readout(self):
        self.circuit.append("M", self._all_data_idx())
        self.circuit.append("TICK")

    def build(self) -> tuple[stim.Circuit, stim.Circuit]:
        self._add_coords()
        self._initialize()
        self._phase_pre()
        self._phase_merge()
        self._phase_split_post()
        self._phase_final_readout()

        bare = self.circuit
        

        n_m   = bare.num_measurements
        p1d   = list(self.patch1.data_qubits())
        p2d   = list(self.patch2.data_qubits())
        fs    = n_m - len(p1d) - len(p2d)
        p1pos = {q: i for i, q in enumerate(p1d)}
        p2pos = {q: i for i, q in enumerate(p2d)}
        lz1   = self.patch1.z_logicals()[0]
        lz2   = self.patch2.z_logicals()[0]

        ann = None # 기억해: 지금 오류가 많아서 임시로 아래 주석처리한거야.

        # ann  = tqecd.annotate_detectors_automatically(bare)
        # if self._seam_z_last_meas:
        #     ann.append("OBSERVABLE_INCLUDE",
        #                [stim.target_rec(j - n_m) for j in self._seam_z_last_meas], 0)
        # ann.append("OBSERVABLE_INCLUDE",
        #            [stim.target_rec(fs + p1pos[q] - n_m) for q in lz1 if q in p1pos], 1)
        # ann.append("OBSERVABLE_INCLUDE",
        #            [stim.target_rec(fs + len(p1d) + p2pos[q] - n_m) for q in lz2 if q in p2pos], 2)

        return bare, ann


def build_zz_surgery_circuit_cl(d: int, rounds: int, p: float):
    return SurgeryCircuitCL(d=d, rounds=rounds, p=p).build()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",      type=int,   default=3)
    parser.add_argument("--rounds", type=int,   default=None)
    parser.add_argument("--p",      type=float, default=1e-3)
    parser.add_argument("--out",    type=str,   default=None)
    args = parser.parse_args()

    d      = args.d
    rounds = args.rounds if args.rounds is not None else d
    p      = args.p
    out    = args.out if args.out else f"zz_surgery_lrc_d{d}.stim"

    print(f"d={d}, rounds={rounds}, p={p}")
    bare, ann = build_zz_surgery_circuit_cl(d=d, rounds=rounds, p=p)
    # print(f"  detectors: {ann.num_detectors}, observables: {ann.num_observables}")
    # shots = ann.without_noise().compile_detector_sampler().sample(64)
    # print(f"  noiseless no-fire: {(shots==0).all()}")
    # dem = ann.detector_error_model(decompose_errors=False)
    # print(f"  circuit distance: {len(dem.shortest_graphlike_error())}")
    with open(out, "w") as f:
        f.write(str(bare))
    print(f"  saved: {out}")