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
        
    def build_init(self, logical_op: tuple[list[Qubit],str]) -> stim.Circuit:
        # 데이터 qubit |0> 준비 후 LRC cycle 1회 실행
        def z_ancilla(check_i): return self.n_data + check_i
        def x_ancilla(check_i): return self.n_data + self.n_z + check_i

        circuit = stim.Circuit()

        for i in range(self.n_data):
            coord = self.patch.local_coord(self.patch._get_qubit_from_idx(i))
            circuit.append("QUBIT_COORDS", [i], coord)
        for check_i in range(self.n_z):
            coord = self.ancilla_coord[check_i]
            circuit.append("QUBIT_COORDS", [z_ancilla(check_i)], coord)
        for check_i in range(self.n_x):
            coord = self.ancilla_coord[self.n_z + check_i]
            circuit.append("QUBIT_COORDS", [x_ancilla(check_i)], coord)

        circuit += self.build_ec(round=1)

        support, pauli = logical_op
        # logical op meas ("non-destructive")
        if pauli == "Z":
            circuit.append("R", [x_ancilla(0)])  # logical X ancilla idx=0으로 고정
            circuit.append("TICK", [])
            for q in support:
                idx = self.patch.get_idx(qubit=q)
                circuit.append("CNOT", [idx, x_ancilla(0)])  # logical X ancilla idx=0으로 고정
                circuit.append("TICK", [])
            circuit.append("MX", [x_ancilla(0)])
            circuit.append("TICK", [])
        elif pauli == "X":
            circuit.append("R", [z_ancilla(0)])  # logical Z ancilla idx=0으로 고정
            circuit.append("TICK", [])
            for q in support:
                idx = self.patch.get_idx(qubit=q)
                circuit.append("CNOT", [z_ancilla(0), idx])  # logical Z ancilla idx=0으로 고정
                circuit.append("TICK", [])
            circuit.append("M", [z_ancilla(0)])
            circuit.append("TICK", [])
        else:
            raise ValueError(f"Unsupported logical operator type: {pauli}")

        return circuit
    
    def build_ec(self, round: int):
        def z_ancilla(check_i): return self.n_data + check_i
        def x_ancilla(check_i): return self.n_data + self.n_z + check_i
        # t=0: Z ancilla reset
        # t=1~t2: LZ CNOT (data → Z ancilla)
        # t=t2: X ancilla reset (X basis)
        # t=t2+1...t2+t1에는 LX (ancilla → data) + RZ (data → ancilla) CNOT
        # t=t2+t1+1에는 Z ancilla measure
        # t=t2+t1+1...t2+t1+t2에는 RX (ancilla → data) CNOT
        # t=t2+t1+t2+1에는 X ancilla measure

        circuit = stim.Circuit()
        schedule = {}

        def add(t, gate, targets):
            schedule.setdefault(t, []).append((gate, targets))

        # t=0: Z ancilla reset
        add(0, "R", self.z_ancilla_qubits)

        # t=1~t2: LZ CNOT (data → Z ancilla), 
        for t, edges in self.lz_coloring.items():
            for (check_i, qubit_j) in edges:
                add(t + 1, "CX", [qubit_j, z_ancilla(check_i)])

        # X ancilla reset at t=t2+1
        add(self.t2 + 1, "RX", self.x_ancilla_qubits)

        for round in range(self.rounds-1):
            # Lx+Rz CNOT부터 ... Lz CNOT, X ancilla reset까지 반복
            # LX CNOT (X ancilla → data), t=t2+2 ~ t2+1+t1
            for t, edges in self.lx_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(round * self.t_per_round + self.t2 + 2 + t, "CX", [x_ancilla(check_i), qubit_j])

            # RZ CNOT (data → Z ancilla), 같은 구간
            for t, edges in self.rz_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(round * self.t_per_round + self.t2 + 2 + t, "CX", [qubit_j, z_ancilla(check_i)])
            # RX CNOT (X ancilla → data)
            for t, edges in self.rx_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(round * self.t_per_round + self.t2 + self.t1 + 3 + t, "CX", [x_ancilla(check_i), qubit_j])
            # Z ancilla measure & reset
            add(round * self.t_per_round + self.t2 + self.t1 + 3, "M", self.z_ancilla_qubits)
            add(round * self.t_per_round + self.t2 + self.t1 + 4, "R", self.z_ancilla_qubits)
            # LZ CNOT (data → Z ancilla)
            for t, edges in self.lz_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(round * self.t_per_round + self.t2 + self.t1 + 5 + t, "CX", [qubit_j, z_ancilla(check_i)])
            # X ancilla measure & reset
            add(round * self.t_per_round + 2*self.t2 + self.t1 + 3, "MX", self.x_ancilla_qubits)
            add(round * self.t_per_round + 2*self.t2 + self.t1 + 4, "RX", self.x_ancilla_qubits)

        # LX CNOT (X ancilla → data), 
        for t, edges in self.lx_coloring.items():
            for (check_i, qubit_j) in edges:
                add(self.rounds * self.t_per_round + self.t2 + 2 + t, "CX", [x_ancilla(check_i), qubit_j])

        # RZ CNOT (data → Z ancilla), 같은 구간
        for t, edges in self.rz_coloring.items():
            for (check_i, qubit_j) in edges:
                add(self.rounds * self.t_per_round + self.t2 + 2 + t, "CX", [qubit_j, z_ancilla(check_i)])

        # Z ancilla measure
        add(self.rounds * self.t_per_round + self.t2 + self.t1 + 2, "M", self.z_ancilla_qubits)

        # RX CNOT (X ancilla → data)
        for t, edges in self.rx_coloring.items():
            for (check_i, qubit_j) in edges:
                add(self.rounds * self.t_per_round + self.t2 + self.t1 + 2 + t, "CX", [x_ancilla(check_i), qubit_j])

        # X ancilla measure
        add(self.rounds * self.t_per_round + 2*self.t2 + self.t1 + 2, "MX", self.x_ancilla_qubits)

        # apply
        for t in sorted(schedule.keys()):
            for (gate, targets) in schedule[t]:
                circuit.append(gate, targets)
            circuit.append("TICK", [])
        return circuit

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

        self.n_data = self.patch.n_qubits()
        self.n_z = len(self.patch.z_checks())
        self.n_x = len(self.patch.x_checks())

        # qubit 번호 계산
        def z_ancilla(check_i): return self.n_data + check_i
        def x_ancilla(check_i): return self.n_data + self.n_z + check_i
        
        self.z_ancilla_qubits = list(range(self.n_data, self.n_data + self.n_z))
        self.x_ancilla_qubits = list(range(self.n_data + self.n_z, self.n_data + self.n_z + self.n_x))

        # edge coloring
        self.lz_coloring = self.edge_coloring(support=left, pauli="Z")
        self.lx_coloring = self.edge_coloring(support=left, pauli="X")
        self.rz_coloring = self.edge_coloring(support=right, pauli="Z")
        self.rx_coloring = self.edge_coloring(support=right, pauli="X")

        tlz = max(self.lz_coloring.keys()) + 1 if self.lz_coloring else 0
        tlx = max(self.lx_coloring.keys()) + 1 if self.lx_coloring else 0
        trz = max(self.rz_coloring.keys()) + 1 if self.rz_coloring else 0
        trx = max(self.rx_coloring.keys()) + 1 if self.rx_coloring else 0
        self.t1 = max(tlx, trz)
        self.t2 = max(tlz, trx)

        self.t_per_round = self.t1 + self.t2 + 2

        circuit_init = self.build_init(logical_op=(self.patch.data_qubits()[0:3], "Z"))  # logical operator는 일단 dummy로. build_init에서 logical op meas 포함하도록 나중에 수정할 것.
        circuit_ec = self.build_ec(round=round) 
        circuit = circuit_init + circuit_ec

        # observable
        # 전체 data qubit Z로 measure
        circuit.append("M", range(len(self.patch.data_qubits())))
        # init에서 측정한 logical z operator 값과 마지막에 측정한 logical z operator 값 비교.

        with open("debug_circuit.svg", "w") as f:
            f.write(str(circuit.diagram("timeslice-svg")))
        # with open("debug_circuit.html", "w") as f:
        #     f.write(circuit.diagram("timeline-3d")._repr_html_())
        return circuit
