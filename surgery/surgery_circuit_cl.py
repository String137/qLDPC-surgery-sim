"""
surgery_circuit_cl.py -- ZZ surgery circuit-level noise (LRC scheduling)
LRC order per Strikis, Browne, Beverland (arXiv:2603.05481):
  One round: LZ -> RZ+LX -> RX -> H -> M

Phenomenological noise model approximation:
  - Z ancilla reset 시점에 L data qubit에 DEPOLARIZE1(p_data)
  - X ancilla reset 시점에 R data qubit에 DEPOLARIZE1(p_data)
  - ancilla measurement에 p_meas flip error
  - init/final 단계에는 noise 없음
"""

from __future__ import annotations
import stim
from collections import defaultdict
from core.primitives import Qubit, Check, QubitRegistry
from core.codes import CodePatch, EmptyPatch, SurfaceCode
import numpy as np


class LRCCircuitBuilder:
    def __init__(self, patch: CodePatch, rounds: int, p_data: float, p_meas: float):
        self.patch = patch
        self.rounds = rounds
        self.p_data = p_data
        self.p_meas = p_meas
        self.left = None
        self.right = None
        self.check_idx = {chk: i for i, chk in enumerate(patch.z_checks() + patch.x_checks())}
        self.idx_to_check = {i: chk for chk, i in self.check_idx.items()}
        self.ancilla_coord = None
        self.meas_record = []
        self.meas_count = 0

    def find_partition(self):
        raise NotImplementedError

    def set_partition(self, left: list[int]):
        self.left = left
        self.right = [i for i in range(self.patch.n_qubits()) if i not in left]

    def set_check_idx(self, check_idx: dict):
        self.check_idx = check_idx
        self.idx_to_check = {i: chk for chk, i in check_idx.items()}

    def set_ancilla_coord(self, ancilla_coord: list[tuple[float, float]]):
        self.ancilla_coord = ancilla_coord

    def set_default_ancilla_coord(self):
        ancilla_coord = [None] * len(self.check_idx)
        for chk, idx in self.check_idx.items():
            if self.patch.check2coord is None:
                raise ValueError("patch.check2coord is not set.")
            coord = self.patch.check2coord[chk]
            ancilla_coord[idx] = coord
        self.ancilla_coord = ancilla_coord

    def append_meas(self, circuit: stim.Circuit, gate: str, targets: list, label: str = None, noisy: bool = False):
        assert gate in {"M", "MX", "MY", "MZ", "MR", "MRX"}
        current_tick = circuit.num_ticks
        if noisy and self.p_meas > 0:
            circuit.append(gate, targets, self.p_meas)
        else:
            circuit.append(gate, targets)
        for i, tgt in enumerate(targets):
            ancilla_idx = tgt - self.n_data
            check = self.idx_to_check.get(ancilla_idx, None)
            self.meas_record.append({
                "gate": gate,
                "target": tgt,
                "check": check,
                "tick": current_tick,
                "label": label,
                "meas_idx": self.meas_count + i,
            })
        self.meas_count += len(targets)

    def get_rec(self, tick: int = None, check=None, target: int = None, label: str = None) -> stim.GateTarget:
        for entry in self.meas_record:
            if tick is not None and entry["tick"] != tick:
                continue
            if check is not None and entry["check"] != check:
                continue
            if target is not None and entry["target"] != target:
                continue
            if label is not None and entry["label"] != label:
                continue
            return stim.target_rec(entry["meas_idx"] - self.meas_count)
        raise KeyError(f"No measurement found for tick={tick}, check={check}, target={target}, label={label}")

    def get_meas_idx(self, tick: int = None, check=None, target: int = None, label: str = None) -> int:
        for entry in self.meas_record:
            if tick is not None and entry["tick"] != tick:
                continue
            if check is not None and entry["check"] != check:
                continue
            if target is not None and entry["target"] != target:
                continue
            if label is not None and entry["label"] != label:
                continue
            return entry["meas_idx"]
        raise KeyError(f"No measurement found for tick={tick}, check={check}, target={target}, label={label}")

    def edge_coloring(self, support: list[Qubit]|list[int], pauli: str, existing_coloring: dict | None = None) -> dict:
        assert pauli in ("X", "Z")
        chk_mat = self.patch.h_x() if pauli == "X" else self.patch.h_z()
        support_idx = (
            [self.patch.get_idx(qubit=q) for q in support]
            if isinstance(support[0], Qubit) else list(support)
        )
        chk_mat_restricted = chk_mat[:, support_idx]
        n_checks, n_qubits = chk_mat_restricted.shape

        edges = [
            (i, j)
            for i in range(n_checks)
            for j in range(n_qubits)
            if chk_mat_restricted[i, j]
        ]
        edge_index = {(i, j): idx for idx, (i, j) in enumerate(edges)}

        color = [-1] * len(edges)
        check_colors = [dict() for _ in range(n_checks)]
        qubit_colors  = [dict() for _ in range(n_qubits)]

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
            path = []
            u = start_check
            visited = set()
            while True:
                if u in visited:
                    break
                visited.add(u)
                if c2 not in check_colors[u]:
                    break
                j = check_colors[u][c2]
                path.append((u, j, c2))
                if c1 not in qubit_colors[j]:
                    break
                i = qubit_colors[j][c1]
                path.append((i, j, c1))
                u = i
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
                color[idx] = c2
                check_colors[i][c2] = j
                qubit_colors[j][c2] = i

        result = defaultdict(list)
        for idx, (i, j) in enumerate(edges):
            result[color[idx]].append((i, support_idx[j]))
        return dict(result)

    def build_init(self, circuit: stim.Circuit, logical_op: tuple[list[Qubit], str]):
        def z_ancilla(check_i): return self.n_data + check_i
        def x_ancilla(check_i): return self.n_data + self.n_z + check_i

        for i in range(self.n_data):
            coord = self.patch.local_coord(self.patch._get_qubit_from_idx(i))
            circuit.append("QUBIT_COORDS", [i], coord)
        for check_i in range(self.n_z):
            circuit.append("QUBIT_COORDS", [z_ancilla(check_i)], self.ancilla_coord[check_i])
        for check_i in range(self.n_x):
            circuit.append("QUBIT_COORDS", [x_ancilla(check_i)], self.ancilla_coord[self.n_z + check_i])

        circuit.append("R", list(range(self.n_data)))
        circuit.append("TICK", [])

        self.build_ec(circuit, n_rounds=1, noisy=False)

        support, pauli = logical_op
        if pauli == "X":
            anc = x_ancilla(0)
            circuit.append("RX", [anc])
            circuit.append("TICK", [])
            for q in support:
                circuit.append("CX", [anc, self.patch.get_idx(qubit=q)])
            circuit.append("TICK", [])
            self.append_meas(circuit, "MX", [anc], label="init_logical_x", noisy=False)
            circuit.append("TICK", [])
        elif pauli == "Z":
            anc = z_ancilla(0)
            circuit.append("R", [anc])
            circuit.append("TICK", [])
            for q in support:
                circuit.append("CX", [self.patch.get_idx(qubit=q), anc])
            circuit.append("TICK", [])
            self.append_meas(circuit, "M", [anc], label="init_logical_z", noisy=False)
            circuit.append("TICK", [])
        else:
            raise ValueError(f"Unsupported logical operator type: {pauli}")

    def build_ec(self, circuit: stim.Circuit, n_rounds: int = None, noisy: bool = True):
        def z_ancilla(check_i): return self.n_data + check_i
        def x_ancilla(check_i): return self.n_data + self.n_z + check_i

        if n_rounds is None:
            n_rounds = self.rounds

        meas_gates = {"M", "MX", "MY", "MZ", "MR", "MRX"}
        schedule = {}

        def add(t, gate, targets):
            schedule.setdefault(t, []).append((gate, list(targets)))

        add(0, "R", self.z_ancilla_qubits)
        if noisy and self.p_data > 0:
            add(0, "DEPOLARIZE1", self.left)

        for t, edges in self.lz_coloring.items():
            for (check_i, qubit_j) in edges:
                add(t + 1, "CX", [qubit_j, z_ancilla(check_i)])

        add(self.t2, "RX", self.x_ancilla_qubits)
        if noisy and self.p_data > 0:
            add(self.t2, "DEPOLARIZE1", self.right)

        for r in range(n_rounds - 1):
            offset = r * self.t_per_round

            for t, edges in self.lx_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(offset + self.t2 + 1 + t, "CX", [x_ancilla(check_i), qubit_j])

            for t, edges in self.rz_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(offset + self.t2 + 1 + t, "CX", [qubit_j, z_ancilla(check_i)])

            for t, edges in self.rx_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(offset + self.t2 + self.t1 + 1 + t, "CX", [x_ancilla(check_i), qubit_j])

            add(offset + self.t2 + self.t1 + 1, "M", self.z_ancilla_qubits)
            add(offset + self.t2 + self.t1 + 2, "R", self.z_ancilla_qubits)
            if noisy and self.p_data > 0:
                add(offset + self.t2 + self.t1 + 2, "DEPOLARIZE1", self.left)

            for t, edges in self.lz_coloring.items():
                for (check_i, qubit_j) in edges:
                    add(offset + self.t2 + self.t1 + 3 + t, "CX", [qubit_j, z_ancilla(check_i)])

            add(offset + 2*self.t2 + self.t1 + 1, "MX", self.x_ancilla_qubits)
            add(offset + 2*self.t2 + self.t1 + 2, "RX", self.x_ancilla_qubits)
            if noisy and self.p_data > 0:
                add(offset + 2*self.t2 + self.t1 + 2, "DEPOLARIZE1", self.right)

        final_offset = (n_rounds - 1) * self.t_per_round

        for t, edges in self.lx_coloring.items():
            for (check_i, qubit_j) in edges:
                add(final_offset + self.t2 + 3 + t, "CX", [x_ancilla(check_i), qubit_j])

        for t, edges in self.rz_coloring.items():
            for (check_i, qubit_j) in edges:
                add(final_offset + self.t2 + 3 + t, "CX", [qubit_j, z_ancilla(check_i)])

        add(final_offset + self.t2 + self.t1 + 3, "M", self.z_ancilla_qubits)

        for t, edges in self.rx_coloring.items():
            for (check_i, qubit_j) in edges:
                add(final_offset + self.t2 + self.t1 + 3 + t, "CX", [x_ancilla(check_i), qubit_j])

        add(final_offset + 2*self.t2 + self.t1 + 3, "MX", self.x_ancilla_qubits)

        for t in sorted(schedule.keys()):
            for (gate, targets) in schedule[t]:
                if gate in meas_gates:
                    self.append_meas(circuit, gate, targets, noisy=noisy)
                elif gate == "DEPOLARIZE1":
                    circuit.append(gate, targets, self.p_data)
                else:
                    circuit.append(gate, targets)
            circuit.append("TICK", [])

    def build(self):
        if self.left is None or self.right is None:
            raise ValueError("Please set partition")
        if self.ancilla_coord is None:
            try:
                self.set_default_ancilla_coord()
            except ValueError as e:
                raise ValueError("Ancilla coordinates are not set.") from e

        self.meas_record = []
        self.meas_count = 0

        self.n_data = self.patch.n_qubits()
        self.n_z = len(self.patch.z_checks())
        self.n_x = len(self.patch.x_checks())

        self.z_ancilla_qubits = list(range(self.n_data, self.n_data + self.n_z))
        self.x_ancilla_qubits = list(range(self.n_data + self.n_z, self.n_data + self.n_z + self.n_x))

        self.lz_coloring = self.edge_coloring(support=self.left, pauli="Z")
        self.lx_coloring = self.edge_coloring(support=self.left, pauli="X")
        self.rz_coloring = self.edge_coloring(support=self.right, pauli="Z")
        self.rx_coloring = self.edge_coloring(support=self.right, pauli="X")

        tlz = max(self.lz_coloring.keys()) + 1 if self.lz_coloring else 0
        tlx = max(self.lx_coloring.keys()) + 1 if self.lx_coloring else 0
        trz = max(self.rz_coloring.keys()) + 1 if self.rz_coloring else 0
        trx = max(self.rx_coloring.keys()) + 1 if self.rx_coloring else 0
        self.t1 = max(tlx, trz)
        self.t2 = max(tlz, trx)
        self.t_per_round = self.t1 + self.t2 + 2

        circuit = stim.Circuit()

        # TODO: logical_op을 외부에서 받도록 수정 필요
        self.build_init(circuit, logical_op=(self.patch.data_qubits()[0:3], "Z"))

        # EC rounds (noisy)
        self.build_ec(circuit, n_rounds=self.rounds, noisy=True)

        # 마지막 data qubit Z measurement (noise 없음)
        self.append_meas(circuit, "M", list(range(self.n_data)), label="final_data", noisy=False)

        # OBSERVABLE_INCLUDE: init logical meas XOR final logical meas
        # TODO: logical Z support qubit idx 정확히 지정 필요
        logical_z_support_idx = [self.patch.get_idx(q) for q in self.patch.data_qubits()[0:3]]
        init_rec   = self.get_rec(label="init_logical_z")
        final_recs = [self.get_rec(label="final_data", target=q) for q in logical_z_support_idx]
        circuit.append("OBSERVABLE_INCLUDE", [init_rec] + final_recs, 0)

        with open("debug_circuit.svg", "w") as f:
            f.write(str(circuit.diagram("timeslice-svg")))

        return circuit