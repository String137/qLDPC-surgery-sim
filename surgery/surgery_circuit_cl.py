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
from core.codes import RotatedSurfaceCode
from surgery.surgery_circuit import SurgeryCircuitBuilder


def edge_color_restricted(checks: list[Check], restrict_qubits: set) -> dict:
    check_used: dict[int, set] = defaultdict(set)
    qubit_used: dict[Qubit, set] = defaultdict(set)
    coloring: dict = {}
    for ci, chk in enumerate(checks):
        for q in sorted(chk.qubits, key=lambda x: x.idx):
            if q not in restrict_qubits:
                continue
            used = check_used[ci] | qubit_used[q]
            c = 0
            while c in used:
                c += 1
            coloring[(ci, q)] = c
            check_used[ci].add(c)
            qubit_used[q].add(c)
    return coloring


def n_colors(col: dict) -> int:
    return max(col.values()) + 1 if col else 0


def cb_left(q: Qubit, d: int, global_col_offset: int) -> bool:
    row, col = divmod(q.idx, d)
    return (col + global_col_offset + row) % 2 == 0


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
            print([b._data_coord(q) for q in chk.qubits])
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
        if extra_z:
            self.circuit.append("R", extra_z)
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