"""
surgery_circuit_cl.py -- ZZ surgery circuit-level noise (LRC scheduling)
LRC order per Strikis, Browne, Beverland (arXiv:2603.05481):
  One round: LZ -> RZ+LX -> RX -> H -> M -> R+H

Checkerboard partition (global coords):
  patch1: col 0..d-1
  ancilla: col d+1
  patch2: col d+3..2d+2
  left if (global_col + row) % 2 == 0 else right
"""

from __future__ import annotations
import stim
import tqecd
import numpy as np
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
        p2_left  = {q for q in patch2.data_qubits() if cb_left(q, d, d+3)}
        p2_right = {q for q in patch2.data_qubits() if not cb_left(q, d, d+3)}
        anc_col  = d + 1
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

    def _emit_round(self, c: stim.Circuit,
                    zchk, xchk, syn_z, syn_x,
                    col_LZ, col_RZ, col_LX, col_RX,
                    p: float, add_data_noise: bool = True):
        """LZ -> RZ+LX -> RX -> H -> M -> R+H"""
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
        # phenomenological: data qubit depolarization between rounds (skip after last round)
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
        for _ in range(self.rounds):
            self._emit_round(
                self.circuit,
                self.b.zchk_p1 + self.b.zchk_p2,
                self.b.xchk_p1 + self.b.xchk_p2,
                self.syn_z_pp, self.syn_x_pp,
                self.col_LZ_pp, self.col_RZ_pp,
                self.col_LX_pp, self.col_RX_pp,
                p=self.p,
            )

    def _phase_merge(self):
        b = self.b
        all_data = self._all_data() + list(b.ancilla.data_qubits())

        # init ancilla data qubits (seam bridge)
        anc_data_idx = [self.reg[q] for q in b.ancilla.data_qubits()]
        self.circuit.append("RX", anc_data_idx)
        self.circuit.append("TICK")

        # extra Z ancilla for seam checks
        extra_z = [i for i in self.syn_z_m if i not in set(self.syn_z_pp)]
        if extra_z:
            self.circuit.append("R", extra_z)
            self.circuit.append("TICK")

        # track seam Z meas indices
        n_zpp = len(self.b.zchk_p1 + self.b.zchk_p2)
        n_seam = len(self.b.seam_z_checks)
        seam_anc_set = frozenset(self.syn_z_m[n_zpp: n_zpp + n_seam])

        all_seam: list[list[int]] = []
        for rnd in range(self.rounds):
            self._emit_round(
                self.circuit,
                b.zchk_merge, b.xchk_merge,
                self.syn_z_m, self.syn_x_m,
                self.col_LZ_m, self.col_RZ_m,
                self.col_LX_m, self.col_RX_m,
                p=self.p,
                add_data_noise=True,
            )
            # collect seam meas indices from this round
            mc = 0
            cur = []
            for inst in self.circuit.flattened():
                if inst.name in ('M', 'MX'):
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
        # noiseless final round to catch errors from last DEPOLARIZE1
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

    def _add_coords(self):
        b = self.b; d = self.d; c = self.circuit
        x_anc = float(d) + 2.0
        x_p2  = x_anc + 1.0 + 2.0
        for q in self.patch1.data_qubits():
            r, col = divmod(q.idx, d)
            c.append("QUBIT_COORDS", [self.reg[q]], [float(col), float(r)])
        for q in b.ancilla.data_qubits():
            c.append("QUBIT_COORDS", [self.reg[q]], [x_anc, float(q.idx)])
        for q in self.patch2.data_qubits():
            r, col = divmod(q.idx, d)
            c.append("QUBIT_COORDS", [self.reg[q]], [x_p2 + col, float(r)])
        # syndrome ancilla coords - all unique indices
        seen = set()
        zpp = b.zchk_p1 + b.zchk_p2
        xpp = b.xchk_p1 + b.xchk_p2
        for chk_list, idx_list, x_off in [
            (b.zchk_merge, self.syn_z_m, 0.0),
            (b.xchk_merge, self.syn_x_m, 0.05),
            (zpp, self.syn_z_pp, 0.0),
            (xpp, self.syn_x_pp, 0.05),
        ]:
            for chk, idx in zip(chk_list, idx_list):
                if idx in seen: continue
                seen.add(idx)
                coords = [b._data_coord(q) for q in chk.qubits]
                cx = sum(co[0] for co in coords) / len(coords)
                cy = sum(co[1] for co in coords) / len(coords)
                c.append("QUBIT_COORDS", [idx], [cx + x_off, cy])

    def build(self) -> tuple[stim.Circuit, stim.Circuit]:
        self._add_coords()
        self._initialize()
        self._phase_pre()
        self._phase_merge()
        self._phase_split_post()
        self._phase_final_readout()

        bare = self.circuit

        # annotate with tqecd
        ann = tqecd.annotate_detectors_automatically(bare)

        # add observables
        n_m = bare.num_measurements
        p1d = list(self.patch1.data_qubits())
        p2d = list(self.patch2.data_qubits())
        fs  = n_m - len(p1d) - len(p2d)
        p1pos = {q: i for i, q in enumerate(p1d)}
        p2pos = {q: i for i, q in enumerate(p2d)}
        lz1 = self.patch1.z_logicals()[0]
        lz2 = self.patch2.z_logicals()[0]

        if self._seam_z_last_meas:
            ann.append("OBSERVABLE_INCLUDE",
                       [stim.target_rec(j - n_m) for j in self._seam_z_last_meas], 0)
        ann.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(fs + p1pos[q] - n_m) for q in lz1 if q in p1pos], 1)
        ann.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(fs + len(p1d) + p2pos[q] - n_m) for q in lz2 if q in p2pos], 2)

        return bare, ann


def build_zz_surgery_circuit_cl(d: int, rounds: int, p: float):
    return SurgeryCircuitCL(d=d, rounds=rounds, p=p).build()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",      type=int,   default=3)
    parser.add_argument("--rounds", type=int,   default=None)
    parser.add_argument("--p",      type=float, default=1e-3)
    parser.add_argument("--out",    type=str,   default=None,
                        help="stim 파일 저장 경로 (기본: zz_surgery_lrc_d{d}.stim)")
    args = parser.parse_args()

    d      = args.d
    rounds = args.rounds if args.rounds is not None else d
    p      = args.p
    out    = args.out if args.out else f"zz_surgery_lrc_d{d}.stim"

    print(f"d={d}, rounds={rounds}, p={p}")
    bare, ann = build_zz_surgery_circuit_cl(d=d, rounds=rounds, p=p)
    print(f"  detectors: {ann.num_detectors}, observables: {ann.num_observables}")
    shots = ann.without_noise().compile_detector_sampler().sample(64)
    print(f"  noiseless no-fire: {(shots==0).all()}")
    dem = ann.detector_error_model(decompose_errors=False)
    print(f"  circuit distance: {len(dem.shortest_graphlike_error())}")
    with open(out, "w") as f:
        f.write(str(ann))
    print(f"  saved: {out}")