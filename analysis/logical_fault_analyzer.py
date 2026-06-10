import numpy as np
import pulp
import stim
from core.primitives import Check, Qubit
from surgery.surgery import MergedCodeWithGraph

NOISE_GATES = {'Z_ERROR', 'X_ERROR', 'Y_ERROR'}
DEPOLARIZE1_PAULIS = ['X', 'Y', 'Z']
DEPOLARIZE2_PAULIS = [
    'IX', 'IY', 'IZ',
    'XI', 'XX', 'XY', 'XZ',
    'YI', 'YX', 'YY', 'YZ',
    'ZI', 'ZX', 'ZY', 'ZZ',
]


class ErrorSimulator:
    """
    Circuit의 모든 fault를 열거하고, 각 fault의 효과를 미리 계산해 저장.

    주요 속성:
        fault_list      : list[dict] — 각 fault 정보
                          {noisy_idx, target_idx, qubit, gate, pauli, qubit2, round, [mpp_group]}
        round_of        : dict[noisy_idx -> round]  (TICK 카운트 기반)
        H_dem           : (n_detectors, n_faults) — fault → detector flip 행렬
        fault_z_errors  : (n_faults, n_qubits)   — fault → 최종 Z pauli frame
        fault_x_errors  : (n_faults, n_qubits)   — fault → 최종 X pauli frame
        fault_meas_flips: (n_faults, n_meas)      — fault → measurement flip
    """
    def __init__(self, circ: stim.Circuit):
        self.circ = circ
        self.n_qubits    = circ.num_qubits
        self.n_detectors = circ.detector_error_model(decompose_errors=False).num_detectors
        self.n_meas      = circ.num_measurements

        self.round_of   = self._build_round_map()
        self.n_rounds   = max(self.round_of.values()) + 1 if self.round_of else 0

        self.fault_list = self._build_fault_list()
        self.n_faults   = len(self.fault_list)

        self.H_dem            = np.zeros((self.n_detectors, self.n_faults), dtype=int)
        self.fault_z_errors   = np.zeros((self.n_faults, self.n_qubits),    dtype=int)
        self.fault_x_errors   = np.zeros((self.n_faults, self.n_qubits),    dtype=int)
        self.fault_meas_flips = np.zeros((self.n_faults, self.n_meas),      dtype=int)

        for j, f in enumerate(self.fault_list):
            effect = self._calc_effect(f)
            self.H_dem[:, j]         = effect['det_flips']
            self.fault_z_errors[j]   = effect['z_errors']
            self.fault_x_errors[j]   = effect['x_errors']
            self.fault_meas_flips[j] = effect['meas_flips']

    def _build_round_map(self) -> dict:
        """
        flattened circuit의 각 instruction index -> round 번호.
        TICK을 만날 때마다 round +1. TICK 이전(맨 앞)은 round 0.
        """
        flat = list(self.circ.flattened())
        round_of = {}
        r = 0
        for idx, inst in enumerate(flat):
            if inst.name == 'TICK':
                r += 1
            round_of[idx] = r
        return round_of

    def _build_fault_list(self) -> list[dict]:
        """
        fault dict 필드:
            noisy_idx  : circ.flattened()에서의 instruction index
            target_idx : instruction 내에서의 target 순서 (DEPOLARIZE2는 pair index)
            qubit      : fault가 직접 발생하는 qubit (MPP는 None)
            gate       : 'Z_ERROR', 'X_ERROR', 'Y_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2', 'MPP'
            pauli      : 발생하는 Pauli ('X', 'Y', 'Z', 또는 'IX' 등 2-qubit, MPP는 None)
            qubit2     : DEPOLARIZE2의 두 번째 qubit (나머지는 None)
            round      : 해당 instruction의 round 번호 (TICK 카운트)
            mpp_group  : MPP fault일 때 해당 group 문자열 (나머지는 없음)
        """
        flat = list(self.circ.flattened())
        fault_list = []

        for noisy_idx, inst in enumerate(flat):
            cur_round = self.round_of[noisy_idx]

            if inst.name in NOISE_GATES:
                for j, t in enumerate(inst.targets_copy()):
                    fault_list.append({
                        'noisy_idx':  noisy_idx,
                        'target_idx': j,
                        'qubit':      t.value,
                        'gate':       inst.name,
                        'pauli':      inst.name[0],
                        'qubit2':     None,
                        'round':      cur_round,
                    })

            elif inst.name == 'DEPOLARIZE1':
                for j, t in enumerate(inst.targets_copy()):
                    for pauli in DEPOLARIZE1_PAULIS:
                        fault_list.append({
                            'noisy_idx':  noisy_idx,
                            'target_idx': j,
                            'qubit':      t.value,
                            'gate':       'DEPOLARIZE1',
                            'pauli':      pauli,
                            'qubit2':     None,
                            'round':      cur_round,
                        })

            elif inst.name == 'DEPOLARIZE2':
                targets = inst.targets_copy()
                for j in range(0, len(targets), 2):
                    q0 = targets[j].value
                    q1 = targets[j + 1].value
                    for pauli in DEPOLARIZE2_PAULIS:
                        fault_list.append({
                            'noisy_idx':  noisy_idx,
                            'target_idx': j // 2,
                            'qubit':      q0,
                            'gate':       'DEPOLARIZE2',
                            'pauli':      pauli,
                            'qubit2':     q1,
                            'round':      cur_round,
                        })

            elif inst.name == 'MPP':
                args = inst.gate_args_copy()
                p = args[0] if args else 0.0
                if p > 0:
                    groups = str(inst).split()[1:]
                    for j, group in enumerate(groups):
                        fault_list.append({
                            'noisy_idx':  noisy_idx,
                            'target_idx': j,
                            'qubit':      None,
                            'gate':       'MPP',
                            'pauli':      None,
                            'qubit2':     None,
                            'round':      cur_round,
                            'mpp_group':  group,
                        })

            elif inst.name == 'M':
                args = inst.gate_args_copy()
                p = args[0] if args else 0.0
                if p > 0:
                    for j, t in enumerate(inst.targets_copy()):
                        fault_list.append({
                            'noisy_idx':  noisy_idx,
                            'target_idx': j,
                            'qubit':      t.value,
                            'gate':       'M',
                            'pauli':      None,
                            'qubit2':     None,
                            'round':      cur_round,
                        })

        return fault_list

    def _calc_effect(self, fault: dict) -> dict:
        """
        noisy circuit에서 해당 fault만 p=1로, 나머지는 p=0으로 설정해서 효과 계산.
        """
        flat = list(self.circ.flattened())
        new_circuit = stim.Circuit()
        target_noisy_idx = fault['noisy_idx']

        for i, inst in enumerate(flat):
            if i != target_noisy_idx:
                # 다른 noise gate는 p=0으로 (제거)
                if inst.name in NOISE_GATES or inst.name in ('DEPOLARIZE1', 'DEPOLARIZE2'):
                    pass  # skip
                elif inst.name == 'MPP':
                    args = inst.gate_args_copy()
                    p = args[0] if args else 0.0
                    if p > 0:
                        # p=0인 noiseless MPP로 교체
                        groups = str(inst).split()[1:]
                        for group in groups:
                            new_circuit += stim.Circuit(f"MPP {group}")
                    else:
                        new_circuit.append(inst)
                elif inst.name == 'M':
                    args = inst.gate_args_copy()
                    p = args[0] if args else 0.0
                    if p > 0:
                        for t in inst.targets_copy():
                            new_circuit.append('M', [t.value])  # noiseless
                    else:
                        new_circuit.append(inst)
                else:
                    new_circuit.append(inst)
            else:
                # 이 fault만 p=1로
                gate = fault['gate']
                if gate in NOISE_GATES:
                    new_circuit.append(f"{fault['pauli']}_ERROR", [fault['qubit']], [1.0])
                elif gate == 'DEPOLARIZE1':
                    new_circuit.append(f"{fault['pauli']}_ERROR", [fault['qubit']], [1.0])
                elif gate == 'DEPOLARIZE2':
                    p0, p1 = fault['pauli'][0], fault['pauli'][1]
                    if p0 != 'I':
                        new_circuit.append(f"{p0}_ERROR", [fault['qubit']], [1.0])
                    if p1 != 'I':
                        new_circuit.append(f"{p1}_ERROR", [fault['qubit2']], [1.0])
                elif gate == 'MPP':
                    groups = str(inst).split()[1:]
                    for j, group in enumerate(groups):
                        if j == fault['target_idx']:
                            new_circuit += stim.Circuit(f"MPP(1.0) {group}")
                        else:
                            new_circuit += stim.Circuit(f"MPP {group}")
                elif gate == 'M':
                    for j, t in enumerate(inst.targets_copy()):
                        if j == fault['target_idx']:
                            new_circuit.append('M', [t.value], [1.0])
                        else:
                            new_circuit.append('M', [t.value])

        sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
        sim.do(new_circuit)
        det_flips     = sim.get_detector_flips(instance_index=0)
        pauli_frame   = sim.peek_pauli_flips(instance_index=0)
        meas_flips_raw = sim.get_measurement_flips(instance_index=0)

        det_array = np.zeros(self.n_detectors, dtype=int)
        for d in np.where(det_flips)[0]:
            det_array[int(d)] = 1

        z_errors = np.zeros(self.n_qubits, dtype=int)
        x_errors = np.zeros(self.n_qubits, dtype=int)
        for k, p in enumerate(pauli_frame):
            if p in (2, 3):
                z_errors[k] = 1
            if p in (1, 2):
                x_errors[k] = 1

        meas_flips = np.zeros(self.n_meas, dtype=int)
        for m in np.where(meas_flips_raw)[0]:
            meas_flips[int(m)] = 1

        return {
            'det_flips':  det_array,
            'z_errors':   z_errors,
            'x_errors':   x_errors,
            'meas_flips': meas_flips,
        }

    def get_effect(self, fault_idx: int) -> dict:
        return {
            'det_flips':  self.H_dem[:, fault_idx],
            'z_errors':   self.fault_z_errors[fault_idx],
            'x_errors':   self.fault_x_errors[fault_idx],
            'meas_flips': self.fault_meas_flips[fault_idx],
        }

    def explain_fault(self, fault_idx: int, file=None) -> None:
        f = self.fault_list[fault_idx]
        print(f"fault_idx={fault_idx}, gate={f['gate']}, qubit={f['qubit']}, "
              f"pauli={f['pauli']}, round={f['round']}", file=file)
        if f['gate'] == 'MPP':
            print(f"  -> mpp_group: {f['mpp_group']}", file=file)
        elif f.get('qubit2') is not None:
            print(f"  -> qubits: ({f['qubit']}, {f['qubit2']})", file=file)

    def find_faults(
        self,
        fault_qubit: int | None = None,  # fault가 직접 발생하는 qubit
        gate: str | None = None,          # 'Z_ERROR', 'X_ERROR', 'MPP' 등
        noisy_idx: int | None = None,     # noisy circuit instruction index
        mpp_check: str | None = None,     # MPP fault의 mpp_group 문자열
        round: int | None = None,         # round 번호
    ) -> list[dict]:
        """
        fault_list 필드 기준으로 fault 필터링. 모든 조건은 AND로 적용됨.

        fault_qubit: fault가 직접 발생하는 qubit (fault_list의 'qubit' 필드)
        gate:        gate 종류 ('Z_ERROR', 'X_ERROR', 'Y_ERROR', 'MPP' 등)
        noisy_idx:   noisy circuit instruction index
        mpp_check:   MPP fault의 mpp_group 문자열 (gate='MPP'일 때만 유효)
        round:       round 번호 (TICK 카운트)
        """
        results = []
        for i, f in enumerate(self.fault_list):
            if fault_qubit is not None and f['qubit'] != fault_qubit:
                continue
            if gate is not None and f['gate'] != gate:
                continue
            if noisy_idx is not None and f['noisy_idx'] != noisy_idx:
                continue
            if mpp_check is not None and f.get('mpp_group') != mpp_check:
                continue
            if round is not None and f['round'] != round:
                continue
            results.append({
                'fault_idx': i,
                'fault': f,
                'noisy_idx': f['noisy_idx'],
                'round': f['round'],
            })
        return results


class LogicalFaultAnalyzer:
    def __init__(self, error_sim: ErrorSimulator, patch):
        self.error_sim = error_sim
        self.circuit = error_sim.circ
        self.patch = patch

        css = patch.css
        self.H_X = np.array(css.Hx, dtype=int)
        self.H_Z = np.array(css.Hz, dtype=int)
        self.L_X = [np.array(l, dtype=int) for l in css.logical_x_repr]
        self.L_Z = [np.array(l, dtype=int) for l in css.logical_z_repr]

        self.fault_list = error_sim.fault_list
        self.n_faults = error_sim.n_faults
        self.n_qubits = error_sim.n_qubits
        self.n_detectors = error_sim.n_detectors
        self.H_dem = error_sim.H_dem

        self.fault_z_map, self.fault_x_map = self._build_fault_maps()

    def _build_fault_maps(self):
        fault_z_map, fault_x_map = {}, {}
        for i in range(self.n_faults):
            fault_z_map[i] = list(np.where(self.error_sim.fault_z_errors[i])[0])
            fault_x_map[i] = list(np.where(self.error_sim.fault_x_errors[i])[0])
        return fault_z_map, fault_x_map

    def _get_config(self, error_type: str):
        if error_type == 'Z':
            return self.fault_z_map, self.H_X, self.L_X
        elif error_type == 'X':
            return self.fault_x_map, self.H_Z, self.L_Z
        else:
            raise ValueError("error_type must be 'X' or 'Z'")

    def _inner_ilp(self, r_vec, H):
        n = len(r_vec)
        prob = pulp.LpProblem("inner", pulp.LpMinimize)
        c = [pulp.LpVariable(f"c_{i}", cat='Binary') for i in range(n)]
        s = H @ r_vec % 2
        for i in range(H.shape[0]):
            k = pulp.LpVariable(f"k_{i}", cat='Integer', lowBound=0)
            prob += pulp.lpSum(int(H[i, j]) * c[j] for j in range(n)) - int(s[i]) == 2 * k
        prob += pulp.lpSum(c)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return np.array([int(round(pulp.value(c[i]))) for i in range(n)])

    def _outer_ilp(self, cuts, weight=None):
        prob = pulp.LpProblem("outer", pulp.LpMinimize)
        e = [pulp.LpVariable(f"e_{i}", cat='Binary') for i in range(self.n_faults)]

        for i in range(self.n_detectors):
            k = pulp.LpVariable(f"k_{i}", cat='Integer', lowBound=0)
            prob += pulp.lpSum(int(self.H_dem[i, j]) * e[j]
                               for j in range(self.n_faults)) == 2 * k

        if weight is not None:
            prob += pulp.lpSum(e) == weight
        else:
            prob += pulp.lpSum(e) >= 1

        for cut in cuts:
            support = [i for i in range(self.n_faults) if cut[i] == 1]
            prob += pulp.lpSum(e[i] for i in support) <= len(support) - 1

        prob += pulp.lpSum(e)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            return None
        return np.array([int(round(pulp.value(e[i]))) for i in range(self.n_faults)])

    def fault_to_r(self, e_vec: np.ndarray, error_type: str = 'Z') -> np.ndarray:
        fault_map, _, _ = self._get_config(error_type)
        r = np.zeros(self.n_qubits, dtype=int)
        for i, ei in enumerate(e_vec):
            if ei == 1:
                for q in fault_map[i]:
                    r[q] ^= 1
        return r

    def is_logical_fault(
        self,
        e_vec: np.ndarray,
        error_type: str = 'Z',
        observable_indices=None,
    ) -> bool:
        fault_map, H, L_list = self._get_config(error_type)
        if observable_indices is None:
            observable_indices = list(range(len(L_list)))

        if not np.all(self.H_dem @ e_vec % 2 == 0):
            return False

        r = self.fault_to_r(e_vec, error_type)
        c = self._inner_ilp(r, H)
        final = (r + c) % 2

        return any(int(L_list[i] @ final % 2) for i in observable_indices)

    def explain(self, e_vec: np.ndarray, file=None) -> None:
        for i, ei in enumerate(e_vec):
            if ei == 1:
                self.error_sim.explain_fault(i, file=file)
                eff = self.error_sim.get_effect(i)
                print(f"fault {i}: "
                      f"det_flips={np.where(eff['det_flips'])[0]}, "
                      f"z_errors={np.where(eff['z_errors'])[0]}, "
                      f"x_errors={np.where(eff['x_errors'])[0]}, "
                      f"meas_flips={np.where(eff['meas_flips'])[0]}", file=file)

    def find_min_weight(
        self,
        error_type: str = 'Z',
        observable_indices=None,
        weight: int = None,
        max_iter: int = 10000,
        verbose: bool = True,
    ) -> np.ndarray:
        fault_map, H, L_list = self._get_config(error_type)
        if observable_indices is None:
            observable_indices = list(range(len(L_list)))
        cuts = []

        for iteration in range(1, max_iter + 1):
            e_val = self._outer_ilp(cuts, weight=weight)
            if e_val is None:
                if verbose:
                    print("No more feasible solutions.")
                return None

            r_val = self.fault_to_r(e_val, error_type)
            c_val = self._inner_ilp(r_val, H)
            final = (r_val + c_val) % 2
            logical_err = any(int(L_list[i] @ final % 2) for i in observable_indices)

            if verbose:
                print(f"iter={iteration}: wt(e)={sum(e_val)}, wt(c)={sum(c_val)}, logical_error={logical_err}")

            if logical_err:
                if verbose:
                    print(f"\nFound: min weight = {sum(e_val)}")
                return e_val

            cuts.append(e_val)

        if verbose:
            print("Max iterations reached.")
        return None

    def find_faults_by_qubit(self, qubit_idx: int, error_type: str = 'Z') -> list[dict]:
        fault_map = self.fault_z_map if error_type == 'Z' else self.fault_x_map
        results = []
        for i, qubits in fault_map.items():
            if qubit_idx in qubits:
                f = self.fault_list[i]
                results.append({
                    'fault_idx': i,
                    'fault': f,
                    'noisy_idx': f['noisy_idx'],
                })
        return results


class SurgeryLogicalFaultAnalyzer:
    """
    Surgery logical error 정의:

    1. split 이후 C1, C2 각각에 ideal decoder 적용 시 residual logical error가 남는 경우.
       - decoding 전 전처리: f0[i].weight() > 0인 deformed Z check i에 대해
         split MZ 결과로 prod_{e in γ} ω_e를 계산.
         이 값이 -1이면 해당 Z check의 post 첫 round syndrome을 뒤집어줌.
       - 전처리 후 C1, C2 각각에 독립적으로 ideal (min-weight) decoder 적용.

    2. XX logical measurement 결과가 잘못된 경우.
       - merge_0 round에서 deformed A_v 측정값들의 product가 XX outcome.
       - 이 값이 fault에 의해 flip됐는지 확인.
    """

    def __init__(self, error_sim: ErrorSimulator, merged: MergedCodeWithGraph, cb):
        """
        error_sim: ErrorSimulator
        merged: MergedCodeWithGraph
        cb: CircuitBuilder (meas_record 포함)
        """
        self.error_sim = error_sim
        self.merged = merged
        self.cb = cb
        self.meas_record = cb.meas_record

        self.n_faults = error_sim.n_faults
        self.n_qubits = error_sim.n_qubits
        self.n_detectors = error_sim.n_detectors
        self.H_dem = error_sim.H_dem
        self.fault_list = error_sim.fault_list

        # C1, C2 qubit stim index
        registry = merged._registry
        self.c1_qubit_indices = [registry[q] for q in merged.C1.data_qubits()]
        self.c2_qubit_indices = [registry[q] for q in merged.C2.data_qubits()]
        self.n_c1 = len(merged.C1.data_qubits())
        self.n_c2 = len(merged.C2.data_qubits())

        # ancilla qubit stim index
        self.anc_qubit_indices = [registry[q] for q in merged.anc_graph.data_qubits()]
        # ancilla qubit → f0 index 매핑 (split MZ 결과로 syndrome 전처리용)
        # f0[i].qubits에 있는 ancilla qubit들이 deformed Z check i에 대응
        self._f0 = merged._f0
        c_z_checks = merged.C1.z_checks() + merged.C2.z_checks()
        self.c_z_checks = c_z_checks

        # C1 H_X, H_Z, logical repr (C1 local index 기준)
        self.H_X1 = self._checks_to_matrix(merged.C1.x_checks(), merged.C1.data_qubits(), registry)
        self.H_Z1 = self._checks_to_matrix(merged.C1.z_checks(), merged.C1.data_qubits(), registry)
        self.L_X1 = self._logicals_to_matrix(merged.C1, registry, 'X')
        self.L_Z1 = self._logicals_to_matrix(merged.C1, registry, 'Z')

        # C2 H_X, H_Z, logical repr (C2 local index 기준)
        self.H_X2 = self._checks_to_matrix(merged.C2.x_checks(), merged.C2.data_qubits(), registry)
        self.H_Z2 = self._checks_to_matrix(merged.C2.z_checks(), merged.C2.data_qubits(), registry)
        self.L_X2 = self._logicals_to_matrix(merged.C2, registry, 'X')
        self.L_Z2 = self._logicals_to_matrix(merged.C2, registry, 'Z')

        # deformed A_v meas index (merge_0 round):
        # merged.x_checks()에서 C1, C2 x_checks 다음이 deformed A_v
        n_c1x = len(merged.C1.x_checks())
        n_c2x = len(merged.C2.x_checks())
        deformed_av_checks = merged.x_checks()[n_c1x + n_c2x:]
        # check → meas index dict (label='merge_0'인 것만)
        self.av_merge0_meas_indices = []
        check_to_key = {entry['check']: k for k, entry in self.meas_record.items()
                        if entry['label'] == 'merge_0' and entry['check'] is not None}
        for av_check in deformed_av_checks:
            if av_check in check_to_key:
                self.av_merge0_meas_indices.append(check_to_key[av_check])

        # split tick MZ meas index: ancilla qubit q → meas index
        self.anc_q_to_meas_idx = {}
        for k, entry in self.meas_record.items():
            if entry['label'] == 'split' and entry['check'] is None:
                # split은 Qubit 단위로 저장됨
                for q in entry.get('qubits', []):
                    self.anc_q_to_meas_idx[q] = k

        # post 첫 round Z check meas index: check → meas index
        post_first_label = f"post_0"
        self.post0_z_meas_idx = {}
        for k, entry in self.meas_record.items():
            if entry['label'] == post_first_label and entry['check'] is not None:
                if entry['check'].pauli == 'Z':
                    self.post0_z_meas_idx[entry['check']] = k

    def _checks_to_matrix(self, checks, data_qubits, registry):
        """Check list → (r, n) binary matrix, n = len(data_qubits)"""
        q_to_local = {q: i for i, q in enumerate(data_qubits)}
        n = len(data_qubits)
        H = np.zeros((len(checks), n), dtype=int)
        for i, chk in enumerate(checks):
            for q in chk.qubits:
                if q in q_to_local:
                    H[i, q_to_local[q]] = 1
        return H

    def _logicals_to_matrix(self, patch, registry, pauli_type: str):
        """patch의 logical repr → list of binary vectors (local qubit index 기준)"""
        data_qubits = patch.data_qubits()
        q_to_local = {q: i for i, q in enumerate(data_qubits)}
        n = len(data_qubits)
        logicals = patch._x_logicals if pauli_type == 'X' else patch._z_logicals
        result = []
        for l in logicals:
            vec = np.zeros(n, dtype=int)
            for q in l:
                if q in q_to_local:
                    vec[q_to_local[q]] = 1
            result.append(vec)
        return result

    def _inner_ilp(self, r_vec, H, syndrome_offset=None):
        n = len(r_vec)
        prob = pulp.LpProblem("inner", pulp.LpMinimize)
        c = [pulp.LpVariable(f"c_{i}", cat='Binary') for i in range(n)]
        s = H @ r_vec % 2
        if syndrome_offset is not None:
            s = (s + syndrome_offset) % 2
        for i in range(H.shape[0]):
            k = pulp.LpVariable(f"k_{i}", cat='Integer', lowBound=0)
            prob += pulp.lpSum(int(H[i, j]) * c[j] for j in range(n)) - int(s[i]) == 2 * k
        prob += pulp.lpSum(c)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        return np.array([int(round(pulp.value(c[i]))) for i in range(n)])

    def is_surgery_logical_fault(self, e_vec: np.ndarray) -> bool:
        # undetectable 조건
        if not np.all(self.H_dem @ e_vec % 2 == 0):
            print("It is detectable!")
            # return False

        # e_vec에 해당하는 fault들의 효과 합산
        meas_flips = np.zeros(self.error_sim.n_meas, dtype=int)
        z_errors = np.zeros(self.n_qubits, dtype=int)
        x_errors = np.zeros(self.n_qubits, dtype=int)
        for i, ei in enumerate(e_vec):
            if ei == 1:
                meas_flips ^= self.error_sim.fault_meas_flips[i]
                z_errors ^= self.error_sim.fault_z_errors[i]
                x_errors ^= self.error_sim.fault_x_errors[i]
        print("meas_flips:")
        print(meas_flips)
        print("z_errors: ")
        print(z_errors)
        print("x_errors: ")
        print(x_errors)
        # --- 조건 2: XX measurement outcome 오류 ---
        xx_outcome = 0
        for idx in self.av_merge0_meas_indices:
            xx_outcome ^= meas_flips[idx]
        if xx_outcome == 1:
            print("The XX measurement outcome is wrong!")
            # return True

        # --- 조건 1: code block logical error ---
        anc_meas = {q: meas_flips[idx] for q, idx in self.anc_q_to_meas_idx.items()}

        z1 = np.array([z_errors[i] for i in self.c1_qubit_indices])
        z2 = np.array([z_errors[i] for i in self.c2_qubit_indices])
        x1 = np.array([x_errors[i] for i in self.c1_qubit_indices])
        x2 = np.array([x_errors[i] for i in self.c2_qubit_indices])

        # 전처리: f0[i].weight() > 0인 Z check i에 대해
        # prod_{e in γ} ω_e == -1이면 post_0의 해당 Z check syndrome flip
        # → z_errors에 반영 (Z error를 추가하면 syndrome이 뒤집힘)
        # 실제로는 syndrome을 직접 계산하므로 z_errors를 수정하는 대신
        # H_Z @ r % 2에서 해당 행을 flip하는 방식으로 처리
        syndrome_flip_c1 = np.zeros(len(self.c_z_checks[:len(self.merged.C1.z_checks())]), dtype=int)
        syndrome_flip_c2 = np.zeros(len(self.c_z_checks[len(self.merged.C1.z_checks()):]), dtype=int)

        n_c1z = len(self.merged.C1.z_checks())
        for i, f0_check in enumerate(self._f0):
            print(f"i={i}, f0_check={f0_check}, weight={f0_check.weight()}")
            if f0_check.weight() == 0:
                continue
            prod = 0
            for q in f0_check.qubits:
                prod ^= anc_meas.get(q, 0)
            if prod == 1:  # prod = -1
                if i < n_c1z:
                    syndrome_flip_c1[i] ^= 1
                else:
                    syndrome_flip_c2[i - n_c1z] ^= 1
        print("syndrome_flip_c1 ",syndrome_flip_c1)
        print("syndrome_flip_c2 ",syndrome_flip_c2)
        # C1 logical error check (Z error → X logical)
        # syndrome_flip_c1: MZ 결과로 인한 Z check syndrome offset
        c1_z = self._inner_ilp(z1, self.H_X1)
        final1_z = (z1 + c1_z) % 2
        print("***** Codeblock 1 *****")
        print("cumulated Z error: ",z1)
        print("Z correction: ",c1_z)
        print("Final Z error: ",final1_z)
        logical_err_c1_z = any(int(l @ final1_z % 2) for l in self.L_X1)

        c1_x = self._inner_ilp(x1, self.H_Z1, syndrome_offset=syndrome_flip_c1)
        final1_x = (x1 + c1_x) % 2
        print("cumulated X error: ",x1)
        print("X correction: ",c1_x)
        print("Final X error: ",final1_x)
        logical_err_c1_x = any(int(l @ final1_x % 2) for l in self.L_Z1)

        # C2 logical error check
        c2_z = self._inner_ilp(z2, self.H_X2)
        final2_z = (z2 + c2_z) % 2
        print("***** Codeblock 2 *****")
        print("cumulated Z error: ",z2)
        print("Z correction: ",c2_z)
        print("Final Z error: ",final2_z)
        logical_err_c2_z = any(int(l @ final2_z % 2) for l in self.L_X2)

        c2_x = self._inner_ilp(x2, self.H_Z2, syndrome_offset=syndrome_flip_c2)
        final2_x = (x2 + c2_x) % 2
        print("cumulated X error: ",x2)
        print("X correction: ",c2_x)
        print("Final X error: ",final2_x)
        logical_err_c2_x = any(int(l @ final2_x % 2) for l in self.L_Z2)

        return logical_err_c1_z or logical_err_c1_x or logical_err_c2_z or logical_err_c2_x

    def find_min_weight(
        self,
        weight: int = None,
        max_iter: int = 10000,
        verbose: bool = True,
    ) -> np.ndarray | None:
        cuts = []

        for iteration in range(1, max_iter + 1):
            prob = pulp.LpProblem("outer", pulp.LpMinimize)
            e = [pulp.LpVariable(f"e_{i}", cat='Binary') for i in range(self.n_faults)]

            for i in range(self.n_detectors):
                k = pulp.LpVariable(f"k_{i}", cat='Integer', lowBound=0)
                prob += pulp.lpSum(int(self.H_dem[i, j]) * e[j]
                                   for j in range(self.n_faults)) == 2 * k

            if weight is not None:
                prob += pulp.lpSum(e) == weight
            else:
                prob += pulp.lpSum(e) >= 1

            for cut in cuts:
                support = [i for i in range(self.n_faults) if cut[i] == 1]
                prob += pulp.lpSum(e[i] for i in support) <= len(support) - 1

            prob += pulp.lpSum(e)
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if prob.status != 1:
                if verbose:
                    print("No more feasible solutions.")
                return None

            e_val = np.array([int(round(pulp.value(e[i]))) for i in range(self.n_faults)])
            logical_err = self.is_surgery_logical_fault(e_val)

            if verbose:
                print(f"iter={iteration}: wt(e)={sum(e_val)}, logical_error={logical_err}")

            if logical_err:
                if verbose:
                    print(f"\nFound: min weight = {sum(e_val)}")
                return e_val

            cuts.append(e_val)

        if verbose:
            print("Max iterations reached.")
        return None

    def explain(self, idxs, file=None) -> None:
        for idx in idxs:
            self.error_sim.explain_fault(idx, file=file)
            eff = self.error_sim.get_effect(idx)
            print(f"fault {idx}: "
                  f"det_flips={np.where(eff['det_flips'])[0]}, "
                  f"z_errors={np.where(eff['z_errors'])[0]}, "
                  f"x_errors={np.where(eff['x_errors'])[0]}, "
                  f"meas_flips={np.where(eff['meas_flips'])[0]}", file=file)