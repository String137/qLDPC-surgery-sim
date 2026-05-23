import numpy as np
import pulp
import stim

NOISE_GATES = {'Z_ERROR', 'X_ERROR', 'Y_ERROR'}
DEPOLARIZE1_PAULIS = ['X', 'Y', 'Z']
DEPOLARIZE2_PAULIS = [
    'IX', 'IY', 'IZ',
    'XI', 'XX', 'XY', 'XZ',
    'YI', 'YX', 'YY', 'YZ',
    'ZI', 'ZX', 'ZY', 'ZZ',
]


class ErrorSimulator:
    def __init__(self, circ: stim.Circuit):
        self.circ = circ
        self.noiseless_flat = list(circ.without_noise().flattened())
        self.n_qubits = circ.num_qubits
        self.fault_list = self._build_fault_list()
        self.n_detectors = circ.detector_error_model(decompose_errors=False).num_detectors
        self.n_faults = len(self.fault_list)

        self.fault_z_errors = np.zeros((self.n_faults, self.n_qubits), dtype=int)
        self.fault_x_errors = np.zeros((self.n_faults, self.n_qubits), dtype=int)
        self.H_dem = np.zeros((self.n_detectors, self.n_faults), dtype=int)

        for j, f in enumerate(self.fault_list):
            effect = self.get_fault_effect(f)
            self.H_dem[:, j] = effect['det_flips']
            self.fault_z_errors[j] = effect['z_errors']
            self.fault_x_errors[j] = effect['x_errors']

    def _build_fault_list(self) -> list[dict]:
        flat = list(self.circ.flattened())
        fault_list = []
        noiseless_idx = 0

        for inst in flat:
            if inst.name in NOISE_GATES:
                for j, t in enumerate(inst.targets_copy()):
                    pauli = inst.name[0]
                    fault_list.append({
                        'inst_idx':   noiseless_idx,
                        'target_idx': j,
                        'qubit':      t.value,
                        'gate':       inst.name,
                        'pauli':      pauli,
                        'qubit2':     None,
                    })

            elif inst.name == 'DEPOLARIZE1':
                for j, t in enumerate(inst.targets_copy()):
                    for pauli in DEPOLARIZE1_PAULIS:
                        fault_list.append({
                            'inst_idx':   noiseless_idx,
                            'target_idx': j,
                            'qubit':      t.value,
                            'gate':       'DEPOLARIZE1',
                            'pauli':      pauli,
                            'qubit2':     None,
                        })

            elif inst.name == 'DEPOLARIZE2':
                targets = inst.targets_copy()
                for j in range(0, len(targets), 2):
                    q0 = targets[j].value
                    q1 = targets[j + 1].value
                    for pauli in DEPOLARIZE2_PAULIS:
                        fault_list.append({
                            'inst_idx':   noiseless_idx,
                            'target_idx': j // 2,
                            'qubit':      q0,
                            'gate':       'DEPOLARIZE2',
                            'pauli':      pauli,
                            'qubit2':     q1,
                        })

            elif inst.name == 'MPP':
                args = inst.gate_args_copy()
                p = args[0] if args else 0.0
                groups = str(inst).split()[1:]
                if p > 0:
                    for j, group in enumerate(groups):
                        fault_list.append({
                            'inst_idx':   noiseless_idx,
                            'target_idx': j,
                            'qubit':      None,
                            'gate':       'MPP',
                            'pauli':      None,
                            'qubit2':     None,
                            'mpp_group':  group,
                        })
                noiseless_idx += 1

            else:
                noiseless_idx += 1

        return fault_list

    def _build_mpp_fault_circuit(self, inst_idx: int, group_idx: int) -> stim.Circuit:
        inst = self.noiseless_flat[inst_idx]
        groups = str(inst).split()[1:]

        new_circuit = stim.Circuit()
        for i, orig_inst in enumerate(self.noiseless_flat):
            if i == inst_idx:
                for j, group in enumerate(groups):
                    if j == group_idx:
                        new_circuit += stim.Circuit(f"MPP(1.0) {group}")
                    else:
                        new_circuit += stim.Circuit(f"MPP {group}")
            else:
                new_circuit.append(orig_inst)
        return new_circuit

    def _build_pauli_fault_circuit(self, inst_idx: int, qubit: int, pauli: str,
                                    qubit2: int | None = None) -> stim.Circuit:
        new_circuit = stim.Circuit()
        for i, inst in enumerate(self.noiseless_flat):
            if i == inst_idx:
                if qubit2 is None:
                    new_circuit.append(f"{pauli}_ERROR", [qubit], [1.0])
                else:
                    p0, p1 = pauli[0], pauli[1]
                    if p0 != 'I':
                        new_circuit.append(f"{p0}_ERROR", [qubit], [1.0])
                    if p1 != 'I':
                        new_circuit.append(f"{p1}_ERROR", [qubit2], [1.0])
            new_circuit.append(inst)
        return new_circuit

    def get_fault_effect(self, fault: dict) -> dict:
        if fault['gate'] == 'MPP':
            new_circuit = self._build_mpp_fault_circuit(fault['inst_idx'], fault['target_idx'])
        else:
            new_circuit = self._build_pauli_fault_circuit(
                fault['inst_idx'],
                fault['qubit'],
                fault['pauli'],
                fault['qubit2'],
            )

        sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
        sim.do(new_circuit)
        det_flips = sim.get_detector_flips(instance_index=0)
        pauli_frame = sim.peek_pauli_flips(instance_index=0)

        det_array = np.zeros(self.n_detectors, dtype=int)
        for d in np.where(det_flips)[0]:
            det_array[int(d)] = 1

        z_errors = np.zeros(self.n_qubits, dtype=int)
        x_errors = np.zeros(self.n_qubits, dtype=int)
        for i, p in enumerate(pauli_frame):
            if p in (3, 2):  # Z, Y
                z_errors[i] = 1
            if p in (1, 2):  # X, Y
                x_errors[i] = 1

        return {
            'det_flips': det_array,
            'z_errors':  z_errors,
            'x_errors':  x_errors,
        }

    def inject_fault(self, fault: dict) -> dict:
        return self.get_fault_effect(fault)

    def get_dem(self):
        return self.H_dem

    def explain_fault(self, fault: dict) -> None:
        f = fault
        inst = self.noiseless_flat[f['inst_idx']]
        print(f"fault: gate={f['gate']}, inst_idx={f['inst_idx']}, qubit={f['qubit']}, pauli={f['pauli']}")
        print(f"  -> instruction at inst_idx={f['inst_idx']}: {inst}")
        if f['gate'] == 'MPP':
            print(f"  -> MPP group: {f['mpp_group']}")
        elif f.get('qubit2') is not None:
            print(f"  -> qubits: ({f['qubit']}, {f['qubit2']})")


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

    def explain(self, e_vec: np.ndarray) -> None:
        for i, ei in enumerate(e_vec):
            if ei == 1:
                f = self.fault_list[i]
                self.error_sim.explain_fault(f)
                print()

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
                inst = self.error_sim.noiseless_flat[f['inst_idx']]
                results.append({
                    'fault_idx': i,
                    'fault': f,
                    'inst_idx': f['inst_idx'],
                    'instruction': inst,
                })
        return results