import numpy as np
import pulp
import stim


class LogicalFaultAnalyzer:
    def __init__(self, circuit: stim.Circuit, patch):
        """
        Parameters
        ----------
        circuit : stim.Circuit
        patch   : CodePatch instance (css 속성 필요)
        """
        self.circuit = circuit
        self.patch = patch

        css = patch.css
        self.H_X = np.array(css.Hx, dtype=int)
        self.H_Z = np.array(css.Hz, dtype=int)
        self.L_X = [np.array(l, dtype=int) for l in css.logical_x_repr]
        self.L_Z = [np.array(l, dtype=int) for l in css.logical_z_repr]

        dem = circuit.detector_error_model(decompose_errors=False)
        self.faults = [inst for inst in dem.flattened() if inst.type == "error"]
        self.n_faults = len(self.faults)
        self.n_qubits = circuit.num_qubits
        self.n_detectors = dem.num_detectors

        self.H_dem = np.zeros((self.n_detectors, self.n_faults), dtype=int)
        for j, inst in enumerate(self.faults):
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    self.H_dem[t.val, j] = 1

        self.fault_z_map, self.fault_x_map = self._build_fault_maps()

    def _build_fault_maps(self):
        fault_z_map, fault_x_map = {}, {}
        for i, inst in enumerate(self.faults):
            targets_str = " ".join(str(t) for t in inst.targets_copy())
            explained = self.circuit.explain_detector_error_model_errors(
                dem_filter=stim.DetectorErrorModel(f"error(1) {targets_str}"),
                reduce_to_one_representative_error=False
            )
            z_qubits, x_qubits = [], []
            for e in explained:
                for loc in e.circuit_error_locations:
                    for target in loc.flipped_pauli_product:
                        gt = target.gate_target
                        if gt.is_z_target or gt.is_y_target:
                            z_qubits.append(gt.value)
                        if gt.is_x_target or gt.is_y_target:
                            x_qubits.append(gt.value)
            fault_z_map[i] = z_qubits
            fault_x_map[i] = x_qubits
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
        """
        fault vector e -> data qubit error vector r (mod 2).
        error_type='Z': Z error, error_type='X': X error
        """
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
        observable_indices = None,
    ) -> bool:
        """
        e_vec이 undetectable하고, minimum weight decoder 후
        observable_indices 중 하나라도 logical error를 발생시키는지 확인.
        observable_indices=None이면 모든 observable 확인.
        """
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
        """
        e_vec의 active fault들을 circuit error location으로 설명 출력.
        """
        for i, ei in enumerate(e_vec):
            if ei == 1:
                inst = self.faults[i]
                targets_str = " ".join(str(t) for t in inst.targets_copy())
                explained = self.circuit.explain_detector_error_model_errors(
                    dem_filter=stim.DetectorErrorModel(f"error(1) {targets_str}"),
                    reduce_to_one_representative_error=False
                )
                print(f"fault {i}: {inst}")
                for e in explained:
                    for loc in e.circuit_error_locations:
                        print(f"  -> {loc}")
                print()

    def find_min_weight(
        self,
        error_type: str = 'Z',
        observable_indices=None,
        weight: int = None,
        max_iter: int = 10000,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Benders decomposition으로 min weight undetectable logical fault e를 찾는다.
        weight=None이면 자동 탐색, int이면 해당 weight만 탐색.
        observable_indices 중 하나라도 logical error가 나면 반환.
        logical error가 없으면 None 반환.
        """
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
        """
        특정 qubit에 error를 발생시키는 fault 목록 반환.
        각 fault는 {fault_idx, tick, instruction, pauli, coord} 정보 포함.
        """
        fault_map = self.fault_z_map if error_type == 'Z' else self.fault_x_map
        results = []
        for i, qubits in fault_map.items():
            if qubit_idx in qubits:
                inst = self.faults[i]
                targets_str = " ".join(str(t) for t in inst.targets_copy())
                explained = self.circuit.explain_detector_error_model_errors(
                    dem_filter=stim.DetectorErrorModel(f"error(1) {targets_str}"),
                    reduce_to_one_representative_error=False
                )
                for e in explained:
                    for loc in e.circuit_error_locations:
                        for target in loc.flipped_pauli_product:
                            gt = target.gate_target
                            if gt.value == qubit_idx:
                                results.append({
                                    'fault_idx': i,
                                    'tick': loc.tick_offset,
                                    'instruction': loc.instruction_targets,
                                    'pauli': str(gt),
                                    'coord': target.coords,
                                })
        return results