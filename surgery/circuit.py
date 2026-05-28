import stim
from collections import defaultdict
from dataclasses import dataclass, field
from core.primitives import Qubit, Check


@dataclass
class Instruction:
    tick: int
    gate: str
    targets: list
    # gate별 targets 형식:
    # single qubit gate (H, DEPOLARIZE1, M, MX ...): list[Qubit]
    # two qubit gate (CNOT, DEPOLARIZE2, ...): list[tuple[Qubit, Qubit]]
    # MPP: list[Check]
    gate_args: list[float] = field(default_factory=list)
    label: str = None
    check: Check | None = None  # MPP: None (targets에 Check가 이미 있음), single meas: 해당 Check


class CircuitSchedule:
    def __init__(self, qubits: set, instructions: dict[int, list[Instruction]] = None):
        self.qubits = qubits  # 회로에 등장하는 모든 qubit
        self.qubit_idxs: dict[Qubit, int] = None
        self.instructions: dict[int, list[Instruction]] = instructions if instructions is not None else defaultdict(list)  # tick -> instruction 목록
        self.qubit_coords: dict = {}  # qubit -> (x, y)
        self.extra_detectors: list[list[tuple[int, Qubit|Check]]] = []

class CircuitBuilder:
    """
    CircuitSchedule을 받아 stim circuit과 meas_record를 생성
    """

    def __init__(self, schedule: CircuitSchedule):
        self.meas_record = {}  # global_meas_idx -> {tick, gate, qubits, label, check}
        self._circuit = stim.Circuit()
        self._qubit_idx = schedule.qubit_idxs # 이거 이상함. 알파벳 순서임.
        # merged code에서의 global(local) idx가 필요. merged code를 넘겨줘야하나..? 
        self._schedule = schedule

    @property
    def circuit(self):
        return self._circuit.copy()

    def build(self):
        if self._schedule.qubit_coords:
            for q, (x, y) in self._schedule.qubit_coords.items():
                idx = self._qubit_idx[q]
                # print(q,idx)
                self._circuit.append("QUBIT_COORDS", [idx], [x, y])

        for tick in sorted(self._schedule.instructions.keys()):
            for instr in self._schedule.instructions[tick]:
                self._append_instruction(instr)
            self._circuit.append("TICK", [])

    def _append_instruction(self, instr: Instruction):
        gate_data = stim.gate_data(instr.gate)

        if gate_data.takes_pauli_targets:  # MPP
            stim_targets = []
            for check in instr.targets:  # list[Check]
                qubits = sorted(check.qubits)
                for k, qubit in enumerate(qubits):
                    if check.pauli == "X":
                        stim_targets.append(stim.target_x(self._qubit_idx[qubit]))
                    elif check.pauli == "Y":
                        stim_targets.append(stim.target_y(self._qubit_idx[qubit]))
                    elif check.pauli == "Z":
                        stim_targets.append(stim.target_z(self._qubit_idx[qubit]))
                    if k < len(qubits) - 1:
                        stim_targets.append(stim.target_combiner())
        elif gate_data.is_two_qubit_gate:  # CNOT, DEPOLARIZE2 등
            stim_targets = [self._qubit_idx[q] for pair in instr.targets for q in pair]
        else:  # single qubit gate, M, MX 등
            stim_targets = [self._qubit_idx[q] for q in instr.targets]

        if gate_data.produces_measurements:
            meas_count_before = len(self.meas_record)
            if gate_data.takes_pauli_targets:  # MPP: Check 하나당 measurement 하나
                for i, check in enumerate(instr.targets):
                    self.meas_record[meas_count_before + i] = {
                        "tick": instr.tick,
                        "gate": instr.gate,
                        "qubits": sorted(check.qubits),
                        "label": instr.label,
                        "check": check,
                    }
            else:  # single qubit meas
                for i, qubit in enumerate(instr.targets):
                    self.meas_record[meas_count_before + i] = {
                        "tick": instr.tick,
                        "gate": instr.gate,
                        "qubits": [qubit],
                        "label": instr.label,
                        "check": instr.check,
                    }

        self._circuit.append(instr.gate, stim_targets, instr.gate_args)

    def get_rec(self, tick: int = None, qubit: Qubit = None, label: str = None, check: Check = None, gate: str = None) -> list[stim.GateTarget]:
        res = self.get_meas_idx(tick, qubit, label, check, gate)
        return [stim.target_rec(r - len(self.meas_record)) for r in res]

    def get_meas_idx(self, tick: int = None, qubit: Qubit = None, label: str = None, check: Check = None, gate: str = None) -> list[int]:
        res = []
        for idx, entry in self.meas_record.items():
            if tick is not None and entry["tick"] != tick:
                continue
            if qubit is not None and qubit not in entry["qubits"]:
                continue
            if label is not None and entry["label"] != label:
                continue
            if check is not None and entry["check"] != check:
                continue
            if gate is not None and entry["gate"] != gate:
                continue
            res.append(idx)
        return res

    def add_detectors(self):
        # 같은 check 객체가 연속으로 측정된 경우 자동으로 detector 추가
        
        check_to_meas = defaultdict(list)
        for idx, entry in self.meas_record.items():
            if entry['check'] is not None:
                check_to_meas[entry['check']].append((entry['tick'], idx))

        for check, meas_list in check_to_meas.items():
            meas_list.sort()

                
            for i in range(1, len(meas_list)):
                tick_cur, cur_idx = meas_list[i]
                tick_prev, prev_idx = meas_list[i-1]
                
                if tick_cur - tick_prev != 1:
                    continue
                # prev MPP meas가 p=0이라면 cur 단독으로 추가하는 로직 필요.
                recs = [
                    stim.target_rec(cur_idx - len(self.meas_record)),
                    stim.target_rec(prev_idx - len(self.meas_record)),
                ]
                self._circuit.append('DETECTOR', recs)

        # 자동으로 못 잡는 경계 부분 detector (schedule.extra_detectors에 저장된 것들)
        for detector in self._schedule.extra_detectors:
            recs = []
            for tick, target in detector:
                if isinstance(target, Check):
                    idxs = self.get_meas_idx(tick=tick, check=target)
                else:  # Qubit
                    idxs = self.get_meas_idx(tick=tick, qubit=target)
                recs += [stim.target_rec(idx - len(self.meas_record)) for idx in idxs]
            self._circuit.append('DETECTOR', recs)

            
    def add_observables(self, supports: list[list[Qubit]]):
        final_data_idxs = self.get_meas_idx(label='final_data')
        qubit_to_final_idx = {self.meas_record[idx]['qubits'][0]: idx for idx in final_data_idxs}

        for obs_idx, support in enumerate(supports):
            obs_recs = [
                stim.target_rec(qubit_to_final_idx[q] - len(self.meas_record))
                for q in support
            ]
            self._circuit.append('OBSERVABLE_INCLUDE', obs_recs, obs_idx)
    
    def explain_detectors(self):
        total = len(self.meas_record)
        for inst in self._circuit:
            if inst.name != 'DETECTOR':
                continue
            parts = []
            for t in inst.targets_copy():
                idx = total + t.value
                entry = self.meas_record[idx]
                parts.append(f"tick={entry['tick']} label={entry['label']} check={entry['check']}")
            print("DETECTOR: " + " | ".join(parts))