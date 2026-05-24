import stim
from collections import defaultdict
from core.primitives import *
from core.codes import *
import numpy as np
from surgery.circuit import *
from surgery.surgery import MergedCodeWithGraph


class MemoryCircuitScheduler:
    def __init__(self, patch: CodePatch, rounds: int, p_data: float, p_meas: float):
        self.patch = patch
        self.rounds = rounds
        self.p_data = p_data
        self.p_meas = p_meas

    def build(self) -> CircuitSchedule:
        # tick 0: RX (|+> 초기화)
        # tick 1: DEPOLARIZE1 + MPP X checks only, noisy, label="round_0", DETECTOR 없음
        # tick 2 ~ rounds: DEPOLARIZE1 + MPP X+Z checks, noisy, label="round_k"
        # 마지막 round: noiseless MPP X checks only, label="final"
        # MX data qubits
        data_qubits = self.patch.data_qubits()
        x_checks = self.patch.x_checks()
        z_checks = self.patch.z_checks()

        schedule = CircuitSchedule(set(data_qubits))
        tick = 0

        # tick 0: RX (|+> 초기화)
        schedule.instructions[tick].append(
            Instruction(tick, "RX", data_qubits)
        )
        tick += 1

        # tick 1: MPP X checks only (noisy, label="round_0")
        # schedule.instructions[tick].append(
        #     Instruction(tick, "DEPOLARIZE1", data_qubits, gate_args=[self.p_data])
        # )
        schedule.instructions[tick].append(
            Instruction(tick, "MPP", x_checks, gate_args=[0], label="round_0")
        )
        schedule.instructions[tick].append(
            Instruction(tick, "MPP", z_checks, gate_args=[0], label="round_0")
        )
        tick += 1

        # tick 2 ~ rounds: DEPOLARIZE1 + MPP X+Z checks (noisy)
        for r in range(1, self.rounds+1):
            schedule.instructions[tick].append(
                Instruction(tick, "Z_ERROR", data_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "X_ERROR", data_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "MPP", x_checks + z_checks, gate_args=[self.p_meas], label=f"round_{r}")
            )
            tick += 1

        # 마지막 round: data Z error + noiseless MPP X checks only + MX data qubits
        schedule.instructions[tick].append(
            Instruction(tick, "Z_ERROR", data_qubits, gate_args=[self.p_data])
        )
        schedule.instructions[tick].append(
            Instruction(tick, "X_ERROR", data_qubits, gate_args=[self.p_data])
        )
        # schedule.instructions[tick].append(
        #     Instruction(tick, "MPP", x_checks, gate_args=[0], label="final")
        # )
        schedule.instructions[tick].append(
            Instruction(tick, "MX", data_qubits, gate_args=[0], label="final_data")
        )

        return schedule
    

class SurgeryCircuitScheduler:
    
    """
    XX logical measurement via ancillary graph surgery.

    Circuit structure:
        1. Pre-rounds  : C1, C2 각자 EC (merge 전)
        2. Merge-rounds: merged code EC
        3. Post-rounds : C1, C2 각자 EC (split 후)

    Initialization:
        - C1, C2 data qubit: RX (|+> 초기화)
        - Ancillary graph qubit: R (|0> 초기화)
          → ancilla qubit은 C1, C2의 Z check에 추가되므로 |0>이어야
            merge 시작 round의 Z check meas결과가 이전 round와 같아야 한다는
            detector를 걸 수 있음.

    Reference measurement (noiseless, circuit 시작 시):
        - C1, C2의 X check, Z check 모두 noiseless로 측정 → detector reference
        - Ancillary graph의 Z check (B_p, cycle check)도 noiseless로 측정 → detector reference
          → ancilla qubit이 |0>으로 init되므로 B_p의 첫 측정값은 결정론적으로 0.
            따라서 noiseless reference로 잡고 merge 첫 round의 B_p 측정값과
            비교하는 detector를 걸 수 있음.
        - Ancillary graph의 X check (A_v)는 merge 시작 round부터 새로 생기는
            merged code의 check이므로 이전 round가 없어 첫 측정값이 reference가 됨.
            그 product가 XX logical measurement 결과가 됨.

    Detector 설정:
        - 일반적으로: 이전 round의 같은 check meas결과와 비교하는 detector 추가
        - Ancillary graph의 X check (A_v)는 merge 첫 round에 detector 없음.
          두 번째 round부터 이전 round와 비교하는 detector 추가.
        - Ancillary graph의 Z check (B_p)는 merge 첫 round에
          noiseless reference와 비교하는 detector 추가.
          이후 round부터 이전 round와 비교하는 detector 추가.
        - C1, C2의 Z check는 merge 시작 round부터 ancilla qubit (edge)을 포함한
          merged Z check로 측정됨. ancilla qubit이 |0>이므로 meas결과는
          이전 round (pre-round 마지막)와 같아야 함.
          따라서 merge 시작 round의 해당 Z check meas결과와
          직전 round의 같은 Z check meas결과를 비교하는 detector를 추가해야 함.

    Split (ancilla 측정):
        - Ancillary graph qubit을 MZ로 측정 (|0> init과 같은 basis)
        - Deformed Z check는 s̃_j = s_j * prod_{e in γ} Z_e 로 정의됨.
          즉 원래 Z check에 ancilla edge qubit들의 Z가 곱해진 형태.
        - Split 시점에 edge qubit을 MZ로 측정하면 Z_e가 classical value ω_e로 collapse됨.
          따라서 split 후 원래 Z check 값은 s_j = s̃_j * prod_{e in γ} ω_e 로 복원됨.
        - Detector:
            (마지막 merge round의 s̃_j meas결과)
            XOR (γ path에 속하는 ancilla qubit들의 MZ 결과)
            XOR (split 후 첫 post-round의 s_j meas결과)
            == 0
          에러가 없으면 위 관계가 항상 성립하므로 detector로 사용 가능.
    """

    def __init__(self, merged: MergedCodeWithGraph, pre_round: int, merge_round: int, post_round: int):
        # pre_round, merge_round, post_round
        # merge 전, merge 도중, merge 후의 EC round 반복 횟수
        self.C1 = merged.C1
        self.C2 = merged.C2
        data_qubits = self.merged.data_qubits()
        print("data_qubits: ")
        print(data_qubits)
        print("port_function:")
        print(merged.port_func)
        schedule = CircuitSchedule(set(data_qubits))
        tick = 0

        # tick 0: RX (|+> 초기화)
        schedule.instructions[tick].append(
            Instruction(tick, "RX", data_qubits)
        )
        tick += 1

        pass