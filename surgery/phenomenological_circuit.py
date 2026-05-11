import stim
from collections import defaultdict
from core.primitives import *
from core.codes import *
import numpy as np
from surgery.circuit import *


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

        # tick 1: DEPOLARIZE1 + MPP X checks only (noisy, label="round_0")
        # schedule.instructions[tick].append(
        #     Instruction(tick, "DEPOLARIZE1", data_qubits, gate_args=[self.p_data])
        # )
        schedule.instructions[tick].append(
            Instruction(tick, "MPP", x_checks, gate_args=[self.p_meas], label="round_0")
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
            Instruction(tick, "MPP", x_checks, gate_args=[self.p_meas], label="final")
        )
        schedule.instructions[tick].append(
            Instruction(tick, "MX", data_qubits, gate_args=[0], label="final_data")
        )

        return schedule