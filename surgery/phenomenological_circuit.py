from core.primitives import *
from core.codes import *
from surgery.circuit import *
from surgery.surgery import MergedCodeWithGraph, HighRateSurgeryPatch


class MemoryCircuitScheduler:
    def __init__(self, patch: CodePatch, rounds: int, p_data: float, p_meas: float):
        self.patch = patch
        self.rounds = rounds
        self.p_data = p_data
        self.p_meas = p_meas

    def build(self) -> CircuitSchedule:
        data_qubits = self.patch.data_qubits()
        
        x_checks = self.patch.x_checks()
        z_checks = self.patch.z_checks()
        all_checks = x_checks + z_checks

        schedule = CircuitSchedule(set(data_qubits))
        schedule.qubit_idxs = {q: self.patch.get_idx(q) for q in data_qubits}
        tick = 0

        # tick 0: RX (|+> 초기화)
        schedule.instructions[tick].append(
            Instruction(tick, "RX", data_qubits)
        )
        tick += 1

        # tick 1: noiseless reference measurement (round_0)
        schedule.instructions[tick].append(
            Instruction(tick, "MPP", all_checks, gate_args=[0], label="round_0")
        )
        tick += 1

        # tick 2 ~ rounds: noisy MPP X+Z checks
        for r in range(1, self.rounds + 1):
            schedule.instructions[tick].append(
                Instruction(tick, "Z_ERROR", data_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "X_ERROR", data_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "MPP", all_checks, gate_args=[self.p_meas], label=f"round_{r}")
            )
            tick += 1

        # 마지막: data error + MX data qubits
        schedule.instructions[tick].append(
            Instruction(tick, "Z_ERROR", data_qubits, gate_args=[self.p_data])
        )
        schedule.instructions[tick].append(
            Instruction(tick, "X_ERROR", data_qubits, gate_args=[self.p_data])
        )
        schedule.instructions[tick].append(
            Instruction(tick, "MX", data_qubits, gate_args=[0], label="final_data")
        )

        # 연속 check 비교 detector는 add_detectors가 자동으로 추가
        # 마지막 MX 결과로 X check를 다시 만들어서 마지막 round와 비교
        final_tick = tick            # MX 넣은 tick
        last_round_tick = tick - 1   # 마지막 noisy MPP tick
        for chk in x_checks:
            schedule.extra_detectors.append(
                [(last_round_tick, chk)] + [(final_tick, q) for q in chk.qubits]
            )

        schedule.qubit_coords = {q: self.patch.local_coord(q) for q in data_qubits}
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
        # 자동으로 잡히는 것들 (add_detectors):
        # - pre rounds 내부: C1, C2 X/Z check 연속 비교 → 같은 객체라서 자동
        # - merge rounds 내부:
        #   - C1, C2 X check: deform 안 되므로 같은 객체 → 자동
        #   - ancilla X check (A_v): merge 첫 round는 detector 없음 (의도),
        #     merge_1부터 연속 비교 → 자동
        #   - deformed Z check: 같은 객체 연속 → 자동
        #   - ancilla Z check (B_p): 같은 객체 연속 → 자동 (ref → merge_0 포함)
        # - post rounds 내부: C1, C2 X/Z check 연속 비교 → 자동
        # - pre → merge X check: C1, C2 X check deform 안 되므로 같은 객체 → 자동
        # - merge → post X check: C1, C2 X check 같은 객체 → 자동

        # extra_detectors에 추가해야 하는 것들:
        # 경계는 두 군데: pre→merge, merge→post(split)
        #
        # [pre → merge 경계]
        # - C1, C2 Z check: merge에서 ancilla qubit이 추가된 deformed Z check로 바뀜
        #                   → 다른 객체라 자동 감지 불가
        #                   → extra: (pre 마지막 tick, C1/C2 Z check)
        #                            (merge 첫 tick, deformed Z check) 쌍으로 detector 추가
        # - ancilla Z check (B_p): pre round에서 reference용도로 noiseless 측정함
        #                           → ref → merge_0 자동으로 잡힘
        # - ancilla X check (A_v): merge 첫 round는 detector 없음 (의도된 것,
        #                           첫 측정값이 XX measurement outcome의 일부)
        #
        # [merge → post 경계 (split)]
        # - C1, C2 Z check: deformed Z check에서 원래 Z check로 돌아옴
        #                   → 다른 객체라 자동 감지 불가
        #                   → split detector:
        #                     (마지막 merge tick, deformed Z check)
        #                     XOR (split tick, ancilla qubit MZ)  ← Qubit으로 저장
        #                     XOR (post 첫 tick, C1/C2 Z check) == 0
        # - ancilla Z check (B_p): post round에서 측정 안 함 → detector 불필요
        # - ancilla X check (A_v): post round에서 측정 안 함 → detector 불필요

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

    def __init__(self, merged: MergedCodeWithGraph, pre_round: int, merge_round: int, post_round: int, p_data: float, p_meas: float):
        assert post_round >= 1, "split detector가 post 첫 round의 Z check를 쓰므로 post_round >= 1 이어야 함"
        self.merged = merged
        self.C1 = merged.C1
        self.C2 = merged.C2
        self.p_data = p_data
        self.p_meas = p_meas
        self.pre_round = pre_round
        self.merge_round = merge_round
        self.post_round = post_round

    def build(self) -> CircuitSchedule:
        c_data_qubits = self.merged.direct_sum.data_qubits()
        a_data_qubits = self.merged.anc_graph.data_qubits()
        all_qubits = self.merged.data_qubits()
 

        c1_x_checks = self.C1.x_checks()
        c1_z_checks = self.C1.z_checks()
        c2_x_checks = self.C2.x_checks()
        c2_z_checks = self.C2.z_checks()
        anc_z_checks = self.merged.anc_graph.z_checks()  # B_p

        merged_x_checks = self.merged.x_checks()  # C1, C2 X check + A_v
        merged_z_checks = self.merged.z_checks()  # deformed Z check + B_p


        # c_z_checks: C1, C2 Z check (direct_sum 순서와 동일)
        # merged_z_checks[i]가 c_z_checks[i]의 deformed 버전
        c_z_checks = c1_z_checks + c2_z_checks

        schedule = CircuitSchedule(set(all_qubits))
        schedule.qubit_idxs = {q: self.merged.get_idx(q) for q in all_qubits}
        tick = 0

        # tick 0: 초기화
        # C1, C2 data qubit: RX (|+>)
        # ancilla qubit: R (|0>)
        schedule.instructions[tick].append(
            Instruction(tick, "RX", c_data_qubits)
        )
        schedule.instructions[tick].append(
            Instruction(tick, "R", a_data_qubits)
        )
        tick += 1

        # tick 1: noiseless reference measurement
        # C1, C2의 X, Z check + ancilla Z check (B_p)
        # ancilla X check (A_v)는 reference 없음 (첫 측정값이 XX measurement outcome)
        schedule.instructions[tick].append(
            Instruction(tick, "MPP",
                c1_x_checks + c2_x_checks + c1_z_checks + c2_z_checks,
                gate_args=[0], label="ref")
        )
        tick += 1

        # pre-rounds: C1, C2 각자 EC
        for r in range(self.pre_round):
            schedule.instructions[tick].append(
                Instruction(tick, "Z_ERROR", c_data_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "X_ERROR", c_data_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "MPP",
                    c1_x_checks + c2_x_checks + c1_z_checks + c2_z_checks,
                    gate_args=[self.p_meas], label=f"pre_{r}")
            )
            
            tick += 1
        # merge 직전 tick에서 reference로 anc_z_check 측정
        schedule.instructions[tick-1].append(
            Instruction(tick-1, "MPP",
            anc_z_checks,
            gate_args=[0], label="ref")
        )

        # merge-rounds: merged code EC
        for r in range(self.merge_round):
            schedule.instructions[tick].append(
                Instruction(tick, "Z_ERROR", all_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "X_ERROR", all_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "MPP",
                    merged_x_checks + merged_z_checks,
                    gate_args=[self.p_meas], label=f"merge_{r}")
            )
            tick += 1

        # split: ancilla qubit MZ 측정 (|0> init과 같은 basis)
        schedule.instructions[tick].append(
            Instruction(tick, "M", a_data_qubits, gate_args=[self.p_meas], label="split")
        )
        # split -> post 넘어갈 때는 tick 안넘어가야함.

        # post-rounds: C1, C2 각자 EC
        for r in range(self.post_round):
            schedule.instructions[tick].append(
                Instruction(tick, "Z_ERROR", c_data_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "X_ERROR", c_data_qubits, gate_args=[self.p_data])
            )
            schedule.instructions[tick].append(
                Instruction(tick, "MPP",
                    c1_x_checks + c2_x_checks + c1_z_checks + c2_z_checks,
                    gate_args=[self.p_meas], label=f"post_{r}")
            )
            tick += 1

        # 마지막: noiseless MX data qubits
        schedule.instructions[tick].append(
            Instruction(tick, "MX", c_data_qubits, gate_args=[0], label="final_data")
        )

        # final detector: 마지막 MX 결과로 C1, C2 X check를 다시 만들어서 마지막으로 잰 round와 비교
        final_tick = tick
        last_round_tick = tick - 1
        for chk in c1_x_checks + c2_x_checks:
            schedule.extra_detectors.append(
                [(last_round_tick, chk)] + [(final_tick, q) for q in chk.qubits]
            )

        # tick 계산
        pre_last_tick = 1 + self.pre_round
        merge_first_tick = 1 + self.pre_round + 1
        merge_last_tick = 1 + self.pre_round + self.merge_round
        split_tick = 1 + self.pre_round + self.merge_round + 1
        post_first_tick = split_tick   # split이랑 post 첫 round는 같은 tick

        # extra_detectors 1: pre → merge Z check 경계
        # f0[i].weight() > 0인 경우만 deformed된 Z check
        for i, f0_check in enumerate(self.merged._f0):
            if f0_check.weight() > 0:
                original_z = c_z_checks[i]
                deformed_z = merged_z_checks[i]
                schedule.extra_detectors.append([
                    (pre_last_tick, original_z),
                    (merge_first_tick, deformed_z),
                ])

        # extra_detectors 2: split detector (merge → post Z check 경계)
        # deformed Z check XOR ancilla qubit MZ XOR post 첫 round Z check == 0
        for i, f0_check in enumerate(self.merged._f0):
            if f0_check.weight() > 0:
                deformed_z = merged_z_checks[i]
                original_z = c_z_checks[i]
                detector = [
                    (merge_last_tick, deformed_z),
                    (post_first_tick, original_z),
                ]
                for q in f0_check.qubits:
                    detector.append((split_tick, q))  # Qubit으로 저장
                schedule.extra_detectors.append(detector)

        schedule.qubit_coords = {q: self.merged.local_coord(q) for q in all_qubits}
        return schedule


class HighRateSurgeryCircuitScheduler:
    """
    Phenomenological circuit for (Merged) HighRateSurgeryPatch.

    Same pre/merge/post structure and detector strategy as
    SurgeryCircuitScheduler above, generalized from the ancilla-graph
    MergedCodeWithGraph to the ancilla-HGP-code HighRateSurgeryPatch /
    ConePatch (Cs = list of data blocks, A = ancilla HGP code, f0/f1t the
    chain maps ConePatch already stores):

        pre-rounds  : each data block C_j measured alone, its own (undeformed)
                      x/z checks
        merge-rounds: the full merged code (S.x_checks()/z_checks()) measured
                      -- data blocks' Z-checks deformed by f0 (ancilla qubits
                      added), A's X-checks deformed by f1t (data qubits
                      added), everything else unchanged
        split       : A's qubits measured out (same basis as their |0> init)
        post-rounds : each data block C_j measured alone again

    Initialization: data qubits RX (|+>), A's qubits R (|0>).

    Detectors (mirrors SurgeryCircuitScheduler's reasoning):
        - auto (same Check object measured on consecutive ticks):
            each block's X-checks (never deformed) across every boundary;
            A's own Z-checks (never deformed, appear in merged_z_checks
            unchanged) once a noiseless "ref" measurement is taken right
            after A's reset -- ref -> merge_0 links up automatically.
        - NOT auto, by design: A's X-checks (partial1t) only appear as new
          composite objects f1t[i]+partial1t[i] starting at merge round 0 --
          no earlier reference exists, so their first outcome is not a
          detector; it's part of the measured logical operator.
        - extra_detectors (added explicitly, one deformed Z-check i at a
          time, using S._f0[i].weight() > 0 to find which ones are deformed):
            pre -> merge   : (pre_last_tick, original) vs (merge_first_tick, deformed)
            merge -> post  : (merge_last_tick, deformed) XOR (split_tick, f0 support qubits)
                              XOR (post_first_tick, original) == 0
    """

    def __init__(self, S: HighRateSurgeryPatch, pre_round: int, merge_round: int,
                 post_round: int, p_data: float, p_meas: float):
        assert post_round >= 1, "split detector가 post 첫 round의 Z check를 쓰므로 post_round >= 1 이어야 함"
        self.S = S
        self.Cs = S.Cs
        self.A = S.A
        self.p_data = p_data
        self.p_meas = p_meas
        self.pre_round = pre_round
        self.merge_round = merge_round
        self.post_round = post_round

    def build(self) -> CircuitSchedule:
        c_data_qubits = [q for C in self.Cs for q in C.data_qubits()]
        a_data_qubits = self.A.data_qubits()
        all_qubits = self.S.data_qubits()

        c_x_checks = [chk for C in self.Cs for chk in C.x_checks()]
        c_z_checks = [chk for C in self.Cs for chk in C.z_checks()]
        anc_z_checks = self.A.z_checks()          # partial0, unchanged in the merge

        merged_x_checks = self.S.x_checks()
        merged_z_checks = self.S.z_checks()

        schedule = CircuitSchedule(set(all_qubits))
        schedule.qubit_idxs = {q: self.S.get_idx(q) for q in all_qubits}
        tick = 0

        # tick 0: init. data qubits RX (|+>), ancilla qubits R (|0>)
        schedule.instructions[tick].append(Instruction(tick, "RX", c_data_qubits))
        schedule.instructions[tick].append(Instruction(tick, "R", a_data_qubits))
        tick += 1

        # tick 1: noiseless reference measurement of the (undeformed) data checks
        schedule.instructions[tick].append(
            Instruction(tick, "MPP", c_x_checks + c_z_checks, gate_args=[0], label="ref"))
        tick += 1

        # pre-rounds: each block measured alone
        for r in range(self.pre_round):
            schedule.instructions[tick].append(
                Instruction(tick, "Z_ERROR", c_data_qubits, gate_args=[self.p_data]))
            schedule.instructions[tick].append(
                Instruction(tick, "X_ERROR", c_data_qubits, gate_args=[self.p_data]))
            schedule.instructions[tick].append(
                Instruction(tick, "MPP", c_x_checks + c_z_checks, gate_args=[self.p_meas], label=f"pre_{r}"))
            tick += 1
        # reference for A's own Z-checks, taken right before merge starts
        schedule.instructions[tick - 1].append(
            Instruction(tick - 1, "MPP", anc_z_checks, gate_args=[0], label="ref"))

        # merge-rounds: full merged code
        for r in range(self.merge_round):
            schedule.instructions[tick].append(
                Instruction(tick, "Z_ERROR", all_qubits, gate_args=[self.p_data]))
            schedule.instructions[tick].append(
                Instruction(tick, "X_ERROR", all_qubits, gate_args=[self.p_data]))
            schedule.instructions[tick].append(
                Instruction(tick, "MPP", merged_x_checks + merged_z_checks,
                            gate_args=[self.p_meas], label=f"merge_{r}"))
            tick += 1

        # split: measure A's qubits out (same basis as their |0> init)
        schedule.instructions[tick].append(
            Instruction(tick, "M", a_data_qubits, gate_args=[self.p_meas], label="split"))
        # no tick increment: split -> post stays on the same tick boundary

        # post-rounds: each block measured alone again
        for r in range(self.post_round):
            schedule.instructions[tick].append(
                Instruction(tick, "Z_ERROR", c_data_qubits, gate_args=[self.p_data]))
            schedule.instructions[tick].append(
                Instruction(tick, "X_ERROR", c_data_qubits, gate_args=[self.p_data]))
            schedule.instructions[tick].append(
                Instruction(tick, "MPP", c_x_checks + c_z_checks, gate_args=[self.p_meas], label=f"post_{r}"))
            tick += 1

        # final: noiseless MX of the data qubits
        schedule.instructions[tick].append(
            Instruction(tick, "MX", c_data_qubits, gate_args=[0], label="final_data"))

        # final detector: 마지막 MX 결과로 data block X check를 다시 만들어서 마지막으로 잰 round와 비교
        final_tick = tick
        last_round_tick = tick - 1
        for chk in c_x_checks:
            schedule.extra_detectors.append(
                [(last_round_tick, chk)] + [(final_tick, q) for q in chk.qubits]
            )

        pre_last_tick = 1 + self.pre_round
        merge_first_tick = 1 + self.pre_round + 1
        merge_last_tick = 1 + self.pre_round + self.merge_round
        split_tick = 1 + self.pre_round + self.merge_round + 1
        # split does NOT get its own tick (build() deliberately reuses `tick`
        # for the post-round loop right after appending the split M
        # instruction, see the "no tick increment" comment above) -- so
        # post_0 is measured at split_tick itself, not split_tick+1. Using
        # +2 here (an easy copy-paste trap from the split_tick formula, and
        # the original bug: SurgeryCircuitScheduler above has the identical
        # off-by-one) points post_first_tick at the *final_data* MX tick
        # instead, where no Z-check MPP was ever measured -- get_meas_idx()
        # then silently returns an empty list rather than erroring, so the
        # resulting extra_detector is missing its cancelling term and stim
        # correctly reports "non-deterministic detectors" the first time
        # anything actually validates it (e.g. detector_error_model()).
        post_first_tick = split_tick

        # pre -> merge boundary: deformed Z-checks
        for i, f0_check in enumerate(self.S._f0):
            if f0_check.weight() > 0:
                schedule.extra_detectors.append([
                    (pre_last_tick, c_z_checks[i]),
                    (merge_first_tick, merged_z_checks[i]),
                ])

        # merge -> post boundary (split): deformed Z-checks
        for i, f0_check in enumerate(self.S._f0):
            if f0_check.weight() > 0:
                detector = [
                    (merge_last_tick, merged_z_checks[i]),
                    (post_first_tick, c_z_checks[i]),
                ]
                for q in f0_check.qubits:
                    detector.append((split_tick, q))
                schedule.extra_detectors.append(detector)

        schedule.qubit_coords = {q: self.S.local_coord(q) for q in all_qubits}
        return schedule