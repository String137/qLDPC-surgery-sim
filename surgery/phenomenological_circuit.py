import stim
from collections import defaultdict
from core.primitives import *
from core.codes import *
import numpy as np

class CircuitBuilder:
    """
    stim circuit과 meas_record를 관리
    """

    def __init__(self,):
        # measurement 정보들을 기록
        # tick이 몇인지, 어떤 check를 측정한 것인지, 몇 번째 측정인지 기록 
        self.meas_record = {}
        self.current_tick = 0
        self._circuit = stim.Circuit()  # private

    @property
    def circuit(self,):
        return self._circuit.copy()

    def append_meas(self, gate: str, targets: list, label: str = None, noisy: bool = False):
        raise NotImplementedError
        pass

    def append_gate(self, gate: str, targets: list):
        """measurement 없는 gate용. TICK, H, CNOT 등"""
        self._circuit.append(gate, targets)

    def get_rec(self, tick: int = None, target: int = None, label: str = None):
        # 조건을 만족하는 measurement가 현재 meas 기준으로 -몇번째인지.
        # 가장 마지막으로 측정한 meas: rec[-1]
        # raise NotImplementedError
        res = self.get_meas_idx(tick, target, label)
        new_res = []
        for r in res:
            new_res.append(stim.target_rec(r - len(self.meas_record)))
        return new_res
    
    def get_meas_idx(self, tick: int = None, target: int = None, label: str = None):
        # 조건을 만족하는 measurement가 몇번째 meas인지
        # 여러 개일 수 있어서 list로 반환
        raise NotImplementedError

class MemoryCircuitBuilder:
    """
    CodePatch에 대해 rounds동안의 Memory 실험을 위한 Phenomenological noise model 하의 circuit을 생성해줌.
    """
    def __init__(self, patch: CodePatch, rounds: int, p_data: float, p_meas: float):
        self.patch = patch
        self.rounds = rounds
        self.p_data = p_data
        self.p_meas = p_meas
        
        # check에 index 부여
        self.check_idx = {chk: i for i, chk in enumerate(patch.z_checks()+patch.x_checks())}
        self.idx_to_check = {i: chk for chk, i in self.check_idx.items()}

        self.cb = CircuitBuilder()
        

    def build(self):
        # all |+> state로 init
        # 첫 round에 X-check만 측정 (noisy)
        # data qubit error
        # round 2부턴 X,Z-check 모두 측정 (이전 round 측정 결과 있으면 detector 추가)
        pass
    