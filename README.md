# qLDPC-surgery

두 개의 surface-code patch를 ZZ surgery로 연결하는 Stim 회로를 생성하고,
detector 동작과 fault 조합을 분석하는 실험 저장소다.

현재 기준 대표 회로는 surgery/surgery_circuit.py 의
build_zz_surgery_circuit(d=3, rounds=3, p_data=0.01, p_meas=0.01) 이다.

## 현재 확정 상태 요약

이번 작업에서 아래가 반영되어 현재 기준으로 고정됐다.

1. merge -> split/post 경계 detector 보강
- split tick에서 seam 경계 detector 4개만 있던 문제를 수정했다.
- patch1/patch2 Z-check 8개에 대해, merge 마지막 측정값과 split 측정값을 비교하는 detector를 추가했다.
- 결과적으로 split 경계 detector가 4개 -> 12개(= seam 4 + patch 8)로 증가했다.

2. min-fault 기준 명확화
- min-fault는 물리 fault 조합 기준으로 고정한다.
- 사용 기준: detector XOR = 0, observable XOR != 0.
- 현재 회로 기준 최소 weight는 3, 최소 조합 수는 112.

3. detslice 생성 안정화
- detslice tick 파일은 viz/visualize_circuit.py 가 생성한다.
- tick 정렬은 파일 tick과 회로 tick을 1:1로 맞춘다.
- 초기 tick(0~2)에서도 rotated surface 배경이 빠지지 않도록 보정했다.

## 프로젝트 구조

리팩토링 기준 핵심 디렉터리는 아래와 같다.

- core/
  - primitives.py
  - codes.py
  - CSS.py
- surgery/
  - surgery.py
  - surgery_circuit.py
- analysis/
  - fault_analysis.py
  - min_fault.py
- viz/
  - visualize_circuit.py
  - visualize_surgery.py
  - visualize_dem_graph.py
  - visualize_dem_graph_3d.py

각 디렉터리는 패키지로 동작하도록 __init__.py 를 포함한다.

## 디렉터리/파일 가이드

### 회로 생성 핵심

- surgery/surgery_circuit.py
  - ZZ surgery Stim 회로 빌더.
  - detector, observable, phase(pre/merge/split/post/final) 정의의 기준 파일.

- surgery/surgery.py
  - patch 병합 구조와 ancilla 포함 merge 구성 유틸.

- core/codes.py
  - RotatedSurfaceCode 등 코드 패치 모델.

- core/primitives.py
  - Qubit, Check, QubitRegistry 등 기초 자료구조.

### 분석/검증

- analysis/min_fault.py
  - shortest_graphlike_error 기반 최소 fault 확인용 스크립트.

- check_undetected_minfault.py
  - detector/observable 관점에서 error 항목을 요약해 보는 보조 스크립트.
  - 참고 출력으로만 사용하고, 최종 min-fault 결론은 물리 fault 조합 기준 파일을 따른다.

- test_fault_decoding.py, test_fault_distance.py
  - fault 주입 시 detector 반응/거리 해석 실험 스크립트.

### 시각화

- viz/visualize_circuit.py
  - timeline SVG + tick별 detslice SVG 생성.
  - outputs/detslice/all/detslice_d3_tickXX.svg 를 만든다.

- visualize_timeslice.py
  - timeslice 시각화(별도 출력).

- viz/visualize_surgery.py
  - phase/round 단위 stabilizer 배치 시각화.

## 출력 파일 기준

### 회로 스냅샷

- outputs/zz_surgery_d3.stim
  - 현재 회로 텍스트 스냅샷.

### min-fault 결과

- outputs/surgery_min_fault_combos_d3_r3_allticks.csv
  - 최소 weight 조합의 원본 테이블.
  - 컬럼: combo_id, weight, obs_bits, tick, inst, pauli, q, x, y

- outputs/surgery_min_fault_combos_d3_r3_allticks_summary.md
  - CSV 요약 문서.
  - q 배치 ASCII, 조합 분포, 112개 상세 조합 표 포함.

### 시각화

- outputs/timeline_d3.svg
- outputs/detslice/all/detslice_d3_tick00.svg ~ tick20.svg

## 재현 실행 순서

아래 순서로 실행하면 현재 문서 상태를 재현할 수 있다.

```bash
python3 -m viz.visualize_circuit
python3 analysis/min_fault.py
```

중요:

- 리팩토링 후 `viz/visualize_circuit.py` 같은 파일을 직접 실행하면
  패키지 경로 해석 문제로 ModuleNotFoundError가 발생할 수 있다.
- 시각화 스크립트는 `python3 -m viz.visualize_circuit`처럼 모듈 실행을 권장한다.

min-fault 상세 CSV/요약을 다시 만들 때는 프로젝트에서 사용 중인 조합 계산 스크립트를 실행한다.
(현재 저장소에서는 outputs/surgery_min_fault_combos_d3_r3_allticks.csv,
 outputs/surgery_min_fault_combos_d3_r3_allticks_summary.md 를 기준 산출물로 사용한다.)

## 현재 회로 동작 단계(개요)

1. init: patch 논리 상태/기준 측정
2. pre: patch1/patch2 독립 stabilizer 반복
3. merge: seam 포함 stabilizer 반복
4. split: patch stabilizer + ancilla X 측정
5. post: patch1/patch2 stabilizer 반복
6. final: data readout + closing detector + observable include

## 주의사항

1. min-fault 결론은 반드시 물리 fault 조합 기준(detector XOR=0, observable XOR!=0)으로 해석한다.
2. detslice 문제를 수정할 때는 viz/visualize_circuit.py 를 기준으로 본다.
3. outputs 하위 파일은 스크립트 재실행 시 갱신되므로, 커밋 전 갱신 여부를 확인한다.

## TODO

추가 작업 목록은 TODO.md 를 참고.
