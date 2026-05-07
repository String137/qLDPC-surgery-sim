# qLDPC-surgery

일반적인 qLDPC 코드에 대한 fault-tolerant surgery를 구현하는 연구 저장소.
최종 목표는 surface code에 국한되지 않고, **general CSS/qLDPC 코드**에 대한 surgery 회로 생성 및 분석이다.

현재는 `SurfaceCode`를 테스트베드로 사용하며 (`test.py` 참고), `test.ipynb`에는 두 개의 LP/LDPC 코드를 auxiliary graph로 merge하는 예시가 포함되어 있다. `ConePatch` 기반 XX/ZZ surgery 구조를 구현 중이며, 현재 ZZ surgery 회로 생성은 `surgery/surgery_circuit.py`의 `SurgeryCircuitBuilder`가 담당한다. `surgery/surgery_circuit_cl.py`는 LRC 스케줄 기반 CNOT 편성 프로토타입이며, `surgery/surgery_circuit_default.py`는 이 구조를 기반으로 하되 인접 timestep 사이마다 detector를 정의하고 phenomenological error model에 집중하는 기본 경로로 설정되어 있다.


# TODO

- https://arxiv.org/pdf/2510.14895 이 논문 Figure 6처럼 X check meas 시작과 끝에 추가하고, detector는 각 round사이로 수동으로 추가하자.

# Link
- [Universal adapters between quantum LDPC codes](https://arxiv.org/pdf/2410.03628#page=14)
- [Fast and fault-tolerant logical measurements: Auxiliary hypergraphs and transversal surgery](https://arxiv.org/pdf/2510.14895)
- [QGPU: Parallel logic in quantum LDPC codes](https://arxiv.org/pdf/2603.05398#page=17.85)
- [Quantum LDPC Codes with Almost Linear
Minimum Distance](https://arxiv.org/pdf/2012.04068#page=6)
- [Low-overhead fault-tolerant quantum computation by gauging logical operators](https://arxiv.org/pdf/2410.02213v1)

## 프로젝트 구조

```
core/           — 기초 자료구조 및 코드 모델
surgery/        — 회로 생성 (pre/merge/split/post 전 주기)
analysis/       — fault 분석, min-fault 탐색
viz/            — Stim 회로 시각화
```

각 디렉터리는 `__init__.py`를 포함한 패키지다.

---

## 클래스 목록 및 역할

### `core/primitives.py`

| 클래스 | 역할 |
|--------|------|
| `Qubit` | 단일 큐빗. `(patch_name, idx, local_coord)`로 identity 결정. frozen dataclass. |
| `Check` | Pauli stabilizer check. `qubits: frozenset[Qubit]`, `pauli: "X"/"Z"`. `+` 연산으로 mod-2 합 지원. |
| `QubitRegistry` | `Qubit → int` stim index 매핑. `register()` 호출 순서대로 index 부여. `checks_to_matrix()`로 parity-check 행렬 생성. |

---

### `core/CSS.py`

행렬 레벨의 CSS 코드 표현 및 파라미터 계산. `CodePatch`와는 독립적이며, `galois.GF(2)` 행렬로 직접 동작한다. `CodePatch.make_css()`가 내부적으로 `CSS` 인스턴스를 생성해 `get_distance()`, `get_k()` 계산에 사용한다.

| 클래스 | 역할 |
|--------|------|
| `CSS` | CSS 코드의 기반 클래스. `Gx`, `Gz` (generator matrix) 입력 → `get_center()`로 실제 parity-check `Hx`, `Hz` 도출 (null space 소거). `calculate_k()`로 논리 큐빗 수 계산. `get_logical_representatives()`로 logical X/Z operator 추출 (Ker/Rowspace 몫공간). `get_distance()`는 CPLEX ILP로 최소 weight logical operator 계산. |
| `LPCode` | `CSS` 상속. LP code ring 매트릭스를 `Poly` 다항식으로 구성하고 `to_binary()`로 binary parity-check 행렬을 생성. `Ha`,`Hb`,`R` 입력으로 LP/LDPC code를 표현. |
| `BBCode` | `CSS` 상속. bivariate bicycle code. `(l, m)` 크기의 cyclic 그리드 위에서 다항식 `a(x,y)`, `b(x,y)`로 `Hx = [A\|B]`, `Hz = [B^T\|A^T]` 생성. `get_col_idx()`로 순환 행렬 인덱스 직접 접근. |
| `MergedCode` | `CSS` 상속. 두 `BBCode`의 block-diagonal direct sum. `add_edge_qubits()`로 edge qubit(|+⟩ 초기화) 추가. `measure_operator()`로 임의 Pauli operator를 측정하고 anti-commuting check를 정리. |

> `cheeger_constant(G)`: 소규모 그래프에서 Cheeger constant를 brute-force로 계산하는 유틸 함수.

---

### `core/codes.py`

| 클래스 | 역할 |
|--------|------|
| `CodePatch` (ABC) | 모든 코드 패치의 추상 기반. `data_qubits()`, `x_checks()`, `z_checks()` 인터페이스 정의. `h_x()`, `h_z()`로 parity-check 행렬 반환. `check2coord` 속성으로 ancilla 좌표 매핑 보관. `get_distance()`, `get_k()`로 코드 파라미터 계산. |
| `EmptyPatch` | 큐빗·check가 없는 빈 패치. 자리 채우기용. |
| `SurfaceCode` | 비회전 surface code (unrotated). `(x, y)` 반정수 좌표계. `check2coord` 자동 생성. |
| `RotatedSurfaceCode` | 회전 surface code (Fowler 2012 convention). distance `d`, `d²` data qubits. bulk/boundary plaquette 자동 생성. `boundary_qubits(side)`로 경계 큐빗 추출. `z_logicals()`, `x_logicals()` 제공. |
| `TrivialCode` | stabilizer check가 없는 코드 (`k = n`). ancilla 라인 표현에 사용. `direction="vertical"/"horizontal"` 선택 가능. |
| `LPCodePatch` | `core.CSS.LPCode` 인스턴스를 `CodePatch`로 감싸는 adapter. LP/LDPC ring 코드의 stabilizer checks를 `Check` 리스트로 노출. |
| `CCCode` | QGPU/quantum LDPC gauge code 구현 실험용 placeholder. 현재 구체 구현은 미완성. |
| `BBCodePatch` | 외부 `BBCode` 인스턴스를 `CodePatch`로 감싸는 adapter. `Hx`/`Hz` → `Check` list 변환. `load_logicals()`로 logical operator 로드. |
| `AncillaGraph` | auxiliary graph를 `CodePatch`로 표현. graph vertex를 X 체크로, graph edge를 데이터 큐빗으로 변환하여 `AncillaGraph` 기반 merge/port mapping에 사용. |

---

### `surgery/surgery.py`

| 클래스 | 역할 |
|--------|------|
| `MergedPatch` | patch1 + seam qubits + patch2 를 하나의 `CodePatch`로 합친 merged code. seam check(X/Z)를 받아서 전체 check list를 구성. 내부 `QubitRegistry`를 자체 보유. |
| `DirectSum` | 두 `CodePatch`의 direct sum. check는 단순 합집합. `ConePatch` 구성 시 내부적으로 사용. |
| `ConePatch` | homological measurement용 cone code C̃ = cone(f). C1 ⊕ C2 위에 ancilla `A`를 붙여 `f1t`, `f0` check를 추가해 구성. (Ide et al. PRX 2025 eq.13 구조) |
| `MergedCodeWithGraph` | `AncillaGraph` 기반 auxiliary graph를 사용해 두 LP/LDPC 코드 블록을 merge. logical X support와 ancilla graph perfect matching을 연결해 `ConePatch`를 구성. |

> `MergedPatchWithAncilla`, `build_zz_merged`, `build_xx_merged` 등 seam builder 함수들은 현재 주석 처리(비활성) 상태.

---

### `surgery/surgery_circuit.py` — ZZ surgery stim circuit

| 클래스 | 역할 |
|--------|------|
| `SurgeryCircuitBuilder` | pre / merge / split / post / final 단계로 ZZ surgery stim 회로를 구성. `surface_zz_boundary_pairs_with_ancilla` 기반 seam ancilla를 사용하며 `tqecd` 자동 detector annotation을 지원. |
| `build_zz_surgery_circuit()` | bare circuit과 annotated circuit을 반환하는 현재 ZZ surgery 실행 경로. |

> `surgery_circuit.py`는 현재 활성화된 ZZ surgery 회로 생성 경로이다. `surgery_circuit_cl.py`는 실험용 LRC 스케줄 프로토타입으로 유지되며, detector auto 탐색(`analysis/detector.py`)과 인접 timestep 간 detector 정의를 활용한다. `surgery_circuit_default.py`는 이 구조를 기반으로 하되 기본 경로에서는 phenomenological error model만 고려하도록 설계되어 있다.

---

### `surgery/surgery_circuit_cl.py` — LRC 스케줄 CNOT 편성 (프로토타입)

| 클래스 | 역할 |
|--------|------|
| `LRCCircuitBuilder` | 단일 코드 패치에 대한 LRC(Left-Right-Circuit) 스케줄 CNOT 편성. data qubit을 left/right로 partition하고 edge coloring으로 각 time step의 CNOT 목록을 결정. `build()`는 미완성(TODO 포함). (Strikis, Browne, Beverland arXiv:2603.05481) |
| `SurgeryCircuitCL` | ZZ surgery 전 주기(pre → merge → split/post → final readout) stim 회로. 현재 불완전한 상태로 방치 중. |

> `SurgeryCircuitCL`은 현재 사용하지 않는다. `LRCCircuitBuilder`는 LRC 스케줄 기반 회로 편성 실험을 위한 프로토타입이다.

---

### `surgery/surgery_circuit_default.py` — default phenomenological surgery circuit

`SurgeryCircuitCL`와 유사한 LRC 기반 회로 구조를 따르지만, detector를 인접 timestep 사이마다 명시적으로 정의하고 기본 경로에서는 phenomenological error model만 고려하는 회로 변형이다.

> `surgery_circuit_default.py`는 `surgery_circuit_cl.py` 기반이지만, 이 저장소의 기본 시뮬레이션 경로로 phenomenological noise model에 집중하고 detector auto annotation을 별도 경로로 처리한다.

---

### `analysis/`

| 파일 | 역할 |
|------|------|
| `fault_analysis.py` | 회로에서 fault 조합을 열거하고 detector/observable 기여를 분석. |
| `min_fault.py` | 물리 fault 조합 기준 min-fault 탐색 (detector XOR=0, observable XOR≠0). |

---

### `viz/`

| 파일 | 역할 |
|------|------|
| `visualize_circuit.py` | timeline SVG + tick별 detslice SVG 생성 (`outputs/detslice/all/`). |
| `visualize_dem_graph.py` / `visualize_dem_graph_3d.py` | DEM graph 2D/3D 시각화. |
| `visualize_surgery.py` | surgery patch 구조 시각화. |

---

## 최근 변경 이력 (2026-04)

- `CodePatch`에 `check2coord` 속성 추가: check → 좌표 매핑을 패치 레벨에서 보관. `SurfaceCode._build()`에서 자동 생성.
- `CodePatch.to_stim_circuit()` 제거(주석 처리): 회로 생성을 별도 빌더로 분리.
- `surgery/surgery_circuit.py` 추가: `SurgeryCircuitBuilder` 기반 ZZ surgery stim 회로 생성. bare/annotated 회로를 모두 반환하며 `tqecd` 자동 detector annotation을 지원.
- `LRCCircuitBuilder` 추가: edge coloring 기반 CNOT 스케줄 편성. LRC 스케줄형 회로 편성 프로토타입으로 유지.
- `ConePatch`, `DirectSum` 추가 (homological measurement 기반 qLDPC surgery 구현용).
- `MergedCodeWithGraph`, `AncillaGraph` 추가: auxiliary graph 기반 LP/LDPC 코드 merge 지원.
- `build_xx_cone_surface()` 추가: `SurfaceCode` 두 개와 ancilla로 XX cone surgery 구성.
- `MergedPatchWithAncilla`, seam builder 함수들 주석 처리(비활성).
- `SurgeryCircuitCL` 방치 — 실험용 프로토타입.

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
