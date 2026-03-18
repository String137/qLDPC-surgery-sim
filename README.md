# qLDPC-surgery

두 개의 코드 패치를 병합하고, ZZ surgery 회로를 Stim으로 생성하고, detector 및 fault 분석을 수행하는 실험용 저장소다. 현재 기준의 대표 회로는 `surgery_circuit.py`의 `build_zz_surgery_circuit(...)`가 만드는 rotated surface code 기반 ZZ surgery 회로다.

## 프로젝트 구성

### 핵심 모델

- `primitives.py`
	- `Qubit`, `Check`, `QubitRegistry`를 정의한다.
	- 기하학적 qubit/check 표현과 Stim의 정수 인덱스 매핑을 분리하는 가장 낮은 레벨의 공용 모듈이다.

- `codes.py`
	- `CodePatch` 추상 인터페이스와 `RotatedSurfaceCode`, `TrivialCode` 같은 구체 코드를 정의한다.
	- 각 코드의 데이터 qubit, stabilizer, logical operator, Stim 회로 변환 기능의 출발점이다.

- `CSS.py`
	- 보다 일반적인 CSS 코드, BBCode, 거리 계산, 병합된 CSS 코드 실험용 클래스를 담고 있다.
	- `galois`, `docplex`, `cplex`, `networkx` 등을 이용한 대수적 분석 코드가 들어 있다.

### Surgery 구성

- `surgery.py`
	- 코드 패치 둘 이상을 연결하는 병합 구조를 정의한다.
	- `MergedPatch`, `MergedPatchWithAncilla`, 경계 쌍 생성 함수, ZZ/XX 병합 생성 함수를 제공한다.

- `surgery_circuit.py`
	- 현재 저장소의 대표적인 Stim 회로 생성기다.
	- `MeasurementTracker`, `SurgeryCircuitBuilder`, `build_zz_surgery_circuit(...)`를 통해 init/pre/merge/split/post/verify 단계를 가지는 ZZ surgery 회로를 만든다.
	- detector 선언, observable 포함, MPP 기반 stabilizer 측정, 데이터/측정 오류 모델이 여기에 들어 있다.
	- 관측량은 `init_LZ1 XOR init_LZ2 XOR seam_delta(first XOR last)` 형태로 정의되어,
	  초기 logical 상태와 ZZ 측정 결과의 불일치 여부를 직접 나타낸다.

## 예제와 실험 코드

- `example_ancilla_merge.py`
	- trivial ancilla를 사이에 둔 ZZ/XX merge 예제를 출력한다.
	- ancilla를 끼운 세 코드 병합 구조를 빠르게 확인할 때 쓴다.

- `test.py`
	- BBCode 기반 병합과 거리 계산을 직접 실행해보는 실험 스크립트다.

- `test_fault_decoding.py`
	- 회로에 단일 fault를 주입했을 때 detector syndrome이 어떻게 바뀌는지 보는 간단한 fault 테스트다.

- `test_fault_distance.py`
	- 특정 단일 fault가 어떤 detector 식들을 트리거하는지 사람이 읽을 수 있는 형태로 설명하는 분석 스크립트다.

## 분석 및 디코딩

- `fault_analysis.py`
	- detector error model을 기반으로 logical error rate, threshold sweep, minimum-weight logical fault 탐색, fault pattern 시각화를 수행한다.
	- `pymatching`을 이용한 MWPM 디코딩 경로가 여기에 있다.

- `min_fault.py`
	- Stim detector error model의 `shortest_graphlike_error()`를 이용해 최소 logical fault를 찾고 spacetime 위에 시각화한다.

## 시각화 도구

- `visualize.py`
	- Tanner graph, 병합 코드 구조, ancilla merge 등의 정적 시각화를 담당한다.

- `visualize_circuit.py`
	- Stim 회로 timeline SVG와 tick별 detector slice SVG를 생성한다.
	- 현재 저장소에서 detector 관련 그림을 가장 바로 얻을 수 있는 스크립트다.

- `visualize_surgery.py`
	- surgery의 각 phase와 round에서 활성화되는 stabilizer를 2D 격자에서 보여준다.

- `visualize_timeslice.py`
	- Stim 회로 문자열을 파싱해서 tick별 MPP 측정을 공간적으로 배치해 보여준다.

## 노트북과 문서

- `BBCode.ipynb`
	- BBCode/CSS 관련 대화형 실험용 노트북이다.

- `ANCILLA_MERGE_README.md`
	- ancilla를 끼운 merge 방식에만 집중한 보조 설명 문서다.

## 출력 폴더

- `outputs/`
	- 시각화 스크립트들이 생성한 산출물을 저장한다.
	- 현재는 `timeline_d3.svg`, `detslice_d3_tick*.svg` 같은 circuit/detector slice 결과를 두는 용도로 쓰인다.

## 보통 어디서 시작하면 되는가

- 회로를 만들고 싶으면 `surgery_circuit.py`
- 코드 구조를 보고 싶으면 `codes.py`, `surgery.py`
- detector/fault 분석을 하고 싶으면 `fault_analysis.py`, `min_fault.py`
- 그림을 보고 싶으면 `visualize_circuit.py`, `visualize_surgery.py`, `visualize_timeslice.py`

## 현재 ZZ surgery 회로 단계 요약

기본 빌더(`build_zz_surgery_circuit`)가 만드는 회로는 다음 순서로 동작한다.

1. Init logical (noiseless): patch1/patch2 logical Z를 먼저 측정해 초기 기준값을 저장
2. Pre stabilizer init (noiseless): pre phase의 기준 stabilizer 측정
3. Pre rounds: patch1/patch2 독립 Z stabilizer 반복 측정
4. Merge init (noiseless): seam 포함 merged stabilizer 기준 측정
5. Merge rounds: merged Z stabilizer 반복 측정
6. Split: ancilla X 단발 측정 + split detector
7. Post init (noiseless): 분리 후 patch별 stabilizer 기준 측정
8. Post rounds: patch1/patch2 독립 Z stabilizer 반복 측정
9. Verify logical (noiseless): split/post 이후 각 patch logical Z 보존성 detector 확인
10. Observable finalize: init logical 기준 ZZ 불일치 observable 포함

## 빠른 실행 예시

```bash
python visualize_circuit.py
python fault_analysis.py
python min_fault.py
python example_ancilla_merge.py
```

## detector 관련 참고

현재 워크스페이스에는 이미 `outputs/detslice_d3_tick04.svg` 같은 detector slice 시각화 파일들이 있다. 이것들은 각 tick에서 어떤 연산과 detector coverage가 나타나는지를 보여준다.

반면 일반적인 의미의 detector error graph는 Stim detector error model과 관련 패키지(`stim`, `pymatching` 등)를 실제로 불러와서 생성해야 한다. 이 저장소에는 그 분석 로직의 재료가 이미 `fault_analysis.py`, `min_fault.py`에 들어 있지만, 그래프 자체를 별도 이미지로 뽑는 전용 스크립트는 아직 없다.

# TODO
detslice_d3_tickxx.svg 파일 보는게 나을듯.
근데, 여러 게이트가 겹쳐있어서 error있는 tick이랑 gate랑 분리하는게 나을듯.
weight 4짜리 MPP는 Z모양으로 이어진건지?