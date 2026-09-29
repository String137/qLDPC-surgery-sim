"""
merged_example.py — MergedHighRateSurgeryPatch 를 클릭 가능한 HTML 로 렌더링한다.

    viz/out_merged.html   C1=[[45,9,3]] (d=3)  +  C2=[[100,4,5]] (d=5)  를
                           앤실라 하나로 merge, 각 코드의 첫 X-logical 곱을 측정.

test.py 의 test21() 과 동일한 구성 (C1, C2 의 distance 가 다를 때
merge 에 참여 안 한 쪽의 logical distance 가 보존되는지 보기 위한 예제).

실행:  python -m viz.merged_example
"""

from __future__ import annotations

import os
import numpy as np

from core.CSS import LinearCode, HGPCode
from core.codes import HGPCodePatch
from surgery.surgery import MergedHighRateSurgeryPatch, Homomorphism
from viz.hgp_viz import render

OUT_DIR = os.path.dirname(__file__)


def build_merged() -> MergedHighRateSurgeryPatch:
    # C1 = HGP([6,3,3]) = [[45,9,3]]  (d=3)
    Cb_C1 = LinearCode.from_generator([[1, 0, 0, 1, 1, 0],
                                       [0, 1, 0, 0, 1, 1],
                                       [0, 0, 1, 1, 0, 1]])          # [6,3,3]
    # C2 = HGP([8,2,5]) = [[100,4,5]]  (d=5)
    Cb_C2 = LinearCode.from_generator([[1, 1, 1, 1, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 1, 1, 1]])    # [8,2,5]

    # 앤실라 base = rep[m]. m 은 두 데이터 코드의 첫 logical weight 중 큰 쪽으로
    # 맞춰서 |Γ1| = 1 을 노린다.
    m = max(int(np.asarray(Cb_C1.G, dtype=int)[0].sum()),
            int(np.asarray(Cb_C2.G, dtype=int)[0].sum()))
    Hrep = np.zeros((m - 1, m), dtype=int)
    for i in range(m - 1):
        Hrep[i, i] = 1
        Hrep[i, i + 1] = 1
    Cb_A = LinearCode(Hrep)                                          # rep[m], k=1

    Ca_C1 = LinearCode(np.asarray(Cb_C1.H, dtype=int).T)
    Ca_C2 = LinearCode(np.asarray(Cb_C2.H, dtype=int).T)
    Ca_A = LinearCode([[1, 1]])

    # 나란히 배치: C1 -> C2 -> A, 서로 겹치지 않게 폭만큼 옆으로 이동.
    gap = 2.0
    css_C1_tmp = HGPCode(Ca_C1, Cb_C1)
    css_C2_tmp = HGPCode(Ca_C2, Cb_C2)
    width1 = css_C1_tmp.nb + css_C1_tmp.rb
    width2 = css_C2_tmp.nb + css_C2_tmp.rb

    C1 = HGPCodePatch(HGPCode(Ca_C1, Cb_C1), patch_name="d1", offset=(0.0, 0.0))
    C2 = HGPCodePatch(HGPCode(Ca_C2, Cb_C2), patch_name="d2",
                       offset=(width1 + gap, 0.0))
    A = HGPCodePatch(HGPCode(Ca_A, Cb_A), patch_name="anc",
                      offset=(width1 + width2 + 2 * gap, 0.0))

    # ltol : L1(Cb_A) -> L1(Cb_Ci) 만. 나머지 logical 은 안 건드림.
    ltol1 = np.zeros((Cb_C1.k, 1), dtype=int); ltol1[0, 0] = 1
    ltol2 = np.zeros((Cb_C2.k, 1), dtype=int); ltol2[0, 0] = 1
    g1 = Homomorphism(Cb_A, Cb_C1, ltol1)
    g2 = Homomorphism(Cb_A, Cb_C2, ltol2)

    S = MergedHighRateSurgeryPatch(C1, C2, A, [(0, 0, 0, g1, g2)])
    S.init_logical_representatives(minimize=False)
    return S


def main():
    S = build_merged()
    S.report()
    path = render(S, os.path.join(OUT_DIR, "out_merged.html"),
                  title=f"merged high-rate surgery  "
                        f"[[{S.css.n},{S.css.k}]]  (d1=3, d2=5)")
    print("wrote", path)


if __name__ == "__main__":
    main()
