"""
example.py — hgp_viz 사용 예제. 두 개의 HTML 을 만든다.

    viz/out_hgp.html       HGP 코드 단독:  [[45,9,3]]
    viz/out_surgery.html   surgery: 데이터 [[45,9,3]] + 앤실라, 열 하나에서 logical 하나 측정

실행:  python -m viz.example
"""

from __future__ import annotations

import os
import numpy as np

from core.CSS import LinearCode, HGPCode
from core.codes import HGPCodePatch
from surgery.surgery import HighRateSurgeryPatch, Homomorphism
from viz.hgp_viz import render

OUT_DIR = os.path.dirname(__file__)


def rep(m: int) -> LinearCode:
    """repetition-style 패리티 체크 [m, 1, m]."""
    H = np.zeros((m - 1, m), dtype=int)
    for i in range(m - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return LinearCode(H)


def build_hgp(patch_name: str = "dat") -> HGPCodePatch:
    # 주의: LinearCode(M) 은 M 을 패리티 체크 행렬로 받는다.
    # 생성행렬을 주려면 LinearCode.from_generator(G) 를 써야 한다.
    Cb = LinearCode.from_generator([[1, 0, 0, 1, 1, 0],
                                    [0, 1, 0, 0, 1, 1],
                                    [0, 0, 1, 1, 0, 1]])          # [6,3,3]
    Ca = LinearCode(np.asarray(Cb.H, dtype=int).T)
    return HGPCodePatch(HGPCode(Ca, Cb), patch_name=patch_name)  # HGP(Ca,Cb) = [[45,9,3]]


def build_surgery(C: HGPCodePatch) -> HighRateSurgeryPatch:
    """C 옆에 앤실라를 붙여 열 하나에서 logical 하나를 측정."""
    css = C.css
    # 데이터 패치의 실제 폭 (X-check 칸 수 + block2 칸 수) 만큼 옆으로 띄워서 안 겹치게.
    width = css.nb + css.rb
    gap = 2.0

    CaA, CbA = rep(2), rep(3)
    A = HGPCodePatch(HGPCode(CaA, CbA), patch_name="anc", offset=(width + gap, 0.0))

    ltol = np.zeros((css.Cb.k, CbA.k), dtype=int)
    ltol[0, 0] = 1
    h = Homomorphism(CbA, css.Cb, ltol)

    S = HighRateSurgeryPatch(C, A, [(0, 0, h)])
    S.init_logical_representatives()
    return S


def main():
    C = build_hgp()
    path1 = render(C, os.path.join(OUT_DIR, "out_hgp.html"),
                    title=f"HGP code  [[{C.css.n},{C.css.k}]]")
    print("wrote", path1)

    # ConePatch.__init__ 이 C 의 check 를 참조하므로 surgery 용은 새로 만든다.
    C2 = build_hgp()
    S = build_surgery(C2)
    path2 = render(S, os.path.join(OUT_DIR, "out_surgery.html"),
                    title=f"high-rate surgery  [[{S.css.n},{S.css.k}]]")
    print("wrote", path2)


if __name__ == "__main__":
    main()
