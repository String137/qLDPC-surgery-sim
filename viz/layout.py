"""
CodePatch의 qubit, check 배치를 matplotlib으로 그림.

    visualize(patch)                      # check까지
    visualize(patch, show_checks=False)   # qubit만

qubit 좌표는 global coord로 씀.
check 좌표는 local_coord_chk나 check2coord로 찾고, 없으면 qubit 좌표 평균.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def _own_check_coord(patch, typ, i):
    # patch 하나 안에서의 check 좌표. local_coord_chk -> check2coord 순서. 둘 다 없으면 None
    try:
        return patch.local_coord_chk(typ, i)
    except (AttributeError, NotImplementedError):
        pass
    if getattr(patch, "check2coord", None):
        chk = (patch.x_checks() if typ == "X" else patch.z_checks())[i]
        return patch.check2coord.get(chk)
    return None


def _check_coord(patch, typ, i, chk, coord_of):
    coord = _own_check_coord(patch, typ, i)
    if coord is None and hasattr(patch, "Cs"):
        # ConePatch 계열: check 순서는 C_1, ..., C_m, A. 원래 패치 좌표 + offset
        local = i
        for sub in list(patch.Cs) + [patch.A]:
            n = len(sub.x_checks() if typ == "X" else sub.z_checks())
            if local < n:
                c = _own_check_coord(sub, typ, local)
                if c is not None:
                    coord = (c[0] + sub._offset[0], c[1] + sub._offset[1])
                break
            local -= n
    if coord is None:
        pts = [coord_of[q] for q in chk.qubits]
        coord = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return coord


def visualize(patch, show_checks: bool = True, show_index: bool = True, ax=None, title: str | None = None):
    qubits = patch.data_qubits()
    qubit_coords = [patch.local_coord(q) for q in qubits]
    coord_of = dict(zip(qubits, qubit_coords))
    x_coords = [c[0] for c in qubit_coords]
    y_coords = [c[1] for c in qubit_coords]

    if ax is None:
        w = max(x_coords) - min(x_coords) + 1
        h = max(y_coords) - min(y_coords) + 1
        _, ax = plt.subplots(figsize=(w, h))
    ax.set_aspect('equal')

    # check
    if show_checks:
        for typ, checks, color in [("X", patch.x_checks(), 'red'), ("Z", patch.z_checks(), 'blue')]:
            for i, chk in enumerate(checks):
                if len(chk.qubits) == 0:
                    continue
                cx, cy = _check_coord(patch, typ, i, chk, coord_of)
                for q in chk.qubits:
                    x, y = coord_of[q]
                    ax.plot([cx, x], [cy, y], color=color, lw=0.8, alpha=0.4, zorder=0)
                ax.scatter([cx], [cy], marker='s', s=25, color=color, zorder=1)

    # 패치마다 색 다르게
    patch_names = list(dict.fromkeys(q.patch_name for q in qubits))
    for name in patch_names:
        xs = [x for q, x in zip(qubits, x_coords) if q.patch_name == name]
        ys = [y for q, y in zip(qubits, y_coords) if q.patch_name == name]
        ax.scatter(xs, ys, s=200, alpha=0.5, label=name, zorder=2)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))

    # qubit 번호
    if show_index:
        for i, (x, y) in enumerate(qubit_coords):
            ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center', zorder=3)

    ax.set_title(title if title is not None else f"Qubit layout of {patch.name()}")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid()
    return ax
