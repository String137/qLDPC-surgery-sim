"""
hgp_viz.py — HGPCodePatch / HighRateSurgeryPatch(_ConePatch 계열) 를
클릭 가능한 단일 HTML 파일로 렌더링한다.

외부 의존성 없음. 순수 SVG + vanilla JS, CDN 없음.

사용법:
    from viz.hgp_viz import render
    render(patch, "out.html", title="my code")

patch 에서 뽑아내는 것:
    - 큐빗 좌표         : patch.data_qubits(), patch.local_coord(q)
    - 체크 + 좌표        : patch.x_checks() / z_checks(), patch.local_coord_chk(typ, i)
    - logical operator  : patch.css.logical_x_repr / logical_z_repr
                           (patch.css 가 None 이거나 비어있으면 그 UI 는 숨김.
                            render() 안에서 make_css() 를 호출하지 않는다.)
"""

from __future__ import annotations

import json
import numpy as np


# ---------------------------------------------------------------------------
# patch -> JSON-able dict
# ---------------------------------------------------------------------------

def _patch_roles(patch) -> dict:
    """patch_name -> 'data' | 'ancilla'. _ConePatch/ConePatch 계열이 아니면 전부 'data'."""
    roles = {}
    for attr in ("C", "C1", "C2"):
        sub = getattr(patch, attr, None)
        if sub is not None:
            roles[sub._name] = "data"
    a = getattr(patch, "A", None)
    if a is not None:
        roles[a._name] = "ancilla"
    return roles


def _sub_patches(patch) -> list:
    """local_coord_chk 의 사분면 grid 를 그리기 위한 하위 HGPCodePatch 들.
    _ConePatch/ConePatch 가 아니면 patch 자기 자신 하나."""
    subs = [getattr(patch, attr, None) for attr in ("C", "C1", "C2", "A")]
    subs = [s for s in subs if s is not None]
    return subs if subs else [patch]


def _quadrant_info(sub) -> dict | None:
    """HGP 사분면 grid 선/라벨을 그리는 데 필요한 정보. HGPCodePatch 가 아니면 None."""
    css = getattr(sub, "css", None)
    if css is None or not all(hasattr(css, a) for a in ("na", "nb", "ra", "rb")):
        return None
    ox, oy = sub._offset
    sx, sy = getattr(sub, "scale", (1.0, 1.0))
    return {
        "patch": sub._name,
        "ox": float(ox), "oy": float(oy),
        "sx": float(sx), "sy": float(sy),
        "na": int(css.na), "nb": int(css.nb),
        "ra": int(css.ra), "rb": int(css.rb),
    }


def _extract(patch) -> dict:
    qubits = patch.data_qubits()
    idx_of = {q: i for i, q in enumerate(qubits)}
    roles = _patch_roles(patch)

    qubit_data = []
    for i, q in enumerate(qubits):
        x, y = patch.local_coord(q)
        qubit_data.append({
            "i": i,
            "patch": q.patch_name,
            "local_idx": q.idx,
            "x": float(x), "y": float(y),
            "role": roles.get(q.patch_name, "data"),
        })

    def _checks(cks, typ):
        out = []
        for pos, ck in enumerate(cks):
            x, y = patch.local_coord_chk(typ, pos)
            support = sorted(idx_of[q] for q in ck.qubits)
            patch_names = sorted({q.patch_name for q in ck.qubits})
            out.append({
                "type": typ,
                "label": ck.label or f"{typ}{pos}",
                "x": float(x), "y": float(y),
                "support": support,
                "patches": patch_names,
                "weight": len(support),
                "multi": len(patch_names) > 1,
            })
        return out

    checks = _checks(patch.x_checks(), "X") + _checks(patch.z_checks(), "Z")

    logicals = []
    css = getattr(patch, "css", None)
    lx = getattr(css, "logical_x_repr", None) if css is not None else None
    lz = getattr(css, "logical_z_repr", None) if css is not None else None
    if (lx is not None and lz is not None
            and all(v is not None for v in lx) and all(v is not None for v in lz)):
        for i in range(css.k):
            for typ, reps in (("X", lx), ("Z", lz)):
                vec = np.asarray(reps[i], dtype=int)
                support = sorted(int(j) for j in np.nonzero(vec)[0])
                logicals.append({"type": typ, "index": i, "support": support})

    quadrants = [qi for qi in (_quadrant_info(s) for s in _sub_patches(patch)) if qi is not None]
    patch_names = sorted({q["patch"] for q in qubit_data})

    return {
        "n": len(qubit_data),
        "qubits": qubit_data,
        "checks": checks,
        "logicals": logicals,
        "hasLogicals": bool(logicals),
        "patches": patch_names,
        "roles": {p: roles.get(p, "data") for p in patch_names},
        "quadrants": quadrants,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def render(patch, path: str = "viz.html", title: str | None = None) -> str:
    """patch 를 자체 완결 HTML 파일로 렌더링한다.

    patch 는 HGPCodePatch 또는 HighRateSurgeryPatch(_ConePatch 계열).
    반환값은 실제로 쓰여진 경로.
    """
    data = _extract(patch)
    title = title or f"HGP viz  [[{data['n']}]]"
    json_str = json.dumps(data).replace("</script", "<\\/script")

    html = _HTML_TEMPLATE.replace("__TITLE__", _escape_html(title))
    html = html.replace("__DATA_JSON__", json_str)

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _escape_html(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0b0e14; --panel: #151a23; --border: #2a3242; --text: #dbe2f0;
    --muted: #8b96ab; --accent: #4fb3ff; --accent2: #ff9a4f; --accent3: #b57bff;
    --data: #4fb3ff; --anc: #ff9a4f; --xchk: #63d0a4; --zchk: #f2b84b;
    --sel: #ffe14f;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
               font: 13px/1.4 -apple-system, Segoe UI, Helvetica, Arial, sans-serif; }
  #app { display: flex; flex-direction: column; height: 100%; }
  header { padding: 8px 12px; border-bottom: 1px solid var(--border); font-weight: 600; }
  #main { flex: 1; display: flex; min-height: 0; }
  #svgWrap { flex: 1; position: relative; overflow: hidden; background:
             repeating-linear-gradient(0deg, #0e131c 0 1px, transparent 1px 100%),
             repeating-linear-gradient(90deg, #0e131c 0 1px, transparent 1px 100%);
             background-size: 40px 40px; }
  svg { width: 100%; height: 100%; display: block; cursor: grab; }
  svg.panning { cursor: grabbing; }
  #side { width: 260px; overflow-y: auto; border-left: 1px solid var(--border);
          background: var(--panel); padding: 10px; }
  #side h3 { margin: 14px 0 6px; font-size: 12px; text-transform: uppercase;
             letter-spacing: .04em; color: var(--muted); }
  #side h3:first-child { margin-top: 0; }
  #side label { display: flex; align-items: center; gap: 6px; padding: 2px 0; cursor: pointer; }
  #side select, #side button { width: 100%; background: #1c2330; color: var(--text);
             border: 1px solid var(--border); border-radius: 4px; padding: 5px 6px; margin: 3px 0;
             font: inherit; cursor: pointer; }
  #side button:hover, #side select:hover { border-color: var(--accent); }
  .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; flex: none; }
  #stat { font-size: 12px; color: var(--muted); white-space: pre-line; }
  .qubit { fill: var(--data); stroke: #00000055; stroke-width: 0.03; }
  .qubit.role-ancilla { fill: var(--anc); }
  .qubit.hi-sel { fill: var(--sel); stroke: #000; }
  .qubit.hi-picked { stroke: #fff; stroke-width: 0.09; }
  .chk { stroke: #000000aa; stroke-width: 0.03; cursor: pointer; }
  .chk.X { fill: var(--xchk); }
  .chk.Z { fill: var(--zchk); }
  .chk.multi.X { fill: #2f8f6d; }
  .chk.multi.Z { fill: #b9852a; }
  .chk.sel-explore { stroke: #fff; stroke-width: 0.12; }
  .chk.hidden, .qubit.hidden { display: none; }
  .edge { stroke: var(--sel); stroke-width: 0.05; opacity: .85; }
  .logChip { display: inline-block; padding: 3px 9px; border-radius: 12px; margin: 2px;
             border: 1px solid var(--border); background: #1c2330; color: var(--text);
             font-size: 12px; cursor: pointer; }
  .logChip:hover { border-color: var(--accent); }
  .logChip.X { border-color: var(--xchk); }
  .logChip.Z { border-color: var(--zchk); }
  .logChip.active { background: var(--sel); color: #000; border-color: var(--sel); }
  .qgrid { stroke: #ffffff22; stroke-width: 0.02; fill: none; }
  .qlabel { fill: #ffffff55; font-size: 0.55px; }
  .plabel { fill: #ffffff88; font-size: 0.7px; font-weight: 600; }
  #toolbar { display: flex; gap: 6px; padding: 6px; border-bottom: 1px solid var(--border); }
  #toolbar button { background: #1c2330; color: var(--text); border: 1px solid var(--border);
             border-radius: 4px; padding: 5px 10px; cursor: pointer; font: inherit; }
  #toolbar button:hover { border-color: var(--accent); }
  #tip { position: absolute; pointer-events: none; background: #000c; color: #fff; padding: 4px 7px;
         border-radius: 4px; font-size: 12px; display: none; white-space: nowrap; z-index: 5; }
</style>
</head>
<body>
<div id="app">
  <header>__TITLE__</header>
  <div id="toolbar">
    <button id="btnReset">리셋 뷰</button>
    <button id="btnClearExplore">선택 해제</button>
    <span id="exploreWeight" style="align-self:center; color:var(--sel); font-weight:600;"></span>
    <span id="stat" style="align-self:center; margin-left:auto;"></span>
  </div>
  <div id="main">
    <div id="svgWrap">
      <svg id="svg" viewBox="0 0 100 100">
        <g id="viewport">
          <g id="gQuadrants"></g>
          <g id="gEdges"></g>
          <g id="gChecks"></g>
          <g id="gQubits"></g>
        </g>
      </svg>
      <div id="tip"></div>
    </div>
    <div id="side">
      <h3>필터: check 타입</h3>
      <label><input type="radio" name="typeFilter" value="all" checked> 전부</label>
      <label><input type="radio" name="typeFilter" value="X"> X-check 만</label>
      <label><input type="radio" name="typeFilter" value="Z"> Z-check 만</label>

      <h3>필터: 패치</h3>
      <div id="patchFilters"></div>

      <h3>필터: surgery 연결</h3>
      <label><input type="checkbox" id="surgeryOnly">
        연결 check 만 (2개 이상 패치에 걸침)</label>
      <div id="legendMulti" style="margin-top:4px; font-size:12px; color:var(--muted);">
        <div><span class="swatch" style="background:#2f8f6d"></span> &Gamma;1 (X, 앤실라&harr;데이터)</div>
        <div><span class="swatch" style="background:#b9852a"></span> &Gamma;0 (Z, 앤실라&harr;데이터)</div>
      </div>

      <div id="logicalPanel" style="display:none">
        <h3>logical operator</h3>
        <div id="logicalList"></div>
        <div style="font-size:11px; color:var(--muted); margin-top:6px;">
          check 처럼 클릭해서 껐다 켭니다. 켜진 것들끼리 XOR 로 곱해집니다
          (X 는 X-check/X-logical 끼리, Z 는 Z-check/Z-logical 끼리).
        </div>
      </div>
    </div>
  </div>
</div>
<script id="dataBlock" type="application/json">__DATA_JSON__</script>
<script>
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("dataBlock").textContent);
  const SVGNS = "http://www.w3.org/2000/svg";

  const svg = document.getElementById("svg");
  const viewport = document.getElementById("viewport");
  const gQuadrants = document.getElementById("gQuadrants");
  const gEdges = document.getElementById("gEdges");
  const gChecks = document.getElementById("gChecks");
  const gQubits = document.getElementById("gQubits");
  const tip = document.getElementById("tip");
  const svgWrap = document.getElementById("svgWrap");

  // -- bounding box --------------------------------------------------------
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  function extend(x, y) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  DATA.qubits.forEach(q => extend(q.x, q.y));
  DATA.checks.forEach(c => extend(c.x, c.y));
  DATA.quadrants.forEach(qd => {
    extend(qd.ox, qd.oy);
    extend(qd.ox + (qd.nb + qd.rb) * qd.sx, qd.oy + (qd.na + qd.ra) * qd.sy);
  });
  if (!isFinite(minX)) { minX = 0; minY = 0; maxX = 10; maxY = 10; }
  const pad = Math.max(1, 0.08 * Math.max(maxX - minX, maxY - minY));
  const bbox = { x: minX - pad, y: minY - pad, w: (maxX - minX) + 2 * pad, h: (maxY - minY) + 2 * pad };

  function fitView() {
    svg.setAttribute("viewBox", `${bbox.x} ${bbox.y} ${bbox.w} ${bbox.h}`);
    view = { x: bbox.x, y: bbox.y, w: bbox.w, h: bbox.h };
  }
  let view;

  // -- quadrant grid (HGP 사분면 경계 + 라벨) -------------------------------
  DATA.quadrants.forEach(qd => {
    const x0 = qd.ox, y0 = qd.oy;
    const xm = qd.ox + qd.nb * qd.sx, xr = qd.ox + (qd.nb + qd.rb) * qd.sx;
    const ym = qd.oy + qd.na * qd.sy, yb = qd.oy + (qd.na + qd.ra) * qd.sy;
    const mk = (tag, attrs) => {
      const el = document.createElementNS(SVGNS, tag);
      for (const k in attrs) el.setAttribute(k, attrs[k]);
      return el;
    };
    gQuadrants.appendChild(mk("rect", { x: x0 - 0.5, y: y0 - 0.5, width: (xr - x0) + 1, height: (yb - y0) + 1,
      class: "qgrid" }));
    gQuadrants.appendChild(mk("line", { x1: xm, y1: y0 - 0.5, x2: xm, y2: yb + 0.5, class: "qgrid" }));
    gQuadrants.appendChild(mk("line", { x1: x0 - 0.5, y1: ym, x2: xr + 0.5, y2: ym, class: "qgrid" }));
    const lbl = (x, y, text, anchor) => {
      const t = mk("text", { x, y, class: "plabel", "text-anchor": anchor || "start" });
      t.textContent = text;
      gQuadrants.appendChild(t);
    };
    lbl(x0, y0 - 0.6, `${qd.patch}: X-check`, "start");
    lbl(xr, y0 - 0.6, "block2", "end");
    lbl(x0, yb + 1.0, "block1", "start");
    lbl(xr, yb + 1.0, "Z-check", "end");
  });

  // -- qubits ---------------------------------------------------------------
  const qubitEls = [];
  DATA.qubits.forEach(q => {
    const el = document.createElementNS(SVGNS, "circle");
    el.setAttribute("cx", q.x); el.setAttribute("cy", q.y); el.setAttribute("r", 0.28);
    el.setAttribute("class", "qubit role-" + q.role);
    el.dataset.i = q.i; el.dataset.patch = q.patch;
    const title = document.createElementNS(SVGNS, "title");
    title.textContent = `${q.patch}[${q.local_idx}]  (qubit #${q.i})`;
    el.appendChild(title);
    el.addEventListener("click", (ev) => { ev.stopPropagation(); onQubitClick(q.i); });
    gQubits.appendChild(el);
    qubitEls.push(el);
  });

  // -- checks -----------------------------------------------------------------
  const checkEls = [];
  DATA.checks.forEach((c, idx) => {
    let el;
    if (c.type === "X") {
      el = document.createElementNS(SVGNS, "rect");
      el.setAttribute("x", c.x - 0.24); el.setAttribute("y", c.y - 0.24);
      el.setAttribute("width", 0.48); el.setAttribute("height", 0.48);
    } else {
      el = document.createElementNS(SVGNS, "polygon");
      const r = 0.32;
      const pts = [[0, -r], [r * 0.87, r * 0.5], [-r * 0.87, r * 0.5]]
        .map(p => `${c.x + p[0]},${c.y + p[1]}`).join(" ");
      el.setAttribute("points", pts);
    }
    el.setAttribute("class", "chk " + c.type + (c.multi ? " multi" : ""));
    el.dataset.idx = idx; el.dataset.type = c.type;
    const title = document.createElementNS(SVGNS, "title");
    title.textContent = `${c.label}  (${c.type}, weight=${c.weight}${c.multi ? ", surgery" : ""})`;
    el.appendChild(title);
    el.addEventListener("click", (ev) => { ev.stopPropagation(); onCheckClick(idx); });
    gChecks.appendChild(el);
    checkEls.push(el);
  });

  // -- logical operator 칩 (check 와 똑같이 클릭으로 켰다 껐다) -----------------
  const logicalEls = [];
  if (DATA.hasLogicals) {
    document.getElementById("logicalPanel").style.display = "";
    const list = document.getElementById("logicalList");
    DATA.logicals.forEach((l, li) => {
      const btn = document.createElement("button");
      btn.className = "logChip " + l.type;
      btn.textContent = `${l.type}_${l.index}`;
      btn.title = `weight ${l.support.length}`;
      btn.addEventListener("click", () => onLogicalClick(li));
      list.appendChild(btn);
      logicalEls.push(btn);
    });
  }

  // -- 선택 상태: check / qubit / logical 전부 켰다 껐다 누적 가능 -------------
  const selectedChecks = new Set();
  const selectedQubits = new Set();
  const selectedLogicals = new Set();

  // 켜진 check 들과 켜진 logical 들을 전부 XOR (mod 2 합) 한 것이 "현재 곱" 이다.
  // 겹치는 큐빗은 서로 상쇄되어 꺼져야 하므로 합집합이 아니라 대칭차를 쓴다.
  // X 타입과 Z 타입은 서로 다른 연산자이므로 따로 XOR 한다 (X는 X끼리만, Z는 Z끼리만).
  function redrawSelection() {
    gEdges.innerHTML = "";
    checkEls.forEach((el, idx) => el.classList.toggle("sel-explore", selectedChecks.has(idx)));
    logicalEls.forEach((el, li) => el.classList.toggle("active", selectedLogicals.has(li)));

    const prod = { X: new Set(), Z: new Set() };
    function xorInto(typ, support) {
      support.forEach(qi => { prod[typ].has(qi) ? prod[typ].delete(qi) : prod[typ].add(qi); });
    }
    selectedChecks.forEach(idx => {
      const c = DATA.checks[idx];
      xorInto(c.type, c.support);
      // 선은 원본 연결관계를 보여주기 위해 상쇄 여부와 무관하게 전부 그린다.
      c.support.forEach(qi => {
        const q = DATA.qubits[qi];
        const line = document.createElementNS(SVGNS, "line");
        line.setAttribute("x1", c.x); line.setAttribute("y1", c.y);
        line.setAttribute("x2", q.x); line.setAttribute("y2", q.y);
        line.setAttribute("class", "edge");
        gEdges.appendChild(line);
      });
    });
    selectedLogicals.forEach(li => {
      const l = DATA.logicals[li];
      xorInto(l.type, l.support);   // logical 은 특정 위치가 없으므로 선은 그리지 않는다.
    });

    qubitEls.forEach((el, i) => {
      el.classList.toggle("hi-sel", prod.X.has(i) || prod.Z.has(i));
      el.classList.toggle("hi-picked", selectedQubits.has(i));
    });

    // checks containing a manually-picked qubit -> outline too (선택된 check 는 이미 위에서 처리)
    checkEls.forEach((el, idx) => {
      if (selectedChecks.has(idx)) return;
      const c = DATA.checks[idx];
      const touches = c.support.some(qi => selectedQubits.has(qi));
      el.classList.toggle("sel-explore", touches);
    });

    const active = selectedChecks.size || selectedLogicals.size;
    document.getElementById("exploreWeight").textContent =
      active ? `X weight: ${prod.X.size}   Z weight: ${prod.Z.size}` : "";
  }

  function onCheckClick(idx) {
    if (selectedChecks.has(idx)) selectedChecks.delete(idx); else selectedChecks.add(idx);
    redrawSelection();
  }

  function onQubitClick(i) {
    if (selectedQubits.has(i)) selectedQubits.delete(i); else selectedQubits.add(i);
    redrawSelection();
  }

  function onLogicalClick(li) {
    if (selectedLogicals.has(li)) selectedLogicals.delete(li); else selectedLogicals.add(li);
    redrawSelection();
  }

  document.getElementById("btnClearExplore").addEventListener("click", () => {
    selectedChecks.clear(); selectedQubits.clear(); selectedLogicals.clear(); redrawSelection();
  });

  // -- filters ---------------------------------------------------------------
  const patchFiltersEl = document.getElementById("patchFilters");
  const patchVisible = {};
  DATA.patches.forEach(p => {
    patchVisible[p] = true;
    const role = DATA.roles[p] || "data";
    const id = "pf_" + p;
    const label = document.createElement("label");
    const color = role === "ancilla" ? "var(--anc)" : "var(--data)";
    label.innerHTML = `<input type="checkbox" id="${id}" checked>
      <span class="swatch" style="background:${color}"></span> ${p} (${role})`;
    patchFiltersEl.appendChild(label);
    label.querySelector("input").addEventListener("change", (ev) => {
      patchVisible[p] = ev.target.checked; applyFilters();
    });
  });

  function applyFilters() {
    const typeSel = document.querySelector('input[name="typeFilter"]:checked').value;
    const surgeryOnly = document.getElementById("surgeryOnly").checked;

    qubitEls.forEach((el, i) => {
      const q = DATA.qubits[i];
      el.classList.toggle("hidden", !patchVisible[q.patch]);
    });
    checkEls.forEach((el, idx) => {
      const c = DATA.checks[idx];
      let vis = true;
      if (typeSel !== "all" && c.type !== typeSel) vis = false;
      if (surgeryOnly && !c.multi) vis = false;
      if (!c.patches.every(p => patchVisible[p])) vis = false;
      el.classList.toggle("hidden", !vis);
    });
  }
  document.querySelectorAll('input[name="typeFilter"]').forEach(r => r.addEventListener("change", applyFilters));
  document.getElementById("surgeryOnly").addEventListener("change", applyFilters);
  applyFilters();

  // -- pan / zoom -------------------------------------------------------------
  fitView();
  function applyView() {
    svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`);
  }

  function clientToUser(clientX, clientY) {
    const pt = svg.createSVGPoint();
    pt.x = clientX; pt.y = clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  svg.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const before = clientToUser(ev.clientX, ev.clientY);
    const factor = Math.pow(1.0015, -ev.deltaY);
    view.w /= factor; view.h /= factor;
    applyView();
    const after = clientToUser(ev.clientX, ev.clientY);
    view.x += before.x - after.x; view.y += before.y - after.y;
    applyView();
  }, { passive: false });

  let panning = false, panStart = null, viewStart = null;
  svg.addEventListener("mousedown", (ev) => {
    if (ev.target !== svg && ev.target.id !== "viewport") return;
    panning = true; svg.classList.add("panning");
    panStart = { x: ev.clientX, y: ev.clientY }; viewStart = { ...view };
  });
  window.addEventListener("mousemove", (ev) => {
    if (!panning) return;
    const scale = view.w / svg.clientWidth;
    view.x = viewStart.x - (ev.clientX - panStart.x) * scale;
    view.y = viewStart.y - (ev.clientY - panStart.y) * scale;
    applyView();
  });
  window.addEventListener("mouseup", () => { panning = false; svg.classList.remove("panning"); });

  document.getElementById("btnReset").addEventListener("click", fitView);

  // -- hover tooltip (SVG <title> 만으로도 되지만, 위치를 따라가는 커스텀 tip 도 추가) --
  function bindTip(el, textFn) {
    el.addEventListener("mousemove", (ev) => {
      tip.style.display = "block";
      tip.style.left = (ev.clientX - svgWrap.getBoundingClientRect().left + 12) + "px";
      tip.style.top = (ev.clientY - svgWrap.getBoundingClientRect().top + 12) + "px";
      tip.textContent = textFn();
    });
    el.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  }
  DATA.qubits.forEach((q, i) => bindTip(qubitEls[i], () => `${q.patch}[${q.local_idx}]  (#${q.i})`));
  DATA.checks.forEach((c, i) => bindTip(checkEls[i], () => `${c.label}  (${c.type}, w=${c.weight})`));

  // -- stats ------------------------------------------------------------------
  document.getElementById("stat").textContent =
    `n=${DATA.n}   X-check=${DATA.checks.filter(c => c.type === "X").length}   ` +
    `Z-check=${DATA.checks.filter(c => c.type === "Z").length}`;
})();
</script>
</body>
</html>
"""
