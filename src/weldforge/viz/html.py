"""Writing a scene into one self-contained HTML file.

Constraints that shaped this
----------------------------
* **No CDN.** A welding cell is often not on a network, and a viewer that
  needs one is a viewer that does not open.
* **No install.** The point is that an engineer with neither ROS nor FreeCAD
  can look at the plan.
* **One file.** It gets emailed, attached to a job folder, and opened two
  years later.

So the renderer is about two hundred lines of plain JavaScript on a 2-D
canvas: orbit, zoom, painter's-algorithm triangles with flat shading, and
depth-sorted polylines. Not a game engine -- but enough to see whether the
torch points at the joint, and that is what it is for.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .scene import Scene

logger = logging.getLogger(__name__)

__all__ = ["scene_to_json", "write_html"]


def _round(array: np.ndarray, digits: int = 2) -> list:
    return np.asarray(array, dtype=float).round(digits).tolist()


def scene_to_json(scene: Scene) -> dict:
    """The scene as a JSON-ready dict, with geometry flattened for the browser."""
    lo, hi = scene.bounds
    return {
        "name": scene.name,
        "bounds": {"min": _round(lo), "max": _round(hi)},
        "meshes": [
            {
                "name": mesh.name,
                "group": mesh.group,
                "colour": mesh.colour,
                "opacity": mesh.opacity,
                "vertices": _round(mesh.vertices.reshape(-1), 1),
                "faces": np.asarray(mesh.faces, dtype=int).reshape(-1).tolist(),
            }
            for mesh in scene.meshes
            if len(mesh.vertices) and len(mesh.faces)
        ],
        "lines": [
            {
                "name": line.name,
                "label": line.label,
                "group": line.group,
                "colour": line.colour,
                "width": line.width,
                "segments": line.segments,
                "points": _round(line.points.reshape(-1), 1),
            }
            for line in scene.lines
            if len(line.points)
        ],
        "frames": [
            {
                "arm": _round(frame.arm.reshape(-1), 1),
                "torch": _round(frame.torch.reshape(-1), 1),
                "weld": bool(frame.weld_on),
                "seam": frame.seam_id,
                "pass": frame.pass_index,
                "t": round(frame.time_s, 2),
            }
            for frame in scene.frames
        ],
        "summary": scene.summary,
    }


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #1b2027; --panel: #232a33; --line: #323c49;
    --text: #e6eaf0; --muted: #9aa6b5; --accent: #38bdf8;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 13px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
  #wrap { display: flex; height: 100vh; }
  #view { flex: 1; position: relative; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  #side { width: 300px; background: var(--panel); border-left: 1px solid var(--line);
          overflow-y: auto; padding: 14px 16px; }
  h1 { font-size: 15px; margin: 0 0 4px; }
  h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
       color: var(--muted); margin: 18px 0 6px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 11px; margin-bottom: 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  td { padding: 2px 0; vertical-align: top; }
  td.k { color: var(--muted); padding-right: 8px; white-space: nowrap; }
  td.v { text-align: right; font-variant-numeric: tabular-nums; }
  label { display: flex; align-items: center; gap: 7px; padding: 2px 0;
          cursor: pointer; }
  .swatch { width: 11px; height: 11px; border-radius: 2px; flex: none; }
  #hud { position: absolute; left: 12px; top: 12px; background: rgba(27,32,39,.88);
         border: 1px solid var(--line); border-radius: 6px; padding: 8px 11px;
         font-size: 12px; font-variant-numeric: tabular-nums; pointer-events: none; }
  #hud .arc { color: #ff6b35; font-weight: 600; }
  #hud .off { color: var(--muted); }
  #playbar { position: absolute; left: 12px; right: 12px; bottom: 12px;
             background: rgba(27,32,39,.9); border: 1px solid var(--line);
             border-radius: 6px; padding: 8px 11px; display: flex;
             align-items: center; gap: 11px; }
  #playbar input[type=range] { flex: 1; }
  button { background: var(--line); color: var(--text); border: 0;
           border-radius: 4px; padding: 5px 11px; cursor: pointer; font-size: 12px; }
  button:hover { background: #3d4a5a; }
  .warn { color: #ebcb8b; font-size: 11px; margin-top: 3px; }
  .err { color: #bf616a; font-size: 11px; margin-top: 3px; }
  .ok { color: #a3be8c; }
  .hint { color: var(--muted); font-size: 11px; margin-top: 14px;
          border-top: 1px solid var(--line); padding-top: 10px; }
</style>
</head>
<body>
<div id="wrap">
  <div id="view">
    <canvas id="c"></canvas>
    <div id="hud"></div>
    <div id="playbar">
      <button id="play">Play</button>
      <input id="scrub" type="range" min="0" max="0" value="0" step="1">
      <span id="clock">0.0 s</span>
    </div>
  </div>
  <div id="side">
    <h1 id="title"></h1>
    <div class="sub" id="verdict"></div>
    <h2>Numbers</h2>
    <table id="numbers"></table>
    <h2>Layers</h2>
    <div id="layers"></div>
    <h2>Seams</h2>
    <table id="seams"></table>
    <div id="messages"></div>
    <div class="hint">
      Drag to orbit, scroll to zoom, right-drag to pan.<br>
      Keys: <b>space</b> play, <b>r</b> reset view, <b>1-5</b> toggle layers.
    </div>
  </div>
</div>
<script id="scene" type="application/json">__SCENE__</script>
<script>
"use strict";
const SCENE = JSON.parse(document.getElementById("scene").textContent);

/* ---------- geometry helpers: a scene this size needs no library ---------- */
const sub = (a,b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const norm = v => { const n = Math.hypot(v[0],v[1],v[2]) || 1; return [v[0]/n,v[1]/n,v[2]/n]; };

const bmin = SCENE.bounds.min, bmax = SCENE.bounds.max;
const centre = [(bmin[0]+bmax[0])/2, (bmin[1]+bmax[1])/2, (bmin[2]+bmax[2])/2];
const extent = Math.max(bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]) || 1000;

const HOME = { yaw: -0.9, pitch: 0.42, dist: extent * 2.1, pan: [0,0] };
let cam = Object.assign({}, HOME);

/* Z is up in the cell, which is the convention the whole toolchain uses. */
function project(p, w, h) {
  const d = sub(p, centre);
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const x1 =  d[0]*cy + d[1]*sy;
  const y1 = -d[0]*sy + d[1]*cy;
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const y2 = y1*cp - d[2]*sp;
  const z2 = y1*sp + d[2]*cp;
  const scale = Math.min(w,h) / cam.dist * 1.6;
  return [w/2 + (x1*scale) + cam.pan[0], h/2 - (y2*scale) + cam.pan[1], z2];
}

/* ---------- layers ---------- */
const GROUPS = [
  ["part",   "Part",        "#8a94a6"],
  ["cell",   "Cell & jig",  "#4c566a"],
  ["seams",  "Weld seams",  "#33b4ff"],
  ["wire",   "Wire direction", "#ffd966"],
  ["path",   "Torch path",  "#ff6b35"],
  ["robot",  "Robot",       "#5e81ac"],
];
const visible = {};
GROUPS.forEach(g => visible[g[0]] = true);

/* ---------- drawing ---------- */
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
let frameIndex = 0, playing = false;

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}

function draw() {
  const w = canvas.width / (window.devicePixelRatio||1);
  const h = canvas.height / (window.devicePixelRatio||1);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "#1b2027"; ctx.fillRect(0,0,w,h);

  const items = [];   /* everything, depth-sorted together: painter's algorithm */

  /* ground grid, for a sense of scale */
  const z0 = bmin[2];
  const stepGrid = 200;
  ctx.strokeStyle = "#2a323d"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = Math.floor(bmin[0]/stepGrid)*stepGrid; x <= bmax[0]+stepGrid; x += stepGrid) {
    const a = project([x, bmin[1]-200, z0], w, h), b = project([x, bmax[1]+200, z0], w, h);
    ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
  }
  for (let y = Math.floor(bmin[1]/stepGrid)*stepGrid; y <= bmax[1]+stepGrid; y += stepGrid) {
    const a = project([bmin[0]-200, y, z0], w, h), b = project([bmax[0]+200, y, z0], w, h);
    ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
  }
  ctx.stroke();

  const light = norm([0.4, -0.6, 0.8]);

  for (const mesh of SCENE.meshes) {
    if (!visible[mesh.group]) continue;
    const v = mesh.vertices, f = mesh.faces;
    for (let i = 0; i < f.length; i += 3) {
      const p = [];
      for (let k = 0; k < 3; k++) {
        const j = f[i+k]*3;
        p.push([v[j], v[j+1], v[j+2]]);
      }
      const n = norm(cross(sub(p[1],p[0]), sub(p[2],p[0])));
      const shade = 0.42 + 0.58 * Math.abs(dot(n, light));
      const s = p.map(q => project(q, w, h));
      items.push({ z: (s[0][2]+s[1][2]+s[2][2])/3, kind: "tri", s,
                   colour: mesh.colour, shade, opacity: mesh.opacity });
    }
  }

  for (const line of SCENE.lines) {
    if (!visible[line.group]) continue;
    const pts = line.points;
    const step = line.segments ? 6 : 3;
    for (let i = 0; i + 5 < pts.length; i += step) {
      const a = project([pts[i], pts[i+1], pts[i+2]], w, h);
      const b = project([pts[i+3], pts[i+4], pts[i+5]], w, h);
      items.push({ z: (a[2]+b[2])/2 - 1, kind: "seg", a, b,
                   colour: line.colour, width: line.width });
    }
  }

  const frame = SCENE.frames[frameIndex];
  if (frame && visible.robot) {
    const arm = frame.arm;
    for (let i = 0; i + 5 < arm.length; i += 3) {
      const a = project([arm[i],arm[i+1],arm[i+2]], w, h);
      const b = project([arm[i+3],arm[i+4],arm[i+5]], w, h);
      items.push({ z: (a[2]+b[2])/2 - 2, kind: "seg", a, b,
                   colour: "#88c0d0", width: 7 });
    }
    const t = frame.torch;
    const a = project([t[0],t[1],t[2]], w, h), b = project([t[3],t[4],t[5]], w, h);
    items.push({ z: (a[2]+b[2])/2 - 3, kind: "seg", a, b,
                 colour: frame.weld ? "#ff6b35" : "#d8dee9", width: 4 });
    if (frame.weld) items.push({ z: b[2]-4, kind: "arc", at: b });
  }

  items.sort((p,q) => p.z - q.z);
  for (const it of items) {
    if (it.kind === "tri") {
      ctx.globalAlpha = it.opacity;
      ctx.fillStyle = shadeColour(it.colour, it.shade);
      ctx.beginPath();
      ctx.moveTo(it.s[0][0], it.s[0][1]);
      ctx.lineTo(it.s[1][0], it.s[1][1]);
      ctx.lineTo(it.s[2][0], it.s[2][1]);
      ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
    } else if (it.kind === "seg") {
      ctx.strokeStyle = it.colour; ctx.lineWidth = it.width;
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(it.a[0],it.a[1]); ctx.lineTo(it.b[0],it.b[1]); ctx.stroke();
    } else {
      const g = ctx.createRadialGradient(it.at[0], it.at[1], 0, it.at[0], it.at[1], 13);
      g.addColorStop(0, "rgba(255,240,180,.95)");
      g.addColorStop(1, "rgba(255,107,53,0)");
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(it.at[0], it.at[1], 13, 0, 6.2832); ctx.fill();
    }
  }
  updateHud();
}

function shadeColour(hex, k) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, ((n>>16)&255)*k) | 0;
  const g = Math.min(255, ((n>>8)&255)*k) | 0;
  const b = Math.min(255, (n&255)*k) | 0;
  return "rgb("+r+","+g+","+b+")";
}

function updateHud() {
  const hud = document.getElementById("hud");
  const f = SCENE.frames[frameIndex];
  if (!f) { hud.textContent = "no animation: the plan has no joint solutions"; return; }
  hud.innerHTML =
    "<div>seam <b>" + (f.seam || "-") + "</b> &nbsp; pass " + (f.pass+1) + "</div>" +
    "<div class=" + (f.weld ? "'arc'" : "'off'") + ">" +
    (f.weld ? "ARC ON" : "arc off") + "</div>";
  document.getElementById("clock").textContent = f.t.toFixed(1) + " s";
  document.getElementById("scrub").value = frameIndex;
}

/* ---------- interaction ---------- */
let drag = null;
canvas.addEventListener("mousedown", e => {
  drag = { x: e.clientX, y: e.clientY, button: e.button };
  canvas.classList.add("dragging"); e.preventDefault();
});
window.addEventListener("mouseup", () => { drag = null; canvas.classList.remove("dragging"); });
window.addEventListener("mousemove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.button === 2) { cam.pan[0] += dx; cam.pan[1] += dy; }
  else {
    cam.yaw += dx * 0.008;
    cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.008));
  }
  draw();
});
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("wheel", e => {
  cam.dist *= Math.exp(e.deltaY * 0.0012);
  cam.dist = Math.max(extent*0.25, Math.min(extent*12, cam.dist));
  e.preventDefault(); draw();
}, { passive: false });

document.getElementById("scrub").addEventListener("input", e => {
  frameIndex = +e.target.value; draw();
});
document.getElementById("play").addEventListener("click", togglePlay);
function togglePlay() {
  playing = !playing;
  document.getElementById("play").textContent = playing ? "Pause" : "Play";
  if (playing) tick();
}
function tick() {
  if (!playing) return;
  frameIndex = (frameIndex + 1) % Math.max(1, SCENE.frames.length);
  draw();
  setTimeout(tick, 45);
}
window.addEventListener("keydown", e => {
  if (e.key === " ") { togglePlay(); e.preventDefault(); }
  if (e.key === "r") { cam = Object.assign({}, HOME); cam.pan = [0,0]; draw(); }
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= GROUPS.length) {
    const key = GROUPS[n-1][0];
    visible[key] = !visible[key];
    document.querySelector("input[data-group='"+key+"']").checked = visible[key];
    draw();
  }
});
window.addEventListener("resize", resize);

/* ---------- side panel ---------- */
function buildPanel() {
  const s = SCENE.summary || {};
  document.getElementById("title").textContent = SCENE.name;
  const r = s.report || {};
  const verdict = document.getElementById("verdict");
  verdict.innerHTML = r.ok
    ? "<span class='ok'>plan OK</span>"
    : "<span class='err'>plan reported problems</span>";

  const rows = [
    ["seams", s.n_selected ?? s.n_seams],
    ["weld length", (s.total_weld_length_mm ?? 0).toFixed(0) + " mm"],
    ["passes", SCENE.frames.length ? (s.n_paths ?? "") : ""],
    ["cycle time", (r.cycle_time_s ?? 0).toFixed(0) + " s"],
    ["arc time", (r.arc_time_s ?? 0).toFixed(0) + " s"],
    ["duty cycle", ((r.duty_cycle ?? 0)*100).toFixed(0) + " %"],
    ["wire", ((r.wire_consumption_kg ?? 0)*1000).toFixed(0) + " g"],
    ["gas", (r.gas_consumption_l ?? 0).toFixed(0) + " l"],
  ];
  document.getElementById("numbers").innerHTML = rows
    .filter(row => row[1] !== "" && row[1] !== undefined)
    .map(row => "<tr><td class='k'>"+row[0]+"</td><td class='v'>"+row[1]+"</td></tr>")
    .join("");

  document.getElementById("layers").innerHTML = GROUPS.map((g,i) =>
    "<label><input type='checkbox' data-group='"+g[0]+"' checked>" +
    "<span class='swatch' style='background:"+g[2]+"'></span>" +
    g[1] + " <span style='color:var(--muted)'>("+(i+1)+")</span></label>").join("");
  document.querySelectorAll("#layers input").forEach(box => {
    box.addEventListener("change", () => {
      visible[box.dataset.group] = box.checked; draw();
    });
  });

  const seams = SCENE.lines.filter(l => l.group === "seams");
  document.getElementById("seams").innerHTML = seams.map(l =>
    "<tr><td class='k'><span class='swatch' style='display:inline-block;background:"
    + l.colour + "'></span> " + (l.label || l.name) + "</td></tr>").join("");

  const messages = document.getElementById("messages");
  let html = "";
  (r.errors || []).slice(0,8).forEach(e => html += "<div class='err'>! "+e+"</div>");
  (r.warnings || []).slice(0,8).forEach(w => html += "<div class='warn'>- "+w+"</div>");
  if (html) messages.innerHTML = "<h2>Messages</h2>" + html;
}

document.getElementById("scrub").max = Math.max(0, SCENE.frames.length - 1);
buildPanel();
resize();
</script>
</body>
</html>
"""


def write_html(scene: Scene, path: str | Path, title: str | None = None) -> Path:
    """Write the scene as one self-contained HTML file."""
    payload = json.dumps(scene_to_json(scene), separators=(",", ":"))
    html = _TEMPLATE.replace("__SCENE__", payload).replace(
        "__TITLE__", title or scene.name or "WeldForge"
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("wrote %s (%.1f kB)", out, len(html) / 1024.0)
    return out
