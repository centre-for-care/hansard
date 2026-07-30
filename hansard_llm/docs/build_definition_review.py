"""Build a self-contained definition-disagreement review page (gold-set worklist).

For every pilot speech it shows, side by side, what the models judged under the
two construct definitions (``current`` and ``era_neutral``), sorted so the
speeches where the definitions disagree most come first. Those are exactly the
cases where the construct boundary is doing real work, so they are the efficient
place to spend expert labelling time.

Unlike the cap explorer this page also *collects* judgements: each speech can be
marked H&SC / borderline / not H&SC with a free-text note, held in the browser's
local storage and exported as CSV. Neither definition is styled as the correct
one, because the pilot cannot establish that; the whole point of the exercise is
that a human decides.

Usage:  python -m hansard_llm.docs.build_definition_review
Writes: hansard_llm/docs/definition_review.html
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hansard_llm import config, run, sample

OUT = Path(__file__).resolve().parent / "definition_review.html"

MODEL_LABELS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B",
    "google/gemma-3-27b-it": "Gemma-27B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-70B",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "Qwen3-235B (ref)",
}
MODEL_ORDER = list(MODEL_LABELS)
FORMATS = ("json", "free")
DEFS = ("current", "era_neutral")
DEF_LABELS = {"current": "Current definition", "era_neutral": "Era-neutral definition"}


def _cell(row) -> dict:
    subs = row["subthemes"]
    present = row["mentions_topic"]
    return {
        "present": bool(present) if present is not None else None,
        "subs": list(subs) if isinstance(subs, list) else [],
        "quote": (row["evidence_quote"] or "")[:300],
    }


def build_data() -> list[dict]:
    df = run.load_results()
    # Matched on the shipping default (role=none, uncapped, temp0) so the only
    # thing differing between the two arms is the definition text.
    df = df[(df["condition"] == "temp0") & (df["role"] == "none")
            & (df["task"] == "v1_nocap")
            & (df["definition"].isin(DEFS))
            & (df["output_format"].isin(FORMATS))].copy()

    meta = sample.load_sample().set_index("speech_id")
    out = []
    for sid, g in df.groupby("speech_id"):
        if sid not in meta.index:
            continue
        m = meta.loc[sid]
        cells = {f: {} for f in FORMATS}
        votes = {d: [] for d in DEFS}
        for fmt in FORMATS:
            for mid in MODEL_ORDER:
                entry = {}
                for d in DEFS:
                    sub = g[(g["output_format"] == fmt) & (g["model_id"] == mid)
                            & (g["definition"] == d)]
                    c = _cell(sub.iloc[0]) if len(sub) else None
                    entry[d] = c
                    if c and c["present"] is not None:
                        votes[d].append(1 if c["present"] else 0)
                cells[fmt][mid] = entry

        rates = {d: (float(np.mean(v)) if v else None) for d, v in votes.items()}
        cur, era = rates["current"], rates["era_neutral"]
        gain = None if (cur is None or era is None) else round(era - cur, 3)
        # How split the models are *within* a definition: 0 = unanimous,
        # 1 = an even 50/50 divide. Reported as the worse (more split) of the two,
        # since either kind of split makes the speech worth a human read.
        split = max((0.0 if r is None else 2 * min(r, 1 - r)) for r in rates.values())
        out.append({
            "id": str(sid),
            "year": int(m["year"]),
            "chamber": str(m["chamber"]),
            "type": str(m["speech_type"]),
            "words": int(m["word_count"]),
            "section": str(m["section_title"]),
            "era": str(m["era"]),
            "text": str(m["speech_text"]),
            "cur": None if cur is None else round(cur, 2),
            "era_rate": None if era is None else round(era, 2),
            "gain": gain,
            "split": round(split, 2),
            "cells": cells,
        })
    out.sort(key=lambda s: abs(s["gain"] or 0), reverse=True)
    return out


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Definition review — which speeches count as health &amp; social care?</title>
<style>
:root{
  --bg:#ffffff; --panel:#f6f7f9; --ink:#1c1c1c; --muted:#6b6b6b; --line:#e2e5e9;
  --a:#8a6d3b; --abg:#f6f0e6; --b:#3b6e8a; --bbg:#e8f0f4;
  --yes:#2f7d4f; --maybe:#c9781a; --no:#b1382f;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#15181c; --panel:#1e2228; --ink:#e9ebee; --muted:#9aa3ad; --line:#2c313a;
  --a:#c9a86a; --abg:#2a2620; --b:#7fb6d4; --bbg:#1d272d;
  --yes:#5cb887; --maybe:#e0a24a; --no:#e0796f;
}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 "Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:18px 22px;border-bottom:1px solid var(--line)}
h1{font-size:19px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13.5px;max-width:78ch}
details.defs{margin-top:10px;font-size:13px;max-width:78ch}
details.defs summary{cursor:pointer;color:var(--muted)}
.defbox{padding:8px 11px;border-radius:8px;margin-top:7px;border:1px solid var(--line)}
.defbox.a{background:var(--abg)} .defbox.b{background:var(--bbg)}
.defbox b{display:block;margin-bottom:3px;font-size:12.5px}
.defbox.a b{color:var(--a)} .defbox.b b{color:var(--b)}
.controls{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;padding:12px 22px;
  border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5}
.controls label{font-size:12.5px;color:var(--muted);margin-right:4px}
select,button{font:inherit;font-size:13px;padding:5px 9px;border:1px solid var(--line);
  border-radius:7px;background:var(--bg);color:var(--ink);cursor:pointer}
select#pick{min-width:min(520px,54vw)}
button:hover{border-color:var(--b)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{border:0;border-radius:0;background:var(--bg)}
.seg button.on{background:var(--b);color:#fff}
.prog{margin-left:auto;font-size:12.5px;color:var(--muted)}
main{padding:18px 22px;max-width:1120px}
.meta{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--muted);font-size:13px;margin-bottom:10px}
.meta b{color:var(--ink);font-weight:600}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12.5px;font-weight:600;color:#fff}
.speech{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:13px 15px;
  max-height:230px;overflow:auto;font-size:14px;margin-bottom:16px;white-space:pre-wrap}
.judge{border:1px solid var(--line);border-radius:9px;padding:12px 14px;margin-bottom:18px;background:var(--panel)}
.judge h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}
.jbtns{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:9px}
.jbtns button{border-width:1.5px;font-weight:600}
.jbtns button.on[data-v="yes"]{background:var(--yes);border-color:var(--yes);color:#fff}
.jbtns button.on[data-v="maybe"]{background:var(--maybe);border-color:var(--maybe);color:#fff}
.jbtns button.on[data-v="no"]{background:var(--no);border-color:var(--no);color:#fff}
textarea{width:100%;min-height:52px;font:inherit;font-size:13.5px;padding:8px 10px;resize:vertical;
  border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink)}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;vertical-align:top;padding:9px 11px;border-top:1px solid var(--line)}
th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
td.model{white-space:nowrap;font-weight:600;font-size:13.5px;width:150px}
.col-a{background:var(--abg)} .col-b{background:var(--bbg)}
th.col-a{color:var(--a)} th.col-b{color:var(--b)}
.yesno{display:inline-block;font-size:11.5px;font-weight:700;padding:1px 7px;border-radius:20px;
  margin-right:6px;color:#fff}
.yesno.y{background:var(--yes)} .yesno.n{background:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.chip{font-size:12.5px;padding:2px 8px;border-radius:6px;border:1px solid var(--line);background:var(--bg)}
.none{color:var(--muted);font-style:italic;font-size:12.5px}
.hint{font-size:12px;color:var(--muted);margin:2px 0 14px}
.flip{box-shadow:inset 3px 0 0 var(--maybe)}
</style>
</head>
<body>
<header>
  <h1>Definition review</h1>
  <div class="sub">The pilot cannot tell us which definition of health and social
  care is the right one, only that the choice moves the answer a lot. This page
  puts the two side by side on the speeches where they disagree most, so the call
  can be made by someone who knows the field. Your judgement is the gold standard
  we will score the pipeline against; the model columns are shown for context, not
  as something to agree with.</div>
  <details class="defs"><summary>The two definitions being compared</summary>
    <div class="defbox a"><b>Current</b>__DEF_A__</div>
    <div class="defbox b"><b>Era-neutral</b>__DEF_B__</div>
    <div style="margin-top:7px;color:var(--muted)">Both are dropped into the same
    sentence: "Determine whether the speech substantively discusses health and
    social care, that is, [definition], rather than merely mentioning it in
    passing."</div>
  </details>
</header>
<div class="controls">
  <span><label>Speech</label><select id="pick"></select></span>
  <button id="prev">Prev</button><button id="next">Next</button>
  <span><label>Show</label><select id="filter">
    <option value="disagree">Definitions disagree</option>
    <option value="split">Models split</option>
    <option value="unlabelled">Not yet labelled</option>
    <option value="all">All speeches</option>
  </select></span>
  <span><label>Sort</label><select id="sort">
    <option value="gap">Definition gap</option>
    <option value="split">Model split</option>
    <option value="year">Year</option>
  </select></span>
  <span><label>Format</label><span class="seg" id="fmt">
    <button data-f="json" class="on">json</button><button data-f="free">free</button>
  </span></span>
  <button id="export">Export CSV</button>
  <span class="prog" id="prog"></span>
</div>
<main>
  <div class="meta" id="meta"></div>
  <div class="speech" id="speech"></div>
  <div class="judge">
    <h3>Your judgement: does this speech substantively discuss health &amp; social care?</h3>
    <div class="jbtns" id="jbtns">
      <button data-v="yes">Yes, H&amp;SC</button>
      <button data-v="maybe">Borderline</button>
      <button data-v="no">No, not H&amp;SC</button>
    </div>
    <textarea id="note" placeholder="Optional: why? Which definition gets it right, and what would you change in the wording?"></textarea>
  </div>
  <div class="hint">Rows marked with an amber edge are cells where the two
  definitions reached different verdicts on the same speech, same model, same
  format. Percentages are the share of the 8 reads (4 models x 2 formats) that
  said yes under each definition.</div>
  <table><thead><tr><th>Model</th>
    <th class="col-a">Current definition</th>
    <th class="col-b">Era-neutral definition</th></tr></thead>
  <tbody id="rows"></tbody></table>
</main>
<script id="data" type="application/json">__DATA__</script>
<script id="models" type="application/json">__MODELS__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const MODELS = JSON.parse(document.getElementById("models").textContent);
const KEY = "hansard_def_review_v1";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { labels = {}; }
let order = [], pos = 0, fmt = "json";

const pick = document.getElementById("pick");
const esc = s => { const d=document.createElement("div"); d.textContent=s; return d.innerHTML; };
const pct = v => v==null ? "—" : Math.round(v*100)+"%";
function gapColor(g){ if(g==null) return "#7c848c"; const a=Math.abs(g);
  if(a>=0.5) return "#b1382f"; if(a>=0.25) return "#c9781a"; if(a>0) return "#8a6d3b"; return "#7c848c"; }
const gapLabel = g => g==null ? "n/a" : (g>0?"+":"")+Math.round(g*100)+"pp";
function save(){ try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch(e){} }

function passes(s){
  const f = document.getElementById("filter").value;
  if(f==="disagree") return s.gain!=null && Math.abs(s.gain)>0;
  if(f==="split") return s.split>0;
  if(f==="unlabelled") return !(labels[s.id] && labels[s.id].v);
  return true;
}
function rebuild(keepId){
  const key = document.getElementById("sort").value;
  order = DATA.map((_,i)=>i).filter(i=>passes(DATA[i]));
  if(!order.length) order = DATA.map((_,i)=>i);
  const gap = i => Math.abs(DATA[i].gain==null ? -1 : DATA[i].gain);
  if(key==="gap") order.sort((a,b)=>gap(b)-gap(a));
  else if(key==="split") order.sort((a,b)=>DATA[b].split-DATA[a].split || gap(b)-gap(a));
  else order.sort((a,b)=>DATA[a].year-DATA[b].year || gap(b)-gap(a));
  pick.innerHTML = order.map((idx,i)=>{ const s=DATA[idx];
    const mark = labels[s.id] && labels[s.id].v ? "* " : "";
    return `<option value="${i}">${mark}${s.year} · ${esc(s.section).slice(0,52)} — gap ${gapLabel(s.gain)}</option>`;
  }).join("");
  const at = keepId==null ? 0 : Math.max(0, order.findIndex(i=>DATA[i].id===keepId));
  pos = at<0 ? 0 : at; pick.value = pos; render();
}
function cellHtml(c, other){
  if(!c) return `<span class="none">no data</span>`;
  if(c.present===null) return `<span class="none">unparsed</span>`;
  const badge = c.present ? `<span class="yesno y">YES</span>`
                          : `<span class="yesno n">NO</span>`;
  if(!c.present) return badge;
  const chips = c.subs.length
    ? `<div class="chips">${c.subs.map(s=>`<span class="chip">${esc(s)}</span>`).join("")}</div>`
    : `<span class="none">(none listed)</span>`;
  return badge + chips;
}
function render(){
  const s = DATA[order[pos]];
  document.getElementById("meta").innerHTML =
    `<span><b>${s.year}</b> · ${esc(s.chamber)}</span>`+
    `<span>${esc(s.section)}</span>`+
    `<span>${s.words} words · ${esc(s.type)}</span>`+
    `<span class="pill" style="background:${gapColor(s.gain)}">gap ${gapLabel(s.gain)} `+
    `(current ${pct(s.cur)} → era-neutral ${pct(s.era_rate)})</span>`;
  document.getElementById("speech").textContent = s.text;
  const cells = s.cells[fmt];
  document.getElementById("rows").innerHTML = MODELS.map(m=>{
    const c = cells[m.id] || {};
    const a = c.current, b = c.era_neutral;
    const flip = a && b && a.present!==null && b.present!==null && a.present!==b.present;
    const k = flip ? " flip" : "";
    return `<tr><td class="model">${esc(m.label)}</td>`+
           `<td class="col-a${k}">${cellHtml(a)}</td>`+
           `<td class="col-b${k}">${cellHtml(b)}</td></tr>`;
  }).join("");
  const rec = labels[s.id] || {};
  document.querySelectorAll("#jbtns button").forEach(b=>
    b.classList.toggle("on", b.dataset.v===rec.v));
  document.getElementById("note").value = rec.note || "";
  const done = Object.values(labels).filter(x=>x && x.v).length;
  document.getElementById("prog").textContent =
    `${pos+1} of ${order.length} shown · ${done} of ${DATA.length} labelled`;
  pick.value = pos;
}
document.querySelectorAll("#jbtns button").forEach(b=>b.onclick=()=>{
  const s = DATA[order[pos]];
  const rec = labels[s.id] || {};
  rec.v = rec.v===b.dataset.v ? null : b.dataset.v;
  rec.note = document.getElementById("note").value;
  labels[s.id] = rec; save(); render();
});
document.getElementById("note").oninput = ()=>{
  const s = DATA[order[pos]];
  const rec = labels[s.id] || {};
  rec.note = document.getElementById("note").value;
  labels[s.id] = rec; save();
};
document.getElementById("export").onclick = ()=>{
  const q = v => `"${String(v==null?"":v).replace(/"/g,'""')}"`;
  const rows = [["speech_id","year","section","expert_label","note",
                 "current_rate","era_neutral_rate","gap","model_split"].join(",")];
  DATA.forEach(s=>{ const r = labels[s.id] || {};
    if(!r.v && !r.note) return;
    rows.push([q(s.id),s.year,q(s.section),q(r.v||""),q(r.note||""),
               s.cur,s.era_rate,s.gain,s.split].join(","));
  });
  const blob = new Blob([rows.join("\n")], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "definition_review_labels.csv";
  a.click(); URL.revokeObjectURL(a.href);
};
document.getElementById("filter").onchange = ()=>rebuild();
document.getElementById("sort").onchange = ()=>rebuild(DATA[order[pos]].id);
pick.onchange = ()=>{pos=+pick.value; render();};
document.getElementById("prev").onclick = ()=>{pos=(pos-1+order.length)%order.length; render();};
document.getElementById("next").onclick = ()=>{pos=(pos+1)%order.length; render();};
document.querySelectorAll("#fmt button").forEach(b=>b.onclick=()=>{
  fmt=b.dataset.f;
  document.querySelectorAll("#fmt button").forEach(x=>x.classList.toggle("on",x===b));
  render();
});
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="TEXTAREA"||e.target.tagName==="SELECT") return;
  if(e.key==="ArrowLeft") document.getElementById("prev").click();
  if(e.key==="ArrowRight") document.getElementById("next").click();
  if(e.key==="1"||e.key==="y") document.querySelector('#jbtns button[data-v="yes"]').click();
  if(e.key==="2"||e.key==="b") document.querySelector('#jbtns button[data-v="maybe"]').click();
  if(e.key==="3"||e.key==="n") document.querySelector('#jbtns button[data-v="no"]').click();
});
rebuild();
</script>
</body>
</html>
"""


def main():
    data = build_data()
    models = [{"id": m, "label": MODEL_LABELS[m]} for m in MODEL_ORDER]
    j = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    html = (PAGE.replace("__DATA__", j(data))
                .replace("__MODELS__", j(models))
                .replace("__DEF_A__", config.HSC_DEFINITIONS["current"].description)
                .replace("__DEF_B__", config.HSC_DEFINITIONS["era_neutral"].description))
    OUT.write_text(html, encoding="utf-8")

    contested = [s for s in data if s["gain"] and abs(s["gain"]) > 0]
    big = [s for s in data if s["gain"] and abs(s["gain"]) >= 0.25]
    print(f"wrote {OUT}  ({len(html)/1e6:.2f} MB, {len(data)} speeches)")
    print(f"  contested (definitions differ at all): {len(contested)}")
    print(f"  substantial gap (>=25pp)             : {len(big)}")
    print(f"  top: {data[0]['year']} '{data[0]['section'][:44]}' "
          f"{data[0]['cur']} -> {data[0]['era_rate']}")


if __name__ == "__main__":
    main()
