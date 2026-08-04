"""Build a self-contained per-speech capped-vs-uncapped viewer.

For every pilot speech it shows, side by side, the sub-topics the models
returned under the capped prompt (``task=v1``, "at most 5") and the uncapped
prompt (``task=v1_nocap``, no limit), so the anchor effect from Section 5 of the
brief can be inspected case by case. Output is one standalone HTML file with the
data embedded (no external resources), matching the offline style of the
existing explorer.

Usage:  python -m hansard_llm.docs.build_cap_explorer   (or run this file directly)
Writes: hansard_llm/docs/cap_explorer.html
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hansard_llm import run, sample

OUT = Path(__file__).resolve().parent / "cap_explorer.html"

MODEL_LABELS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B",
    "google/gemma-3-27b-it": "Gemma-27B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-70B",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "Qwen3-235B (ref)",
}
MODEL_ORDER = list(MODEL_LABELS)
FORMATS = ("json", "free")


def _cell(row) -> dict:
    """Reduce one log row to the per-cell payload: presence, sub-topics, quote."""
    subs = row["subthemes"]
    return {
        "present": bool(row["mentions_topic"]) if row["mentions_topic"] is not None else None,
        "subs": list(subs) if isinstance(subs, list) else [],
        "quote": (row["evidence_quote"] or "")[:400],
    }


def build_data() -> list[dict]:
    """Return one record per pilot speech with capped/uncapped cells per
    format and model, plus per-speech cap-effect summaries, sorted by
    cap effect (largest first)."""
    df = run.load_legacy()
    # The definition filter matters: the definition-sensitivity arm also runs at
    # role=none / v1_nocap / both formats, so without it those cells would be
    # pooled into the uncapped side and the anchor effect would be measured
    # against a mixture of definitions. pool == "pilot" likewise excludes the
    # retrieval spot-check rows logged under the same labels.
    df = df[(df["pool"] == "pilot")
            & (df["condition"] == "temp0") & (df["role"] == "none")
            & (df["task"].isin(["v1", "v1_nocap"]))
            & (df["definition"] == "current")
            & (df["output_format"].isin(FORMATS))].copy()

    meta = sample.load_sample().set_index("speech_id")
    out = []
    for sid, g in df.groupby("speech_id"):
        if sid not in meta.index:
            continue
        m = meta.loc[sid]
        cells = {f: {} for f in FORMATS}
        cap_counts, unc_counts, deltas = [], [], []
        for fmt in FORMATS:
            for mid in MODEL_ORDER:
                cap = g[(g["output_format"] == fmt) & (g["model_id"] == mid) & (g["task"] == "v1")]
                unc = g[(g["output_format"] == fmt) & (g["model_id"] == mid) & (g["task"] == "v1_nocap")]
                cc = _cell(cap.iloc[0]) if len(cap) else None
                uc = _cell(unc.iloc[0]) if len(unc) else None
                cells[fmt][mid] = {"capped": cc, "uncapped": uc}
                if cc and cc["present"]:
                    cap_counts.append(len(cc["subs"]))
                if uc and uc["present"]:
                    unc_counts.append(len(uc["subs"]))
                # Anchor effect: measured only where BOTH arms judged the speech
                # H&SC, so a presence disagreement doesn't masquerade as a cap
                # effect. This is capped_count - uncapped_count on matched cells.
                if cc and uc and cc["present"] and uc["present"]:
                    deltas.append(len(cc["subs"]) - len(uc["subs"]))
        out.append({
            "id": str(sid),
            "year": int(m["year"]),
            "chamber": str(m["chamber"]),
            "type": str(m["speech_type"]),
            "words": int(m["word_count"]),
            "section": str(m["section_title"]),
            "era": str(m["era"]),
            "text": str(m["speech_text"]),
            "cap_mean": round(float(np.mean(cap_counts)), 2) if cap_counts else None,
            "unc_mean": round(float(np.mean(unc_counts)), 2) if unc_counts else None,
            "cap_effect": round(float(np.mean(deltas)), 2) if deltas else None,
            "n_matched": len(deltas),
            "cells": cells,
        })
    # Sort by the anchor effect (largest first); speeches with no matched cell
    # (never judged H&SC by both arms) sort to the end.
    out.sort(key=lambda s: (s["cap_effect"] is not None, s["cap_effect"] or 0),
             reverse=True)
    return out


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cap effect explorer — capped vs uncapped sub-topics</title>
<style>
:root{
  --bg:#ffffff; --panel:#f6f7f9; --ink:#1c1c1c; --muted:#6b6b6b; --line:#e2e5e9;
  --cap:#7c848c; --capbg:#eceef0; --unc:#2A6F97; --uncbg:#e9f1f6; --anchor:#c9781a;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#15181c; --panel:#1e2228; --ink:#e9ebee; --muted:#9aa3ad; --line:#2c313a;
  --cap:#9aa3ad; --capbg:#262b32; --unc:#5aa9d6; --uncbg:#1d2a33; --anchor:#e0a24a;
}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 "Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:18px 22px;border-bottom:1px solid var(--line)}
h1{font-size:19px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13.5px;max-width:70ch}
.controls{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;padding:12px 22px;
  border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5}
.controls label{font-size:12.5px;color:var(--muted);margin-right:4px}
select,button{font:inherit;font-size:13px;padding:5px 9px;border:1px solid var(--line);
  border-radius:7px;background:var(--bg);color:var(--ink);cursor:pointer}
select#pick{min-width:min(560px,60vw)}
button:hover{border-color:var(--unc)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{border:0;border-radius:0;background:var(--bg)}
.seg button.on{background:var(--unc);color:#fff}
main{padding:18px 22px;max-width:1100px}
.meta{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--muted);font-size:13px;margin-bottom:10px}
.meta b{color:var(--ink);font-weight:600}
.effect{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12.5px;font-weight:600}
.speech{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:13px 15px;
  max-height:210px;overflow:auto;font-size:14px;margin-bottom:18px;white-space:pre-wrap}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;vertical-align:top;padding:9px 11px;border-top:1px solid var(--line)}
th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
td.model{white-space:nowrap;font-weight:600;font-size:13.5px;width:150px}
.col-cap{background:var(--capbg)} .col-unc{background:var(--uncbg)}
.count{display:inline-block;min-width:20px;text-align:center;font-size:11.5px;font-weight:700;
  padding:1px 6px;border-radius:20px;margin-right:6px;color:#fff}
.count.cap{background:var(--cap)} .count.unc{background:var(--unc)}
.count.anchor{background:var(--anchor)}
.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
.chip{font-size:12.5px;padding:2px 8px;border-radius:6px;border:1px solid var(--line);background:var(--bg)}
.none{color:var(--muted);font-style:italic;font-size:12.5px}
.hint{font-size:12px;color:var(--muted);margin:2px 0 14px}
.anchor-key{color:var(--anchor);font-weight:600}
</style>
</head>
<body>
<header>
  <h1>Cap effect explorer</h1>
  <div class="sub">Per speech, the sub-topics each model returned under the
  <b>capped</b> prompt ("at most 5") and the <b>uncapped</b> prompt (no limit),
  role and temperature held fixed. A capped count of <span class="anchor-key">5</span>
  is highlighted: that is where the cap tends to bind. Speeches are sorted by how
  much the cap changed the average count, so the clearest cases come first.</div>
</header>
<div class="controls">
  <span><label>Speech</label><select id="pick"></select></span>
  <button id="prev">Prev</button><button id="next">Next</button>
  <span><label>Sort</label><select id="sort">
    <option value="effect">Cap effect (largest first)</option>
    <option value="year">Year</option>
  </select></span>
  <span><label>Format</label><span class="seg" id="fmt">
    <button data-f="json" class="on">json</button><button data-f="free">free</button>
  </span></span>
</div>
<main>
  <div class="meta" id="meta"></div>
  <div class="speech" id="speech"></div>
  <div class="hint">Grey = capped ("at most 5"), the anchored/artefactual arm. Blue = uncapped, the honest arm. Counts are the number of sub-topics returned.</div>
  <table><thead><tr><th>Model</th><th class="col-cap">Capped (at most 5)</th><th class="col-unc">Uncapped (no limit)</th></tr></thead>
  <tbody id="rows"></tbody></table>
</main>
<script id="data" type="application/json">__DATA__</script>
<script id="models" type="application/json">__MODELS__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const MODELS = JSON.parse(document.getElementById("models").textContent);
let order = DATA.map((_,i)=>i), pos = 0, fmt = "json";

const pick = document.getElementById("pick");
function esc(s){const d=document.createElement("div");d.textContent=s;return d.innerHTML;}
function effColor(e){ if(e==null) return "#7c848c"; if(e>=1.0) return "#c0392b"; if(e>0.25) return "#c9781a"; if(e<=-0.25) return "#2A6F97"; return "#7c848c"; }
function effLabel(e){ return e==null ? "n/a" : (e>0?"+":"")+e; }
const num = v => v==null ? "—" : v;

function rebuildPicker(){
  const key = document.getElementById("sort").value;
  const eff = i => DATA[i].cap_effect==null ? -99 : DATA[i].cap_effect;
  order = DATA.map((_,i)=>i);
  if(key==="effect") order.sort((a,b)=>eff(b)-eff(a));
  else order.sort((a,b)=>DATA[a].year-DATA[b].year || eff(b)-eff(a));
  pick.innerHTML = order.map((idx,i)=>{const s=DATA[idx];
    return `<option value="${i}">${s.year} · ${esc(s.section).slice(0,60)} — cap effect ${effLabel(s.cap_effect)}</option>`;
  }).join("");
  pick.value = 0; pos = 0; render();
}
function cellHtml(c, kind){
  if(!c) return `<span class="none">no data</span>`;
  if(c.present===false) return `<span class="none">— not judged H&amp;SC —</span>`;
  const n=c.subs.length;
  const anchor = (kind==="cap" && n===5) ? " anchor" : "";
  const badge = `<span class="count ${kind}${anchor}">${n}</span>`;
  if(!n) return `${badge}<span class="none">(none listed)</span>`;
  const chips = c.subs.map(s=>`<span class="chip">${esc(s)}</span>`).join("");
  return `${badge}<div class="chips">${chips}</div>`;
}
function render(){
  const s = DATA[order[pos]];
  const ec = effColor(s.cap_effect);
  document.getElementById("meta").innerHTML =
    `<span><b>${s.year}</b> · ${esc(s.chamber)}</span>`+
    `<span>${esc(s.section)}</span>`+
    `<span>${s.words} words · ${esc(s.type)} · ${esc(s.era)}</span>`+
    `<span class="effect" style="background:${ec};color:#fff">cap effect ${effLabel(s.cap_effect)}`+
    ` (capped ${num(s.cap_mean)} vs uncapped ${num(s.unc_mean)})</span>`;
  document.getElementById("speech").textContent = s.text;
  const cells = s.cells[fmt];
  document.getElementById("rows").innerHTML = MODELS.map(m=>{
    const c = cells[m.id] || {};
    return `<tr><td class="model">${esc(m.label)}</td>`+
           `<td class="col-cap">${cellHtml(c.capped,"cap")}</td>`+
           `<td class="col-unc">${cellHtml(c.uncapped,"unc")}</td></tr>`;
  }).join("");
  pick.value = pos;
}
document.getElementById("sort").onchange = rebuildPicker;
pick.onchange = ()=>{pos=+pick.value; render();};
document.getElementById("prev").onclick = ()=>{pos=(pos-1+order.length)%order.length; render();};
document.getElementById("next").onclick = ()=>{pos=(pos+1)%order.length; render();};
document.querySelectorAll("#fmt button").forEach(b=>b.onclick=()=>{
  fmt=b.dataset.f;
  document.querySelectorAll("#fmt button").forEach(x=>x.classList.toggle("on",x===b));
  render();
});
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft"){document.getElementById("prev").click();}
  if(e.key==="ArrowRight"){document.getElementById("next").click();}
});
rebuildPicker();
</script>
</body>
</html>
"""


def main():
    """Embed the data into the HTML template and write cap_explorer.html."""
    data = build_data()
    models = [{"id": m, "label": MODEL_LABELS[m]} for m in MODEL_ORDER]
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    mpayload = json.dumps(models, ensure_ascii=False).replace("</", "<\\/")
    html = PAGE.replace("__DATA__", payload).replace("__MODELS__", mpayload)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html)/1e6:.2f} MB, {len(data)} speeches)")
    top = data[0]
    print(f"largest cap effect: {top['year']} '{top['section'][:40]}' "
          f"capped {top['cap_mean']} vs uncapped {top['unc_mean']}")


if __name__ == "__main__":
    main()
