"""Local web UI for labeling your utterance set. Standard library only; binds to 127.0.0.1.

  uv run python -m dealer_core.labeling --input my_utterances.txt --audio-dir benchmarks/recordings

Every label is a human click (nothing is predicted or pre-filled). Progress autosaves to <out>.labels.jsonl and the CSV
used by `dealer_core.eval.compare --csv` is rewritten after every save, so you can stop and resume at any time.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..dialogue.states import SLOT_SCHEMAS, STATE_SPEC, Intent, State
from . import core

ROOT = Path(__file__).resolve().parents[3]

STATE_HELP = {k: v[0] for k, v in core.STATE_GUIDE.items()}


def schema_json() -> dict:
    return {
        "states": [{"value": s.value, "help": STATE_HELP[s.value], "example": core.STATE_GUIDE[s.value][1],
                    "valid": [i.value for i in STATE_SPEC[s][0]]} for s in STATE_SPEC],
        "intents": [{"value": i.value, "doc": core.INTENT_GUIDE[i.value][0], "example": core.INTENT_GUIDE[i.value][1],
                     "slots": core.INTENT_SLOTS.get(i, []), "state_changing": core.intent_is_state_changing(i.value)} for i in Intent],
        "slots": {k: {"type": "int" if k in core.INT_SLOTS else "enum" if k in core.ENUM_SLOTS else "text",
                      "enum": core.ENUM_SLOTS.get(k, [])} for k in SLOT_SCHEMAS},
        "targets": core.TARGETS,
    }


class App:
    def __init__(self, store: core.LabelStore, stock_ids: list[str], existing: dict[str, str]) -> None:
        self.store, self.stock_ids, self.existing = store, stock_ids, existing
        self.known = set(store.by_id)
        self.lock = threading.Lock()

    def item_json(self, n: int) -> dict:
        it = self.store.items[n]
        lb = self.store.labels.get(it.id)
        dup = self.existing.get(re.sub(r"\W+", " ", it.utterance.lower()).strip())
        return {"index": n, "id": it.id, "utterance": it.utterance, "source": it.source, "audio": bool(it.audio),
                "duplicate_of": dup, "label": lb.__dict__ if lb else None}

    def summary(self) -> dict:
        labeled = [l for l in self.store.labels.values() if not l.discard]
        return {"total": len(self.store.items), "labeled": len(labeled), "discarded": len(self.store.labels) - len(labeled),
                "first_unlabeled": self.store.first_unlabeled(), "coverage": self.store.coverage()}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            if path == "/api/schema":
                return self._json({**schema_json(), "stock_ids": app.stock_ids})
            if path == "/api/summary":
                return self._json(app.summary())
            m = re.fullmatch(r"/api/item/(\d+)", path)
            if m and int(m[1]) < len(app.store.items):
                return self._json(app.item_json(int(m[1])))
            m = re.fullmatch(r"/audio/([\w.\-]+)", path)
            if m:
                it = app.store.by_id.get(m[1])
                if it and it.audio and it.audio.exists():
                    return self._send(200, it.audio.read_bytes(), "audio/wav")
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            try:
                raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                with app.lock:
                    label = core.validate_label(raw, app.known)
                    app.store.save(label)
                return self._json({"ok": True, **app.summary()})
            except ValueError as e:
                return self._json({"ok": False, "error": str(e)}, 400)

    return Handler


def serve(app: App, port: int, open_browser: bool) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}/"
    print(f"Labeling {len(app.store.items)} utterances. Open {url}\nCSV: {app.store.csv_path}\nCtrl+C to stop; progress is saved after every label.")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help=".txt (one utterance per line) or .csv with an `utterance` column")
    src.add_argument("--input-dir", type=Path, help="folder with one utterance per file (id = file name); text is used VERBATIM")
    ap.add_argument("--pattern", default="spont*.txt", help="with --input-dir: which files to load (default spont*.txt)")
    ap.add_argument("--audio-dir", type=Path, default=None, help="folder with <id>.wav for spoken utterances")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "nlu_sets" / "mine.csv")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args()
    items = (core.load_items_from_dir(a.input_dir, a.pattern, a.audio_dir) if a.input_dir
             else core.load_items(a.input, a.audio_dir))
    store = core.LabelStore(items, a.out.with_suffix(".labels.jsonl"), a.out)
    stock = [f"S{1000 + i}" for i in range(60)]
    serve(App(store, stock, core.existing_utterances(ROOT / "data" / "nlu_sets")), a.port, a.open)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Utterance labeling</title>
<style>
:root{--bg:#fff;--fg:#1b1b1f;--mute:#667;--line:#d9dbe3;--card:#f6f7fb;--acc:#2a5bd7;--bad:#b3261e;--ok:#17803d;--warn:#8a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#15161a;--fg:#ecedf2;--mute:#9aa0b4;--line:#2c2f3a;--card:#1e2028;--acc:#7ea2ff;--bad:#ff8a80;--ok:#6fd49a;--warn:#f0c36a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px;display:grid;grid-template-columns:1fr 280px;gap:16px}
@media(max-width:820px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
h1{font-size:15px;margin:0 0 10px;color:var(--mute);font-weight:600}
.utt{font-size:24px;font-weight:600;margin:6px 0 10px}
.badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);color:var(--mute);margin-right:6px}
.badge.warn{color:var(--warn);border-color:var(--warn)}.badge.ok{color:var(--ok);border-color:var(--ok)}
label{display:block;font-size:13px;color:var(--mute);margin:12px 0 4px}
select,input[type=text],input[type=number]{width:100%;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.help{font-size:12px;color:var(--mute);margin-top:3px;min-height:1.2em}
.slots{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button{padding:9px 14px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);font:inherit;cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}button:disabled{opacity:.45;cursor:not-allowed}
.bar{height:6px;background:var(--line);border-radius:4px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--acc)}
.cov{display:grid;grid-template-columns:1fr auto;gap:2px 8px;font-size:12px;margin-bottom:6px}
.err{color:var(--bad);min-height:1.4em;margin-top:8px}.sc{font-weight:700}
.top{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-bottom:8px}
</style></head><body><div class="wrap">
<div class="card" id="main">
  <div class="top"><h1 id="prog">Loading...</h1><div><button id="back">Back</button> <button id="jump">First unlabeled</button></div></div>
  <div id="badges"></div><div class="utt" id="utt"></div><audio id="audio" controls style="width:100%;display:none"></audio>
  <div class="row"><div><label for="state">State the agent was in (for these clips: where it would most naturally occur)</label><select id="state"></select><div class="help" id="statehelp"></div></div>
  <div><label for="car">Car in focus (stock id, blank = none)</label><input type="text" id="car" list="stocks" placeholder="e.g. S1042"><datalist id="stocks"></datalist></div></div>
  <label style="display:flex;gap:6px;align-items:center;margin-top:10px"><input type="checkbox" id="sg"> The state is a guess: this utterance would fit more than one state (or any state)</label>
  <div id="sgbox" style="display:none"><label for="others">Other states where it would be equally natural (Ctrl/Cmd-click to pick several; optional)</label><select id="others" multiple size="7"></select></div>
  <label for="intent">What the caller meant (gold intent)</label><select id="intent"></select><div class="help" id="intenthelp"></div>
  <div id="slotsbox"><div style="font-size:13px;color:var(--mute);margin:12px 0 4px">Slots the caller actually said <label style="display:inline;margin:0 0 0 8px"><input type="checkbox" id="allslots"> show all slots</label></div><div class="slots" id="slots"></div></div>
  <label for="notes">Notes (required if you mark it ambiguous without an "also acceptable" intent)</label><input type="text" id="notes">
  <label style="display:flex;gap:6px;align-items:center;margin-top:10px"><input type="checkbox" id="amb"> Ambiguous / more than one intent in this utterance</label>
  <div id="alsobox" style="display:none"><label for="also">Other intents that would also be an acceptable reading (Ctrl/Cmd-click to pick several)</label><select id="also" multiple size="7"></select></div>
  <label style="display:flex;gap:6px;align-items:center;margin-top:10px"><input type="checkbox" id="keep"> keep state and car for the next utterance (still your choice; nothing else carries over)</label>
  <div class="err" id="err"></div>
  <div style="display:flex;gap:8px;margin-top:8px"><button class="primary" id="save" disabled>Save and next (Ctrl+Enter)</button><button id="skip">Skip</button><button id="discard">Discard (unclear / unusable)</button></div>
</div>
<div class="card"><details id="guide"><summary style="cursor:pointer;font-weight:600">Definitions and examples (states, intents)</summary><div id="guidebody" style="font-size:12px;margin-top:8px"></div></details>
<h1 style="margin-top:14px">Coverage (gold intents)</h1><div id="cov"></div><div class="help" id="covhelp"></div></div>
</div>
<script>
let S=null, items=0, idx=0, sum=null, cur=null;
const $=id=>document.getElementById(id);
async function api(p,o){const r=await fetch(p,o);return r.json()}
function opt(v,t,sel){const o=document.createElement('option');o.value=v;o.textContent=t;if(sel)o.selected=true;return o}
async function init(){
  S=await api('/api/schema'); sum=await api('/api/summary'); items=sum.total;
  $('stocks').innerHTML=S.stock_ids.map(s=>`<option value="${s}">`).join('');
  const st=$('state'); st.append(opt('','— choose a state —')); S.states.forEach(s=>st.append(opt(s.value,s.value)));
  st.onchange=()=>{renderIntent();$('statehelp').textContent=(S.states.find(s=>s.value==st.value)||{}).help||'';validate()};
  $('intent').onchange=()=>{renderSlots();validate()}; $('allslots').onchange=renderSlots;
  $('amb').onchange=()=>{$('alsobox').style.display=$('amb').checked?'':'none'};
  $('sg').onchange=()=>{$('sgbox').style.display=$('sg').checked?'':'none'};
  S.states.forEach(s=>$('others').append(opt(s.value,s.value)));
  S.intents.forEach(i=>$('also').append(opt(i.value,i.value)));
  $('guidebody').innerHTML='<b>States</b> (what the agent was doing)<br>'+S.states.map(s=>`<div style="margin:4px 0"><b>${s.value}</b>: ${s.help}<br><i>e.g. "${s.example}"</i></div>`).join('')+'<br><b>Intents</b> (what the caller meant)<br>'+S.intents.map(i=>`<div style="margin:4px 0"><b>${i.value}${i.state_changing?' *':''}</b>: ${i.doc}<br><i>e.g. "${i.example}"</i></div>`).join('');
  $('save').onclick=save; $('skip').onclick=()=>go(idx+1); $('back').onclick=()=>go(idx-1);
  $('jump').onclick=async()=>{sum=await api('/api/summary');go(sum.first_unlabeled)};
  $('discard').onclick=()=>save(true);
  document.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();save()}});
  go(sum.first_unlabeled>=items?0:sum.first_unlabeled); renderCov();
}
function renderIntent(keepVal){
  const stv=$('state').value, sel=$('intent'); const prev=keepVal??sel.value; sel.innerHTML=''; sel.append(opt('','— choose an intent —'));
  const valid=(S.states.find(s=>s.value==stv)||{valid:[]}).valid;
  const g1=document.createElement('optgroup'); g1.label=stv?`Valid in "${stv}"`:'All intents'; const g2=document.createElement('optgroup'); g2.label='NOT valid in this state (choose only if that is truly what the caller meant)';
  S.intents.forEach(i=>{const o=opt(i.value,i.value+(i.state_changing?'  *':''),i.value===prev); (stv&&!valid.includes(i.value)?g2:g1).append(o)});
  sel.append(g1); if(g2.children.length) sel.append(g2); sel.value=prev||''; renderSlots();
}
function renderSlots(vals){
  const iv=$('intent').value, it=S.intents.find(i=>i.value===iv); $('intenthelp').textContent=it?it.doc+(it.state_changing?'   (state-changing: gets a precision gate)':''):'';
  const fields=$('allslots').checked?Object.keys(S.slots):(it?it.slots:[]); const box=$('slots'); const old=vals||currentSlots(); box.innerHTML='';
  $('slotsbox').style.display=(fields.length||it)?'':'none';
  fields.forEach(f=>{const d=S.slots[f]; const w=document.createElement('div'); const l=document.createElement('label'); l.textContent=f+(d.type==='int'?' (number)':''); w.append(l);
    let el; if(d.type==='enum'){el=document.createElement('select'); el.append(opt('','—')); d.enum.forEach(e=>el.append(opt(e,e)))} else {el=document.createElement('input'); el.type='text'; if(d.type==='int') el.inputMode='numeric'}
    el.dataset.slot=f; el.value=old[f]??''; el.oninput=validate; w.append(el); box.append(w)});
}
function currentSlots(){const o={};document.querySelectorAll('[data-slot]').forEach(e=>{if(e.value!=='')o[e.dataset.slot]=e.value});return o}
function validate(){ $('save').disabled=!($('state').value&&$('intent').value) }
async function go(n){
  if(n<0||n>=items) return; idx=n; cur=await api('/api/item/'+n); $('err').textContent='';
  $('prog').textContent=`Utterance ${n+1} of ${items}`; $('utt').textContent=cur.utterance;
  $('badges').innerHTML=`<span class="badge">${cur.source}</span><span class="badge">${cur.id}</span>`+(cur.label?`<span class="badge ok">${cur.label.discard?'discarded':'labeled'}</span>`:'')+(cur.label&&cur.label.ambiguous?'<span class="badge warn">ambiguous</span>':'')+(cur.label&&cur.label.state_guess?'<span class="badge warn">state is a guess</span>':'')+(cur.duplicate_of?`<span class="badge warn">already in the "${cur.duplicate_of}" set: skip or discard to avoid leakage</span>`:'');
  const au=$('audio'); if(cur.audio){au.src='/audio/'+cur.id; au.style.display='block'} else {au.removeAttribute('src'); au.style.display='none'}
  const lb=cur.label&&!cur.label.discard?cur.label:null, keep=$('keep').checked;
  $('amb').checked=!!(lb&&lb.ambiguous); $('alsobox').style.display=$('amb').checked?'':'none';
  $('sg').checked=!!(lb&&lb.state_guess); $('sgbox').style.display=$('sg').checked?'':'none';
  [...$('others').options].forEach(o=>o.selected=!!(lb&&lb.other_states&&lb.other_states.includes(o.value)));
  [...$('also').options].forEach(o=>o.selected=!!(lb&&lb.also_ok&&lb.also_ok.includes(o.value)));
  const prevState=$('state').value, prevCar=$('car').value;
  $('state').value=lb?lb.state:(keep?prevState:''); $('car').value=lb?lb.car:(keep?prevCar:''); $('notes').value=cur.label?cur.label.notes:'';
  $('statehelp').textContent=(S.states.find(s=>s.value==$('state').value)||{}).help||'';
  renderIntent(lb?lb.intent:''); if(lb) renderSlots(Object.fromEntries(Object.entries(lb.slots).map(([k,v])=>[k,String(v)]))); validate();
}
async function save(discard){
  const body={id:cur.id,discard:discard===true,state:$('state').value,car:$('car').value,intent:$('intent').value,slots:currentSlots(),notes:$('notes').value,ambiguous:$('amb').checked,also_ok:[...$('also').selectedOptions].map(o=>o.value),state_guess:$('sg').checked,other_states:[...$('others').selectedOptions].map(o=>o.value)};
  const r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const j=await r.json();
  if(!j.ok){$('err').textContent=j.error;return} sum=j; renderCov(); go(Math.min(idx+1,items-1));
}
function renderCov(){
  const T=S.targets, c=sum.coverage; const sc=new Set(S.intents.filter(i=>i.state_changing).map(i=>i.value)); let h='';
  S.intents.forEach(i=>{const n=c[i.value]||0, pct=Math.min(100,100*n/T.v1); h+=`<div class="cov"><span class="${sc.has(i.value)?'sc':''}">${i.value}${sc.has(i.value)?' *':''}</span><span>${n} / ${T.v1}</span><div class="bar" style="grid-column:1/3"><i style="width:${pct}%"></i></div></div>`});
  $('cov').innerHTML=`<div style="margin-bottom:8px"><b>${sum.labeled}</b> labeled, ${sum.discarded} discarded, ${sum.total-sum.labeled-sum.discarded} left</div>`+h;
  $('covhelp').innerHTML=`* = state-changing. Bars fill at ${T.v1} per intent (a usable first gate decision). First batch: ~${T.batch1} each. A 98% claim for ONE intent needs ${T.gate} correct rule firings with zero errors.`;
}
init();
</script></body></html>"""

if __name__ == "__main__":
    main()
