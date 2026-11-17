"""Screen-labelling page: one key per screen class, served locally.

    python tools/label_screens.py          then open http://127.0.0.1:8765 in a browser

Shows the newest batch from tools/pick_batch.py (http://127.0.0.1:8765/?batch=all for every full frame), with the
old bot's hint and log lines. Keys: see the legend on the page; Enter = take the hint; Left/Right move without
labelling. Labels go to <ADBOT2_HOME>/data/labels/screens.jsonl (append-only, latest wins); counts show at the side.
Only listens on 127.0.0.1. Does not open a browser by itself.

Crop heads (adbot2.crops): http://127.0.0.1:8765/crops?head=auto shows a grid of crops, pre-filled by the newest crop
model or by rules; hover + key (or click) fixes a tile, Enter saves the page and loads the next. Needs
tools/cache_crops.py first.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from datetime import datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from adbot2 import paths
from adbot2.crops import HEADS, Candidate, CropCache, CropLabels, auto_press_times, pick_page, propose
from adbot2.dataset import LabelStore, frame_path, load_manifest
from adbot2.oldlog import read_all_lines
from adbot2.screens import HOTKEYS, Screen

PAGE = r"""<!doctype html><meta charset="utf-8"><title>Screen labels</title>
<style>
 body{margin:0;font:14px system-ui;background:#16161a;color:#ddd;display:flex;height:100vh}
 #main{flex:1;display:flex;flex-direction:column;min-width:0}
 #img{flex:1;min-height:0;object-fit:contain;background:#000}
 #bar{padding:8px 12px;background:#222;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 #side{width:300px;padding:10px;overflow:auto;background:#1d1d22}
 .k{display:flex;gap:8px;padding:2px 4px;border-radius:4px}.k b{width:18px;color:#9cf}
 .cur{background:#2d4a6b}.hint{outline:1px dashed #fc6}
 pre{white-space:pre-wrap;font-size:12px;color:#aaa}
</style>
<div id="main"><img id="img"><div id="bar"></div></div>
<div id="side"><div id="legend"></div><p>Enter = take the hint (dashed). Arrows = move.</p>
<h4>Old bot log</h4><pre id="ctx"></pre><h4>Counts</h4><pre id="counts"></pre></div>
<script>
let items=[],i=0,keys={};const pre=new Image();
const $=id=>document.getElementById(id);
async function counts(){$('counts').textContent=await (await fetch('/api/counts')).text()}
async function load(){const q=new URLSearchParams(location.search).get('batch')||'';
 const d=await (await fetch('/api/items?batch='+encodeURIComponent(q))).json();items=d.items;keys=d.keys;
 batchName=d.batch;document.title='Screen labels '+d.batch;i=Math.max(0,items.findIndex(x=>!x.label));show();counts()}
function show(){const it=items[i];if(!it){$('bar').textContent='nothing to label';return}
 $('img').src='/img/'+it.id;
 const nx=items.find((x,j)=>j>i&&!x.label);if(nx)pre.src='/img/'+nx.id;
 const done=items.filter(x=>x.label).length;
 $('bar').textContent=`${batchName}  ${i+1}/${items.length} (${done} labelled)  ${it.time||it.orig_name}  label: ${it.label||'-'}${it.old&&!it.label?' (was '+it.old+')':''}  hint: ${it.hint||'-'}  ${it.why||''}`;
 if(done===items.length){$('bar').textContent+='   BATCH DONE';counts()}
 $('ctx').textContent=(it.context||[]).join('\n');
 $('legend').innerHTML=Object.entries(keys).map(([k,s])=>
  `<div class="k ${it.label===s?'cur':''} ${it.hint===s?'hint':''}"><b>${k}</b>${s}</div>`).join('')}
let batchName='';
async function label(s){const it=items[i];await fetch('/api/label',{method:'POST',body:JSON.stringify({id:it.id,screen:s})});
 it.label=s;const n=items.findIndex((x,j)=>j>i&&!x.label);
 if(n<0&&items.every(x=>x.label)&&!new URLSearchParams(location.search).get('batch')){
  const d=await (await fetch('/api/items')).json();   // batch done: on to the next one, if any
  if(d.batch!==batchName&&d.items.some(x=>!x.label)){items=d.items;batchName=d.batch;document.title='Screen labels '+d.batch;
   i=items.findIndex(x=>!x.label);show();counts();return}}
 i=n<0?Math.min(i+1,items.length-1):n;show()}
document.onkeydown=e=>{if(e.key==='ArrowRight'){i=Math.min(i+1,items.length-1);show()}
 else if(e.key==='ArrowLeft'){i=Math.max(i-1,0);show()}
 else if(e.key==='Enter'&&items[i]&&items[i].hint){label(items[i].hint)}
 else if(keys[e.key.toLowerCase()]){label(keys[e.key.toLowerCase()])}};
load();
</script>"""

CROP_PAGE = r"""<!doctype html><meta charset="utf-8"><title>Crop labels</title>
<style>
 body{margin:0;font:14px system-ui;background:#16161a;color:#ddd}
 #top{padding:8px 12px;background:#222;position:sticky;top:0;z-index:2}
 #grid{display:flex;flex-wrap:wrap;gap:6px;padding:8px}
 .t{position:relative;border:3px solid #333;border-radius:4px;line-height:0;cursor:pointer}
 .t.cur{border-color:#fff}
 .b{position:absolute;left:0;bottom:0;font:bold 12px system-ui;line-height:16px;padding:0 4px;border-radius:0 4px 0 0}
 .t.low .b{outline:2px dashed #fc6}
 a{color:#9cf}
</style>
<div id="top"></div><div id="grid"></div>
<script>
const COLS=['#2a6','#c33','#a6f','#777','#e90','#39c'];
let d={items:[]},cur=0;const $=id=>document.getElementById(id);
const head=new URLSearchParams(location.search).get('head')||'auto';
async function load(){d=await (await fetch('/api/crop_page?head='+head)).json();cur=0;draw()}
function badge(v){const k=d.classes.indexOf(v);return v===null?'<span class="b" style="background:#555">?</span>':
 `<span class="b" style="background:${COLS[k%COLS.length]}">${v}</span>`}
function draw(){
 const heads=d.heads.map(h=>h===head?`<b>${h}</b>`:`<a href="/crops?head=${h}">${h}</a>`).join(' | ');
 $('top').innerHTML=`${heads} &nbsp; <b>${d.help}</b> &nbsp; ? = skip tile, click = next value.
  <b>Enter</b> = save this page (${d.items.length}) and load the next. &nbsp; guesses: ${d.source}
  (dashed = unsure). &nbsp; labelled: ${Object.entries(d.counts).map(([k,v])=>k+' '+v).join(', ')||'none'}
  ${d.items.length?'':' &nbsp; NOTHING LEFT'}`;
 $('grid').innerHTML=d.items.map((x,j)=>`<div class="t ${j===cur?'cur':''} ${x.conf<0.9?'low':''}" data-j="${j}"
  title="${x.time||''}  screen: ${x.screen||'-'}  conf ${x.conf.toFixed(2)}"><img src="/crop/${head}/${x.id}"
  width="${d.w}" height="${d.h}">${badge(x.value)}</div>`).join('');
 document.querySelectorAll('.t').forEach(t=>{t.onmouseenter=()=>{cur=+t.dataset.j;mark()};
  t.onclick=()=>{const x=d.items[+t.dataset.j],k=d.classes.indexOf(x.value);
   x.value=k<0?d.classes[0]:(k+1<d.classes.length?d.classes[k+1]:null);redraw(+t.dataset.j)}})}
function mark(){document.querySelectorAll('.t').forEach((t,j)=>t.classList.toggle('cur',j===cur))}
function redraw(j){document.querySelector(`.t[data-j="${j}"] .b`).outerHTML=badge(d.items[j].value)}
async function save(){const values={};d.items.forEach(x=>{if(x.value)values[x.id]=x.value});
 await fetch('/api/crop_labels',{method:'POST',body:JSON.stringify({head,values})});load()}
document.onkeydown=e=>{const k=e.key.toLowerCase();
 if(e.key==='Enter'){save();return}
 if(e.key==='ArrowRight'){cur=Math.min(cur+1,d.items.length-1);mark();return}
 if(e.key==='ArrowLeft'){cur=Math.max(cur-1,0);mark();return}
 if(!d.items[cur])return;
 if(k==='?'||k==='/'){d.items[cur].value=null;redraw(cur);return}
 if(d.keys[k]){d.items[cur].value=d.keys[k];redraw(cur);cur=Math.min(cur+1,d.items.length-1);mark()}};
load();
</script>"""


class CropState:
    """Crop cache, crop labels and guesses for the grid page, loaded lazily (per head) once per server."""

    def __init__(self, home: Path, labels: Path, bot_dir: Path | None):
        self.home, self.cache, self.bot_dir = home, CropCache(home / "cache" / "crops"), bot_dir
        self.labels = CropLabels(labels)
        self.lock = threading.Lock()
        self._crops: dict[str, tuple[dict[str, int], np.ndarray]] = {}
        self._press: list | None = None

    def crops(self, head: str) -> tuple[dict[str, int], np.ndarray]:
        with self.lock:
            if head not in self._crops:
                ids, arr = self.cache.load(head)
                self._crops[head] = ({i: k for k, i in enumerate(ids)}, arr)
            return self._crops[head]

    def press_times(self) -> list:
        with self.lock:
            if self._press is None:
                ok = self.bot_dir is not None and self.bot_dir.is_dir()
                self._press = auto_press_times(read_all_lines(self.bot_dir)) if ok else []
            return self._press

    def model_guesses(self, head: str) -> tuple[str, dict[str, tuple[str, float]]]:
        """(source, id -> (value, confidence)) from the newest crop model's predictions, if any."""
        runs = sorted((self.home / "models" / "crops" / head).glob("*/preds.npz"))
        if not runs:
            return "rules", {}
        z = np.load(runs[-1])
        classes = HEADS[head].classes
        return f"model {runs[-1].parent.name}", {
            str(i): (classes[int(p.argmax())], float(p.max())) for i, p in zip(z["ids"], z["probs"])}

    def page(self, head: str, manifest: dict, screens: dict[str, str], n: int = 60) -> dict:
        h = HEADS[head]
        index, _ = self.crops(head)
        done = self.labels.of_head(head)
        source, model = self.model_guesses(head)
        press = self.press_times() if head == "auto" else []
        cands = []
        for fid in index:
            rec = manifest.get(fid)
            if rec is None or fid in done:
                continue
            rule = propose(head, rec, screens.get(fid) or rec.hint, press)
            guess, conf = model.get(fid, (rule, 1.0 if rule else 0.0))
            t = datetime.fromisoformat(rec.time) if rec.time else None
            cands.append(Candidate(fid, t, guess, conf, rule if model else None))
        page = sorted(pick_page(cands, n, seed=len(done)), key=lambda c: (str(c.guess), c.conf))
        w, hh = (round(v / h.store) for v in h.store_wh)

        def screen_of(fid):
            return screens.get(fid) or (manifest[fid].hint and manifest[fid].hint + "?")

        return {"head": head, "heads": list(HEADS), "help": h.help, "classes": list(h.classes),
                "keys": dict(zip(h.keys, h.classes)), "source": source, "w": w, "h": hh, "left": len(cands),
                "counts": dict(Counter(done.values())),
                "items": [{"id": c.id, "value": c.guess, "conf": c.conf, "screen": screen_of(c.id),
                           "time": c.time and f"{c.time:%d.%m %H:%M:%S}"} for c in page]}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, home: Path, manifest: Path, store: LabelStore, batches: Path, crops: CropState, **kw):
        self.home, self.manifest, self.store, self.batches, self.cs = home, manifest, store, batches, crops
        super().__init__(*a, **kw)

    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _since(self, d: dict) -> str | None:
        """A relabel batch (e.g. a review) counts only labels given after it was made."""
        return d.get("made") if d.get("relabel") else None

    def _batch(self, name: str) -> tuple[str, dict | None]:
        """(name, [{id, why}]) of a named batch, or (name "") the oldest batch with unlabelled frames, else the
        newest one; items None = every full frame."""
        if name == "all":
            return "all", None
        files = sorted(self.batches.glob("*.json"), key=lambda f: (f.stat().st_mtime, f.name)) \
            if self.batches.is_dir() else []
        if name:
            files = [f for f in files if f.stem == name]
        if not files:
            return "all", None
        loaded = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        todo = [d for d in loaded if any(self.store.get(i["id"], self._since(d)) is None for i in d["items"])]
        d = todo[0] if todo and not name else loaded[-1]
        return d["name"], d

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/crops":
            return self._send(200, CROP_PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/crop_page":
            head = parse_qs(url.query).get("head", ["auto"])[0]
            if head not in HEADS:
                return self._send(404, b"no such head", "text/plain")
            page = self.cs.page(head, load_manifest(self.manifest), self.store.all())
            return self._send(200, json.dumps(page).encode(), "application/json")
        if url.path.startswith("/crop/"):
            parts = url.path.split("/")
            if len(parts) == 4 and parts[2] in HEADS:
                index, arr = self.cs.crops(parts[2])
                k = index.get(parts[3])
                if k is not None:
                    ok, png = cv2.imencode(".png", np.asarray(arr[k]))
                    if ok:
                        return self._send(200, png.tobytes(), "image/png")
        if url.path == "/api/items":
            manifest = load_manifest(self.manifest)
            name, batch = self._batch(parse_qs(url.query).get("batch", [""])[0])
            since = None
            if batch is None:
                recs = sorted((r for r in manifest.values() if r.full_frame), key=lambda r: (r.time or "", r.id))
                pairs = [(r, None) for r in recs]
            else:
                since = self._since(batch)
                pairs = [(manifest[b["id"]], b.get("why")) for b in batch["items"] if b["id"] in manifest]
            items = [{"id": r.id, "orig_name": r.orig_name, "time": r.time, "hint": r.hint, "context": r.context,
                      "why": why, "label": self.store.get(r.id, since), "old": self.store.get(r.id)}
                     for r, why in pairs]
            keys = {k: s.value for k, s in HOTKEYS.items()}
            return self._send(200, json.dumps({"batch": name, "items": items, "keys": keys}).encode(),
                              "application/json")
        if url.path == "/api/counts":
            c = Counter(self.store.all().values())
            rows = [f"{s.value:18} {c.get(s.value, 0)}" for s in Screen] + [f"{'total':18} {sum(c.values())}"]
            return self._send(200, "\n".join(rows).encode(), "text/plain; charset=utf-8")
        if url.path.startswith("/img/"):
            rec = load_manifest(self.manifest).get(url.path[5:])  # by id: no file paths from the browser
            if rec:
                f = frame_path(rec, self.home)
                if f.is_file():
                    ctype = "image/png" if f.suffix.lower() == ".png" else "image/jpeg"
                    return self._send(200, f.read_bytes(), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/crop_labels":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                classes = HEADS[body["head"]].classes
                values = {str(k): v for k, v in body["values"].items() if v in classes}
            except (ValueError, KeyError, TypeError, AttributeError):
                return self._send(400, b"bad labels", "text/plain")
            self.cs.labels.set_many(body["head"], values)
            return self._send(200, b"{}", "application/json")
        if self.path != "/api/label":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            screen = Screen(body["screen"]).value  # rejects unknown classes
            fid = str(body["id"])
        except (ValueError, KeyError, TypeError):
            return self._send(400, b"bad label", "text/plain")
        self.store.set(fid, screen)
        self._send(200, b"{}", "application/json")


def make_server(port: int, home: Path, manifest: Path, labels: Path, batches: Path,
                bot_dir: Path | None = None) -> ThreadingHTTPServer:
    crops = CropState(home, labels.parent / "crops.jsonl", bot_dir)
    handler = partial(Handler, home=home, manifest=manifest, store=LabelStore(labels), batches=batches, crops=crops)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="screen-labelling page")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bot-dir", type=Path, default=Path(__file__).resolve().parents[2] / "AD-bot",
                    help="old bot folder: its log gives the AUTO-off moments for the crop guesses")
    args = ap.parse_args(argv)
    srv = make_server(args.port, paths.home(), paths.manifest_path(), paths.screen_labels_path(),
                      paths.data_dir() / "labels" / "batches", args.bot_dir)
    print(f"open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
