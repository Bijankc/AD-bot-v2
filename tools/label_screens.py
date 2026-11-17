"""Screen-labelling page: one key per screen class, served locally.

    python tools/label_screens.py          then open http://127.0.0.1:8765 in a browser

Shows the newest batch from tools/pick_batch.py (http://127.0.0.1:8765/?batch=all for every full frame), with the
old bot's hint and log lines. Keys: see the legend on the page; Enter = take the hint; Left/Right move without
labelling. Labels go to <ADBOT2_HOME>/data/labels/screens.jsonl (append-only, latest wins); counts show at the side.
Only listens on 127.0.0.1. Does not open a browser by itself.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from adbot2 import paths
from adbot2.dataset import LabelStore, frame_path, load_manifest
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
 document.title='Screen labels '+d.batch;i=Math.max(0,items.findIndex(x=>!x.label));show();counts()}
function show(){const it=items[i];if(!it){$('bar').textContent='nothing to label';return}
 $('img').src='/img/'+it.id;
 const nx=items.find((x,j)=>j>i&&!x.label);if(nx)pre.src='/img/'+nx.id;
 const done=items.filter(x=>x.label).length;
 $('bar').textContent=`${i+1}/${items.length} (${done} labelled)  ${it.time||it.orig_name}  label: ${it.label||'-'}  hint: ${it.hint||'-'}  ${it.why||''}`;
 if(done===items.length){$('bar').textContent+='   BATCH DONE';counts()}
 $('ctx').textContent=(it.context||[]).join('\n');
 $('legend').innerHTML=Object.entries(keys).map(([k,s])=>
  `<div class="k ${it.label===s?'cur':''} ${it.hint===s?'hint':''}"><b>${k}</b>${s}</div>`).join('')}
async function label(s){const it=items[i];await fetch('/api/label',{method:'POST',body:JSON.stringify({id:it.id,screen:s})});
 it.label=s;const n=items.findIndex((x,j)=>j>i&&!x.label);i=n<0?Math.min(i+1,items.length-1):n;show()}
document.onkeydown=e=>{if(e.key==='ArrowRight'){i=Math.min(i+1,items.length-1);show()}
 else if(e.key==='ArrowLeft'){i=Math.max(i-1,0);show()}
 else if(e.key==='Enter'&&items[i]&&items[i].hint){label(items[i].hint)}
 else if(keys[e.key.toLowerCase()]){label(keys[e.key.toLowerCase()])}};
load();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, home: Path, manifest: Path, store: LabelStore, batches: Path, **kw):
        self.home, self.manifest, self.store, self.batches = home, manifest, store, batches
        super().__init__(*a, **kw)

    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _batch(self, name: str) -> tuple[str, list[dict] | None]:
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
        todo = [d for d in loaded if any(self.store.get(i["id"]) is None for i in d["items"])]
        d = todo[0] if todo and not name else loaded[-1]
        return d["name"], d["items"]

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/items":
            manifest = load_manifest(self.manifest)
            name, batch = self._batch(parse_qs(url.query).get("batch", [""])[0])
            if batch is None:
                recs = sorted((r for r in manifest.values() if r.full_frame), key=lambda r: (r.time or "", r.id))
                pairs = [(r, None) for r in recs]
            else:
                pairs = [(manifest[b["id"]], b.get("why")) for b in batch if b["id"] in manifest]
            items = [{"id": r.id, "orig_name": r.orig_name, "time": r.time, "hint": r.hint, "context": r.context,
                      "why": why, "label": self.store.get(r.id)} for r, why in pairs]
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


def make_server(port: int, home: Path, manifest: Path, labels: Path, batches: Path) -> ThreadingHTTPServer:
    handler = partial(Handler, home=home, manifest=manifest, store=LabelStore(labels), batches=batches)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="screen-labelling page")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    srv = make_server(args.port, paths.home(), paths.manifest_path(), paths.screen_labels_path(),
                      paths.data_dir() / "labels" / "batches")
    print(f"open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
