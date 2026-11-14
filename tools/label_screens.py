"""Screen-labelling page: one key per screen class, served locally.

    python tools/label_screens.py          then open http://127.0.0.1:8765 in a browser

Shows unlabelled full frames first, with the old bot's hint and log lines. Keys: see the legend on the page;
Left/Right move without labelling. Labels go to <ADBOT2_HOME>/data/labels/screens.jsonl (append-only, latest wins).
Only listens on 127.0.0.1. Does not open a browser by itself.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from adbot2 import paths
from adbot2.dataset import LabelStore, load_manifest
from adbot2.screens import HOTKEYS, Screen

PAGE = """<!doctype html><meta charset="utf-8"><title>Screen labels</title>
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
<div id="side"><div id="legend"></div><h4>Old bot log</h4><pre id="ctx"></pre></div>
<script>
let items=[],i=0,keys={};
async function load(){const r=await fetch('/api/items');const d=await r.json();items=d.items;keys=d.keys;
 i=Math.max(0,items.findIndex(x=>!x.label));show()}
function show(){const it=items[i];if(!it){document.getElementById('bar').textContent='nothing to label';return}
 document.getElementById('img').src='/img/'+it.file;
 const done=items.filter(x=>x.label).length;
 document.getElementById('bar').textContent=`${i+1}/${items.length} (${done} labelled)  ${it.orig_name}  label: ${it.label||'-'}  hint: ${it.hint||'-'}`;
 document.getElementById('ctx').textContent=(it.context||[]).join('\\n');
 document.getElementById('legend').innerHTML=Object.entries(keys).map(([k,s])=>
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
    def __init__(self, *a, raw_dir: Path, manifest: Path, store: LabelStore, **kw):
        self.raw_dir, self.manifest, self.store = raw_dir, manifest, store
        super().__init__(*a, **kw)

    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/api/items":
            recs = [r for r in load_manifest(self.manifest).values() if r.full_frame]
            recs.sort(key=lambda r: (r.time or "", r.id))
            items = [{"id": r.id, "file": r.file, "orig_name": r.orig_name, "hint": r.hint, "context": r.context,
                      "label": self.store.get(r.id)} for r in recs]
            keys = {k: s.value for k, s in HOTKEYS.items()}
            return self._send(200, json.dumps({"items": items, "keys": keys}).encode(), "application/json")
        if self.path.startswith("/img/"):
            name = Path(self.path[5:]).name  # no directory traversal
            f = self.raw_dir / name
            if f.is_file():
                return self._send(200, f.read_bytes(), "image/png" if name.endswith(".png") else "image/jpeg")
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


def make_server(port: int, raw_dir: Path, manifest: Path, labels: Path) -> ThreadingHTTPServer:
    handler = partial(Handler, raw_dir=raw_dir, manifest=manifest, store=LabelStore(labels))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="screen-labelling page")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    srv = make_server(args.port, paths.raw_dir(), paths.manifest_path(), paths.screen_labels_path())
    print(f"open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
