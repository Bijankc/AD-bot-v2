import json
import threading
import urllib.request

import pytest

import import_old_shots
import label_screens
from adbot2 import paths
from adbot2.dataset import LabelStore, load_manifest
from adbot2.screens import HOTKEYS, Screen


def test_every_screen_has_one_hotkey():
    assert sorted(s.value for s in HOTKEYS.values()) == sorted(s.value for s in Screen)


def test_import_old_shots(tmp_path, make_png, monkeypatch):
    monkeypatch.setenv("ADBOT2_HOME", str(tmp_path / "home"))
    bot = tmp_path / "bot"
    make_png("20261114_095321_raid_end.png", value=1, folder=bot / "shots")
    make_png("20261114_095349_restart.png", value=2, folder=bot / "shots")
    (bot / "logs").mkdir()
    (bot / "logs" / "raid_20261114.log").write_text(
        "14.11 09:53:20.000  raid: Auto, waiting for the end\n"
        "14.11 09:53:21.000    screenshot: C:\\x\\shots\\20261114_095321_raid_end.png\n", encoding="utf-8")

    assert import_old_shots.main(["--bot-dir", str(bot)]) == 0
    assert import_old_shots.main(["--bot-dir", str(bot)]) == 0  # second run adds nothing
    m = {r.reason: r for r in load_manifest(paths.manifest_path()).values()}
    assert len(m) == 2
    assert m["raid_end"].hint == "results_raid"
    assert m["raid_end"].context == ["raid: Auto, waiting for the end"]
    assert m["restart"].hint is None


@pytest.fixture
def server(tmp_path, make_png):
    from dataclasses import asdict

    from adbot2.dataset import FrameRecord, append_jsonl, import_image
    home = tmp_path / "home"
    raw, man, labels, batches = home / "data" / "raw", tmp_path / "m.jsonl", tmp_path / "l.jsonl", tmp_path / "b"
    a, _ = import_image(make_png("a.png", value=3), raw, man, "t")
    import_image(make_png("crop.png", size=(100, 50), value=4), raw, man, "t")
    make_png("090000_000.jpg", value=9, folder=home / "captures" / "20261117")
    cap = FrameRecord("cap20261117_090000_000", "20261117/090000_000.jpg", "capture", "090000_000.jpg", 1920, 1080,
                      True, "2026-11-17T09:00:00.000", "change", "lobby", ["09:00:00 pressing PLAY"], "captures")
    append_jsonl(man, [asdict(cap)])
    srv = label_screens.make_server(0, home, man, labels, batches)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", labels, batches, a.id, cap.id
    srv.shutdown()


def get_json(url):
    return json.load(urllib.request.urlopen(url))


def test_label_server_roundtrip(server):
    url, labels, _, raw_id, cap_id = server
    d = get_json(url + "/api/items")
    assert d["batch"] == "all" and [x["id"] for x in d["items"]] == [raw_id, cap_id]  # no crops
    assert urllib.request.urlopen(url + "/img/" + raw_id).read()[:4] == b"\x89PNG"
    assert len(urllib.request.urlopen(url + "/img/" + cap_id).read()) > 100  # served from captures/
    req = urllib.request.Request(url + "/api/label", data=json.dumps({"id": cap_id, "screen": "lobby"}).encode(),
                                 method="POST")
    urllib.request.urlopen(req)
    assert LabelStore(labels).get(cap_id) == "lobby"
    assert "lobby              1" in urllib.request.urlopen(url + "/api/counts").read().decode()


def test_crop_grid_roundtrip(server, tmp_path):
    import numpy as np

    from adbot2.crops import HEADS, CropCache, CropLabels

    url, labels, _, raw_id, cap_id = server
    home = tmp_path / "home"
    crops = {k: np.zeros((2, h.store_wh[1], h.store_wh[0], 3), np.uint8) for k, h in HEADS.items()}
    CropCache(home / "cache" / "crops").add("c1", [raw_id, cap_id], crops, [])
    d = get_json(url + "/api/crop_page?head=auto")
    assert d["classes"] == ["on", "off", "gone"] and {x["id"] for x in d["items"]} == {raw_id, cap_id}
    assert {x["id"]: x["value"] for x in d["items"]}[cap_id] == "gone"  # hinted lobby: no AUTO
    assert urllib.request.urlopen(url + f"/crop/auto/{cap_id}").read()[:4] == b"\x89PNG"
    body = json.dumps({"head": "auto", "values": {cap_id: "on", raw_id: "bogus"}}).encode()
    urllib.request.urlopen(urllib.request.Request(url + "/api/crop_labels", data=body, method="POST"))
    assert CropLabels(labels.parent / "crops.jsonl").of_head("auto") == {cap_id: "on"}
    assert [x["id"] for x in get_json(url + "/api/crop_page?head=auto")["items"]] == [raw_id]


def test_label_server_serves_the_oldest_unfinished_batch(server):
    import os

    url, labels, batches, raw_id, cap_id = server
    batches.mkdir()
    old, new = batches / "20261117_0900.json", batches / "regression_20261117_1000.json"
    old.write_text(json.dumps({"name": "20261117_0900", "items": [{"id": raw_id}]}))
    new.write_text(json.dumps({"name": "regression_20261117_1000", "items": [{"id": cap_id, "why": "r"}, {"id": "x"}]}))
    os.utime(old, (1, 1))  # made first
    assert get_json(url + "/api/items")["batch"] == "20261117_0900"
    req = urllib.request.Request(url + "/api/label", data=json.dumps({"id": raw_id, "screen": "bag"}).encode(),
                                 method="POST")
    urllib.request.urlopen(req)
    d = get_json(url + "/api/items")  # the first one is done: on to the next
    assert d["batch"] == "regression_20261117_1000" and [(x["id"], x["why"]) for x in d["items"]] == [(cap_id, "r")]
    assert get_json(url + "/api/items?batch=20261117_0900")["items"][0]["label"] == "bag"


def test_label_server_rejects_unknown_class_and_traversal(server):
    url = server[0]
    req = urllib.request.Request(url + "/api/label", data=b'{"id": "x", "screen": "nope"}', method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
    for bad in ("/img/..%5C..%5Cm.jsonl", "/img/../m.jsonl", "/img/nope"):
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + bad)


def test_frame_logger_index_and_pid(tmp_path):
    import os
    import frame_logger as fl

    day = tmp_path / "20261117"
    day.mkdir()
    (day / "index.csv").write_text("time,file,left,top,width,height,dpi,change\nx,y,0,0,1,1,96,1\n")
    fl.append_index(day, ["t", "a.jpg", 0, 23, 1920, 1017, 96, "7.0", "change"])
    fl.append_index(day, ["t", "b.png", 0, 23, 1920, 1017, 96, "0.0", "png"])
    assert (day / "index.v1.csv").exists()  # the old layout was moved aside, not mixed
    lines = (day / "index.csv").read_text().splitlines()
    assert lines[0] == ",".join(fl.INDEX_HEADER) and len(lines) == 3

    pid = tmp_path / "logger.pid"
    assert fl.other_logger_running(pid) == 0  # no file
    pid.write_text(str(os.getpid()))
    assert fl.other_logger_running(pid) == 0  # that is us
    pid.write_text("999999999")
    assert fl.other_logger_running(pid) == 0  # no such process


def test_review_batch_counts_only_new_labels(server):
    url, labels, batches, raw_id, cap_id = server
    req = urllib.request.Request(url + "/api/label", data=json.dumps({"id": raw_id, "screen": "other"}).encode(),
                                 method="POST")
    urllib.request.urlopen(req)
    batches.mkdir()
    (batches / "review_x.json").write_text(json.dumps({"name": "review_x", "made": "2999-01-01T00:00:00",
                                                       "relabel": True, "items": [{"id": raw_id}]}))
    d = get_json(url + "/api/items")
    assert d["batch"] == "review_x" and d["items"][0]["label"] is None and d["items"][0]["old"] == "other"


def test_shadow_live_row_is_json():
    import json

    import numpy as np

    import shadow_live
    from adbot2.perception import Observation
    from adbot2.shadowing import LiveShadow

    sh = LiveShadow(["fight", "lobby"], opens=2700)
    o = Observation(screen="lobby", screen_p=0.98, screen_probs=np.array([0.02, 0.98]), ms={"screen": 12.3})
    f, a, _ = sh.step(o, 1_800_000_000.0)
    r = json.loads(json.dumps(shadow_live.row(1_800_000_000.0, o, f, sh, a)))
    assert r["raw"] == "lobby" and r["action"] == a.name and r["opens"] == 2700
