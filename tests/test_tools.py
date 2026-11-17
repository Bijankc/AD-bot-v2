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
    from adbot2.dataset import import_image
    raw, man, labels = tmp_path / "raw", tmp_path / "m.jsonl", tmp_path / "l.jsonl"
    import_image(make_png("a.png", value=3), raw, man, "t")
    import_image(make_png("crop.png", size=(100, 50), value=4), raw, man, "t")
    srv = label_screens.make_server(0, raw, man, labels)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", labels
    srv.shutdown()


def test_label_server_roundtrip(server):
    url, labels = server
    items = json.load(urllib.request.urlopen(url + "/api/items"))["items"]
    assert len(items) == 1  # crops are not offered for screen labels
    img = urllib.request.urlopen(url + "/img/" + items[0]["file"]).read()
    assert img[:4] == b"\x89PNG"
    req = urllib.request.Request(url + "/api/label", data=json.dumps({"id": items[0]["id"], "screen": "lobby"}).encode(),
                                 method="POST")
    urllib.request.urlopen(req)
    assert LabelStore(labels).get(items[0]["id"]) == "lobby"


def test_label_server_rejects_unknown_class_and_traversal(server):
    url, _ = server
    req = urllib.request.Request(url + "/api/label", data=b'{"id": "x", "screen": "nope"}', method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(url + "/img/..%5C..%5Cm.jsonl")


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
