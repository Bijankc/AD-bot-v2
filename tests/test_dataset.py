from adbot2.dataset import LabelStore, import_image, load_manifest
from adbot2.frames import FileSource


def test_import_dedupes_and_flags_crops(tmp_path, make_png):
    raw, man = tmp_path / "raw", tmp_path / "manifest.jsonl"
    a = make_png("a.png", value=10)
    a_copy = make_png("a_copy.png", value=10)
    crop = make_png("c.png", size=(1320, 520), value=20)

    rec, added = import_image(a, raw, man, "test", hint="lobby")
    assert added and rec.full_frame and (raw / rec.file).exists()
    _, added = import_image(a_copy, raw, man, "test")
    assert not added  # same bytes
    rec_c, _ = import_image(crop, raw, man, "test")
    assert not rec_c.full_frame

    m = load_manifest(man)
    assert len(m) == 2 and m[rec.id].hint == "lobby"


def test_label_store_latest_wins(tmp_path):
    p = tmp_path / "labels" / "screens.jsonl"
    s = LabelStore(p)
    s.set("x", "lobby")
    s.set("x", "fight_auto")
    s.set("y", "bag")
    assert LabelStore(p).all() == {"x": "fight_auto", "y": "bag"}


def test_file_source(tmp_path, make_png):
    d = tmp_path / "frames"
    make_png("b.png", value=5, folder=d)
    make_png("a.png", value=7, folder=d)
    (d / "notes.txt").write_text("x")
    frames = list(FileSource(d))
    assert [f.image[0, 0, 0] for f in frames] == [7, 5]  # sorted by name, non-images skipped
    assert frames[0].image.shape == (1080, 1920, 3)
