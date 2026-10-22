"""Directories: views of any folder, typed trees with metadata, navigation, new/move/archive."""

import json
import os
import subprocess
import sys
from typing import Literal

import numpy as np
import pytest

import tensordir as td


class Dev(td.Directory):
    __type_name__ = "TestDev"

    class Track(td.Directory):
        __type_name__ = "TestTrack"
        title: str = ""

        class Study(td.Directory):
            __type_name__ = "TestStudy"
            title: str = ""

            class Version(td.Directory):
                __type_name__ = "TestVersion"
                names = "v{n}"
                title: str = ""
                status: Literal["active", "superseded", "private"] = "active"
                builds_on: list[td.NodeId] = []


@pytest.fixture
def dev(tmp_path):
    dev = Dev(tmp_path / "dev")
    for track, studies in {"toy-model": ["inverse", "strf"], "baseline-model": ["nulls"]}.items():
        t = dev.track.new(track, title=track.replace("-", " "))
        for study in studies:
            s = t.study.new(study)
            for _ in range(2):
                s.version.new(title=f"{study} design")
    return dev


def test_a_plain_folder_is_viewed_without_writing(tmp_path):
    folder = tmp_path / "downloads"
    (folder / "sub").mkdir(parents=True)
    (folder / "notes.txt").write_text("hello")
    (folder / "table.json").write_text(json.dumps({"a": 1}))
    view = td.Directory(folder)
    assert view.keys() == ["notes.txt", "sub", "table.json"]
    assert view.load("notes.txt") == "hello" and view["table.json"] == {"a": 1}
    assert isinstance(view["sub"], td.Directory) and "notes.txt" in view
    view.tree()
    assert sorted(p.name for p in folder.iterdir()) == ["notes.txt", "sub", "table.json"]  # nothing written
    with pytest.raises(TypeError, match="no metadata fields"):
        view.update(title="x")


def test_opening_writes_nothing_until_metadata_is_set(tmp_path):
    dev = Dev(tmp_path / "dev")
    assert not dev.exists and dev.track.keys() == []
    assert not (tmp_path / "dev").exists()


def test_navigation_and_listing(dev):
    assert dev.track.keys() == ["baseline-model", "toy-model"]
    assert dev.track("toy-model").study.keys() == ["inverse", "strf"]
    v = dev.track("toy-model").study("inverse").version("v2")
    assert isinstance(v, Dev.Track.Study.Version) and v.id == "toy-model/inverse/v2"
    assert v.title == "inverse design" and v.status == "active" and v.builds_on == []
    assert dev.track(1).study("inverse").version(-1) == v  # positions in keys()
    assert dev.track.study.version.keys() == [
        "baseline-model/nulls/v1", "baseline-model/nulls/v2",
        "toy-model/inverse/v1", "toy-model/inverse/v2",
        "toy-model/strf/v1", "toy-model/strf/v2",
    ]
    assert dev.track.study.keys() == ["baseline-model/nulls", "toy-model/inverse", "toy-model/strf"]
    assert dev.track.study("toy-model/strf").name == "strf"
    with pytest.raises(KeyError, match="available"):
        dev.track("missing")
    assert [s.name for s in dev.track("toy-model").study] == ["inverse", "strf"]


def test_metadata_is_typed_validated_and_tracked_as_yaml(dev):
    study = dev.track("toy-model").study("inverse")
    v = study.version.new(title="rebuilt on release 2", builds_on=["toy-model/inverse/v1"])
    assert v.name == "v3" and v.builds_on == ["toy-model/inverse/v1"]
    text = (v.path / "_meta.yaml").read_text()
    assert text.splitlines()[0] == "type: TestVersion" and "title: rebuilt on release 2" in text
    assert "time" not in text and "host" not in text  # deterministic: no volatile fields
    v.status = "superseded"
    assert Dev(dev.path).track("toy-model").study("inverse").version("v3").status == "superseded"
    with pytest.raises(ValueError, match="not one of"):
        v.status = "done"
    with pytest.raises(KeyError, match="no directory"):
        v.builds_on = ["toy-model/inverse/v9"]
    with pytest.raises(TypeError):
        v.title = 3
    with pytest.raises(AttributeError, match="no metadata field"):
        v.update(colour="red")
    assert v.update(title="new title") is v


def test_new_picks_the_next_number_and_never_reuses_one(dev):
    study = dev.track("toy-model").study("strf")
    study.version("v2").archive()
    assert study.version.keys() == ["v1"] and study.version.keys(archived=True) == ["v1", "v2"]
    assert study.version.new().name == "v3"  # v2 is archived, not free
    study.version("v1").aliases = ["toy-model/strf/v7"]  # an alias reserves its name
    assert study.version.new().name == "v8"
    assert study.version.new(name="v20").name == "v20"
    with pytest.raises(FileExistsError):
        study.version.new(name="v3")
    with pytest.raises(TypeError, match="one parent"):
        dev.track.study.version.new()
    with pytest.raises(ValueError, match="invalid directory name"):
        study.version.new(name="archive")


def test_archive_keeps_the_id_and_resolve_finds_it(dev):
    v = dev.track("toy-model").study("inverse").version("v1").archive()
    assert v.path.parent.name == "archive" and v.id == "toy-model/inverse/v1" and v.archived
    assert dev.resolve("toy-model/inverse/v1") == v
    assert dev.track("toy-model").study("inverse").version("v1") == v  # archived ones are named
    assert "toy-model/inverse/v1" not in dev.track.study.version.keys()


def test_move_adds_the_old_id_as_an_alias(dev):
    dev.track("toy-model").study.new("ablations")
    v = dev.resolve("toy-model/inverse/v2")
    v.move("toy-model/ablations/v1")
    assert v.id == "toy-model/ablations/v1" and v.aliases == ["toy-model/inverse/v2"]
    assert dev.resolve("toy-model/inverse/v2") == v  # the old ID still resolves
    assert dev.track("toy-model").resolve("ablations/v1") == v
    assert dev.track("toy-model").study("inverse").version.new().name == "v3"  # v2 stays taken
    with pytest.raises(KeyError):
        v.move("toy-model/missing-study/v1")


def test_resolve_legacy_aliases(dev):
    v = dev.resolve("toy-model/inverse/v1")
    v.aliases = ["v48"]
    assert dev.resolve("v48") == v
    dev.resolve("toy-model/strf/v1").aliases = ["v48"]
    with pytest.raises(KeyError, match="several"):
        dev.resolve("v48")
    with pytest.raises(KeyError):
        dev.resolve("v99")


def test_files_are_saved_and_loaded_by_suffix(dev):
    from PIL import Image as PILImage

    v = dev.resolve("toy-model/inverse/v1")
    v.save("results/figures/bar.png", PILImage.new("RGB", (4, 3), "red"))
    v.save("results/table.csv", __import__("pandas").DataFrame({"x": [1, 2]}))
    v.save("results/weights.npy", np.arange(3))
    image = v.load("results/figures/bar.png")
    assert isinstance(image, td.Image) and image.to_pil().size == (4, 3)
    assert "<img" in image._repr_html_() and image._repr_png_()[:4] == b"\x89PNG"
    assert v.load("results/table.csv")["x"].tolist() == [1, 2]
    np.testing.assert_array_equal(v.load("results/weights.npy"), [0, 1, 2])
    v.save("movie.webm", b"\x1a\x45\xdf\xa3")
    assert isinstance(v.load("movie.webm"), td.Video) and "<video" in v.load("movie.webm")._repr_html_()

    class Figure:  # duck-typed like matplotlib's
        def savefig(self, path, format, bbox_inches):
            PILImage.new("L", (2, 2)).save(path, format=format.upper())

    assert v.save("fig.png", Figure()).exists()
    with pytest.raises(ValueError, match="no reader"):
        v.load("data.unknown-suffix")


def test_records_store_their_producer(dev, root):
    class Fit(td.Record):
        __record_name__ = "DirectoryFit"

        def __init__(self, /, *, seed: int = 0):
            self.x = np.arange(3) + seed

    v = dev.resolve("toy-model/inverse/v1")
    with v.active():
        fit = Fit()
        script = "import os, json; print(os.environ['TENSORDIR_PRODUCER'])"
        child = json.loads(subprocess.run([sys.executable, "-c", script], capture_output=True, text=True).stdout)
    assert fit.meta["provenance"]["producer"] == {"id": "toy-model/inverse/v1", "path": str(v.path)}
    assert child["id"] == "toy-model/inverse/v1"  # subprocesses inherit it
    assert "TENSORDIR_PRODUCER" not in os.environ
    assert Fit(seed=1).meta["provenance"]["producer"] is None


def test_open_returns_the_registered_type(dev):
    path = dev.resolve("toy-model/strf/v2").path
    opened = td.Directory(path)
    assert type(opened) is Dev.Track.Study.Version and opened.title == "strf design"
    with pytest.raises(TypeError, match="holds a TestVersion"):
        Dev.Track(path)


def test_directory_types_reject_bad_declarations():
    with pytest.raises(TypeError, match="need a default"):

        class NoDefault(td.Directory):
            title: str

    with pytest.raises(TypeError, match="reserved"):

        class Reserved(td.Directory):
            aliases: list = []
