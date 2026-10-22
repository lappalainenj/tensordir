"""Staged builds: order, required stages, resume after a failure, optional stages after done."""

import numpy as np
import pytest

import tensordir as td


class Pipeline(td.Record):
    __record_name__ = "StagedPipeline"
    crash_at: int | None = None
    ran: list = []

    stages = td.Stages(
        prepare=td.Stage(),
        fit=td.Stage(after="prepare"),
        epoch=td.Stage(after="fit", repeat=True),
        preview=td.Stage(after="fit", required=False),
    )

    def __init__(self, /, *, n: int = 4, n_epochs: int = 3):
        self.prepare(n).fit()
        for _ in range(n_epochs):
            self.epoch()

    @stages
    def prepare(self, n):
        type(self).ran.append("prepare")
        self.data = np.arange(n, dtype=float)

    @stages
    def fit(self):
        type(self).ran.append("fit")
        self.mean = float(self.load("data").mean())  # earlier results come from fields

    @stages
    def epoch(self):
        step = len(self.load("loss")) if "loss" in self else 0
        if step == type(self).crash_at:
            type(self).crash_at = None
            raise RuntimeError(f"node lost at epoch {step}")
        type(self).ran.append(f"epoch{step}")
        self.append("loss", [1.0 / (step + 1)])

    @stages
    def preview(self, label="preview"):
        self[label] = {"mean": self.load("mean")}


@pytest.fixture(autouse=True)
def reset():
    Pipeline.crash_at, Pipeline.ran = None, []


def test_stages_chain_and_record_progress(root):
    record = Pipeline()
    assert Pipeline.in_place  # staged records build in place so they can resume
    assert Pipeline.ran == ["prepare", "fit", "epoch0", "epoch1", "epoch2"]
    progress = record.meta["stages"]
    assert progress["epoch"]["calls"] == 3 and progress["prepare"]["fields"] == ["data"]
    assert "preview" not in progress and record.status == "done"


def test_a_failed_build_resumes_after_its_finished_stages(root):
    Pipeline.crash_at = 1
    with pytest.raises(td.BuildFailed, match="node lost at epoch 1"):
        Pipeline()
    assert Pipeline.ran == ["prepare", "fit", "epoch0"]
    Pipeline.ran = []
    record = Pipeline.get(retry_failed=True)
    assert Pipeline.ran == ["epoch1", "epoch2"]  # prepare, fit and the first epoch are skipped
    np.testing.assert_allclose(record.load("loss"), [1.0, 0.5, 1 / 3])


def test_optional_stage_runs_after_done_and_adds_only(root):
    record = Pipeline()
    assert record.preview() is record  # stages return the record
    assert record.load("preview") == {"mean": 1.5}
    assert record.meta["stages"]["preview"]["fields"] == ["preview"]
    record.preview()  # once it ran, calling it again does nothing
    assert Pipeline.get().load("preview") == {"mean": 1.5}
    with pytest.raises(td.ReadOnlyError, match="may not rewrite"):
        Pipeline(n=5).preview(label="mean")  # a field another stage wrote
    assert "preview" not in Pipeline(n=5).meta["stages"]
    with pytest.raises(td.ReadOnlyError, match="required"):
        record.prepare(3)


def test_order_and_required_stages_are_checked(root):
    class Unordered(td.Record):
        __record_name__ = "StagedUnordered"
        stages = td.Stages(first=td.Stage(), second=td.Stage(after="first"))

        def __init__(self, /, *, skip: bool = False, twice: bool = False):
            if skip:
                self.first()
                return
            if twice:
                self.first().first()
            self.second()

        @stages
        def first(self):
            self.a = 1

        @stages
        def second(self):
            self.b = 2

    with pytest.raises(td.BuildFailed, match="StageOrderError.*runs after \\['first'\\]"):
        Unordered()
    with pytest.raises(td.BuildFailed, match="required stages \\['second'\\] never ran"):
        Unordered(skip=True)
    with pytest.raises(td.BuildFailed, match="ran twice"):
        Unordered(twice=True)


def test_stage_names_cannot_shadow_record_methods():
    with pytest.raises(TypeError, match="shadows Record.load"):

        class Shadow(td.Record):
            stages = td.Stages(load=td.Stage())

            @stages
            def load(self):
                pass


def test_stage_declarations_are_validated():
    with pytest.raises(ValueError, match="unknown stages"):
        td.Stages(a=td.Stage(after="b"))
    stages = td.Stages(a=td.Stage())
    with pytest.raises(ValueError, match="no stage named"):

        @stages
        def b(self):
            pass
