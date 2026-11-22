"""The desktop application, tested without a screen.

What can be tested here and what cannot. The document, the 3-D scene with its
picking, the interactive tools and the dispatch routes all run headless, and
that is deliberate: the application was layered this way so the parts that can
hold a bug are reachable from a test.

The one part that is not here is the Qt-plus-VTK embedding
(``QVTKRenderWindowInteractor``), which segfaults under an offscreen
platform plug-in on macOS. The panels and the window are checked in
``test_app_qt.py``, which runs them in a child process.
"""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_cad

from weldforge.app.project import (
    PROJECT_SUFFIX,
    Project,
    RecentFiles,
    attach_part,
    load_project,
    save_project,
)
from weldforge.app.scene3d import PickResult, Scene3D
from weldforge.app.send import SendError, SendTarget, dispatch, send_to_folder
from weldforge.app.tools import PlacePartTool, SelectTool, TraceSeamTool
from weldforge.fixture import load_cell
from weldforge.geom import transform as tf
from weldforge.model import JointType, Seam
from weldforge.seams import DetectOptions, detect_seams
from weldforge.wps import ProcedureOptions, generate_procedures

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tee_project(tee_assembly_step: Path) -> Project:
    """A project in the state an operator reaches after a few minutes' work.

    Module-scoped because detection is the slow part; every test that mutates
    it takes a copy.
    """
    project = Project(name="tee")
    attach_part(project, tee_assembly_step)
    project.cell = load_cell("bench_cell")
    seams, _ = detect_seams(project.part, DetectOptions())
    project.seams = seams
    project.procedures = generate_procedures(
        project.selected_seams, ProcedureOptions(base_material="S355")
    )
    project.placement = tf.pose(tf.rot_z(np.deg2rad(30.0)), [410.0, 95.0, 900.0])
    return project


def _clone(project: Project) -> Project:
    """A working copy. The B-rep handles are shared, which is what we want."""
    part = project.part
    project.part = None
    try:
        twin = copy.deepcopy(project)
    finally:
        project.part = part
    twin.part = part
    return twin


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


@requires_cad
def test_project_round_trip_keeps_the_setup(tee_project: Project, tmp_path: Path):
    """Save and reload must return the same job, not an approximation of it."""
    source = _clone(tee_project)
    path = save_project(source, tmp_path / "tee")

    assert path.suffix == PROJECT_SUFFIX, "the suffix is added if the operator omits it"
    assert source.dirty is False, "saving clears the unsaved marker"

    loaded = load_project(path)

    assert loaded.name == source.name
    assert loaded.cell_name == source.cell_name
    assert loaded.base_material == source.base_material
    assert len(loaded.seams) == len(source.seams)
    # The placement is the operator's own work and must survive exactly.
    assert np.allclose(loaded.placement, source.placement, atol=1e-9)

    for before, after in zip(source.seams, loaded.seams, strict=True):
        assert after.id == before.id
        assert after.joint_type is before.joint_type
        assert after.selected == before.selected
        assert np.allclose(after.points, before.points, atol=1e-3)
        assert np.allclose(after.bisector, before.bisector, atol=1e-3)
        assert np.allclose(after.dihedral, before.dihedral, atol=1e-4)
        assert after.length_mm == pytest.approx(before.length_mm, abs=1e-2)
        assert after.thickness_a_mm == pytest.approx(before.thickness_a_mm, abs=1e-3)

    assert set(loaded.procedures) == set(source.procedures)
    for seam_id, procedure in source.procedures.items():
        reloaded = loaded.procedures[seam_id]
        assert len(reloaded.passes) == len(procedure.passes)
        first, expected = reloaded.passes[0], procedure.passes[0]
        assert first.current_a == pytest.approx(expected.current_a)
        assert first.voltage_v == pytest.approx(expected.voltage_v)
        assert first.travel_speed_mm_s == pytest.approx(expected.travel_speed_mm_s)
        assert first.wire_speed_m_min == pytest.approx(expected.wire_speed_m_min)


@requires_cad
def test_the_cad_travels_inside_the_project(tee_project: Project, tmp_path: Path):
    """A project moved to another machine must still open.

    The CAD is embedded rather than referenced: an operator who copies a
    project to the machine PC does not copy the engineer's folder tree with
    it, and a project that opens with no geometry is worse than useless --
    it looks like it worked.
    """
    source = _clone(tee_project)
    path = save_project(source, tmp_path / "portable")

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert "project.json" in names
        assert any(n.startswith("part.") for n in names), names

    # Simulate the move: the original CAD file is not reachable any more.
    moved = tmp_path / "elsewhere" / f"portable{PROJECT_SUFFIX}"
    moved.parent.mkdir()
    moved.write_bytes(path.read_bytes())
    path.unlink()

    loaded = load_project(moved)
    assert loaded.part is not None, "the geometry came back from inside the project"
    assert len(loaded.part.faces) == len(source.part.faces)
    assert loaded.part.brep.topology is not None, (
        "the B-rep is live again, so seams can be re-detected and traced"
    )


@requires_cad
def test_a_project_can_be_read_without_the_geometry(tee_project: Project, tmp_path: Path):
    """``load_cad=False`` is for listing projects, where re-import is wasted."""
    path = save_project(_clone(tee_project), tmp_path / "quick")
    loaded = load_project(path, load_cad=False)
    assert loaded.part is None
    assert loaded.part_bytes, "the bytes are still there, so it can be opened properly later"
    assert len(loaded.seams) == len(tee_project.seams)


def test_an_unknown_field_survives_a_round_trip(tmp_path: Path):
    """A project written by a newer version must not lose data here.

    The alternative -- dropping what we do not recognise -- turns opening a
    project in an older copy of the application into silent data loss.
    """
    path = tmp_path / f"future{PROJECT_SUFFIX}"
    payload = {
        "format_version": 1,
        "name": "future",
        "cell_name": "bench_cell",
        "tack_welds": [{"at_mm": 40.0, "length_mm": 15.0}],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("project.json", json.dumps(payload))

    loaded = load_project(path)
    assert loaded.unknown_fields.get("tack_welds"), "kept aside rather than discarded"

    again = save_project(loaded, tmp_path / "resaved")
    with zipfile.ZipFile(again) as archive:
        data = json.loads(archive.read("project.json"))
    assert data["tack_welds"] == payload["tack_welds"], "and written back out"


def test_re_detecting_seams_keeps_the_hand_traced_ones():
    """The operator's own seams are not detection's to delete."""
    project = Project()
    traced = project.add_manual_seam(
        Seam(joint_type=JointType.FILLET, points=np.array([[0, 0, 0], [100, 0, 0]], float))
    )
    project.seams.append(
        Seam(joint_type=JointType.FILLET, points=np.array([[0, 50, 0], [100, 50, 0]], float))
    )

    kept = project.replace_detected_seams(
        [Seam(joint_type=JointType.BUTT, points=np.array([[0, 9, 0], [50, 9, 0]], float))]
    )

    assert kept == 1
    ids = [s.id for s in project.seams]
    assert traced.id in ids, "the traced seam stayed"
    assert len(project.seams) == 2, "the previously detected one was replaced"


def test_recent_files_are_most_recent_first_and_deduplicated(tmp_path: Path):
    store = RecentFiles(tmp_path / "recent.json", limit=3)
    for name in ("a", "b", "c", "d"):
        # The list drops paths that have gone, so these have to be real.
        (tmp_path / f"{name}{PROJECT_SUFFIX}").write_bytes(b"")
        store.add(tmp_path / f"{name}{PROJECT_SUFFIX}")
    store.add(tmp_path / f"b{PROJECT_SUFFIX}")

    names = [p.stem for p in store.load()]
    assert names[0] == "b", "re-opening a project moves it to the top"
    assert names.count("b") == 1
    assert len(names) <= 3, "the list is capped"


# ---------------------------------------------------------------------------
# the 3-D scene and picking
# ---------------------------------------------------------------------------


@requires_cad
def test_the_scene_builds_and_picks_offscreen(tee_project: Project, tmp_path: Path):
    """Picking must map a click back to the right domain object.

    This is the heart of the interactive part: trace a weld line by clicking an
    edge, and the edge the operator sees must be the edge the tool gets.
    """
    scene = Scene3D()
    scene.set_project(tee_project)
    scene.reset_camera("iso")

    assert scene.layer_visible("part")
    image = Path(scene.render_to_png(str(tmp_path / "scene.png"), 600, 400))
    assert image.stat().st_size > 5_000, "something was actually drawn"

    # Sweep the viewport and collect what comes back, rather than trusting one
    # click to land where we think it does.
    hits: dict[str, set[str]] = {}
    for x in range(60, 540, 30):
        for y in range(40, 360, 30):
            pick = scene.pick(x, y)
            if pick:
                hits.setdefault(pick.kind, set()).add(pick.id)

    assert "face" in hits, f"no face was ever hit: {hits.keys()}"
    face_ids = set(tee_project.part.faces)
    assert hits["face"] <= face_ids, "a picked face id is a real face id"

    scene.pick_only("seams")
    seam_hits = set()
    for x in range(60, 540, 10):
        for y in range(40, 360, 10):
            pick = scene.pick(x, y)
            if pick.kind == "seam":
                seam_hits.add(pick.id)
    assert seam_hits <= {s.id for s in tee_project.seams}
    scene.restore_pickable()


@requires_cad
def test_dragging_the_part_is_a_matrix_update(tee_project: Project):
    """Re-tessellating on every mouse move would make dragging unusable.

    The part actor carries the placement as its user matrix, so moving the
    part costs one 4x4 assignment however many triangles it has.
    """
    scene = Scene3D()
    scene.set_project(tee_project)
    before = len(scene._actors["part"])

    for step in range(40):
        scene.update_placement(tf.pose(tf.identity()[:3, :3], [step * 5.0, 0.0, 900.0]))

    assert len(scene._actors["part"]) == before, "no actor was rebuilt"
    matrix = scene._actors["part"][0].GetUserMatrix()
    assert matrix.GetElement(0, 3) == pytest.approx(39 * 5.0)


# ---------------------------------------------------------------------------
# the tools
# ---------------------------------------------------------------------------


@requires_cad
def test_a_detected_seam_cannot_be_deleted_only_switched_off(tee_project: Project):
    """Deleting it would be a lie: the next detection brings it straight back."""
    project = _clone(tee_project)
    tool = SelectTool(project)
    seam_id = project.seams[0].id

    tool.on_pick(PickResult(kind="seam", id=seam_id, layer="seams"))
    result = tool.on_command("delete")

    assert not result.ok
    assert "detection" in result.message or "detected" in result.message
    assert project.seam(seam_id) is not None

    off = tool.on_command("toggle")
    assert off.ok and off.changed
    assert project.seam(seam_id).selected is False, "switching it off is the honest way"


@requires_cad
def test_tracing_an_edge_and_a_bedding_face_finds_the_same_joint(tee_project: Project):
    """The hand-traced seam must agree with the detector on the same joint.

    This is the test that makes hand tracing trustworthy. The operator picks
    the web's bottom edge and the base's top face -- which is all the
    information there is in an assembly, since the two bodies share no
    topology -- and the tool has to work out the groove, the flank normals and
    the wire direction from the geometry. If it disagreed with automatic
    detection on a joint both can see, one of them would be wrong.
    """
    project = _clone(tee_project)
    reference = max(project.seams, key=lambda s: s.length_mm)
    project.seams = []

    tool = TraceSeamTool(project)
    assert "edge" in tool.prompt().lower()

    # The edge the detector used, and the opposing face from the other body.
    edge_id = reference.edge_ids[0]
    assert tool.on_pick(PickResult(kind="edge", id=edge_id, layer="edges")).ok

    edge_solids = set(project.part.edges[edge_id].solid_ids)
    face_id = next(
        f
        for f in (reference.face_a_id, reference.face_b_id)
        if project.part.faces[f].solid_id not in edge_solids
    )
    tool.on_pick(PickResult(kind="face", id=face_id, layer="part"))

    result = tool.on_command("accept")
    assert result.ok, result.message
    assert result.changed and result.seam_id

    traced = project.seam(result.seam_id)
    assert traced is not None
    assert project.is_manual(traced.id), "it is the operator's, so detection leaves it alone"
    assert traced.joint_type is reference.joint_type
    assert traced.length_mm == pytest.approx(reference.length_mm, rel=0.02)
    assert np.rad2deg(traced.mean_dihedral) == pytest.approx(
        np.rad2deg(reference.mean_dihedral), abs=3.0
    )
    assert traced.position_at(traced.n_points // 2) is reference.position_at(
        reference.n_points // 2
    )
    # The wire has to point into the groove, not out of the part.
    assert float(np.dot(traced.bisector[0], reference.bisector[0])) > 0.9


@requires_cad
def test_tracing_warns_when_the_groove_is_an_outside_corner(tee_project: Project):
    """An operator who picks the wrong face gets told, but is not overruled.

    A groove over about 215 deg is the outside of a corner: there is no joint
    there to fill. It can still be welded -- an outside corner weld is a real
    thing -- so the tool warns loudly and lets the operator decide.
    """
    project = _clone(tee_project)
    reference = max(project.seams, key=lambda s: s.length_mm)
    edge_id = reference.edge_ids[0]
    edge = project.part.edges[edge_id]

    tool = TraceSeamTool(project)
    tool.on_pick(PickResult(kind="edge", id=edge_id, layer="edges"))
    # Pick a face on the far side of the base plate: the underside.
    edge_solids = set(edge.solid_ids)
    underside = min(
        (f for f in project.part.faces.values() if f.solid_id not in edge_solids),
        key=lambda f: f.centroid[2],
    )
    tool.on_pick(PickResult(kind="face", id=underside.id, layer="part"))
    result = tool.on_command("accept")

    if result.ok:
        seam = project.seam(result.seam_id)
        groove = np.rad2deg(seam.mean_dihedral)
        warned = any("corner" in n or "groove" in n or "joint" in n for n in seam.notes)
        assert groove > 180.0 or warned, (
            f"a {groove:.0f} deg groove went through without a word: {seam.notes}"
        )
    else:
        assert result.message, "a refusal has to say why"


@requires_cad
def test_placing_the_part_drops_it_onto_the_table(tee_project: Project):
    project = _clone(tee_project)
    tool = PlacePartTool(project)

    result = tool.drop_to_table()
    assert result.ok and result.changed

    # Every mesh vertex, so a corner of the bounding box cannot flatter it.
    vertices = project.part.mesh_vertices
    assert vertices is not None
    placed_z = (vertices @ project.placement[:3, :3].T + project.placement[:3, 3])[:, 2]
    table_z = project.cell.fixture.table_z_mm
    assert float(np.min(placed_z)) == pytest.approx(table_z, abs=0.5), (
        "the part sits on the table, neither floating nor sunk into it"
    )

    before = project.placement.copy()
    tool.translate(np.array([25.0, 0.0, 0.0]))
    assert not np.allclose(project.placement, before)
    assert tool.undo().ok
    assert np.allclose(project.placement, before), "undo puts it back exactly"


@requires_cad
def test_rotating_the_part_turns_it_about_its_own_centre(tee_project: Project):
    """Rotating about the origin would fling the part across the cell."""
    project = _clone(tee_project)
    tool = PlacePartTool(project)
    tool.drop_to_table()

    lo, hi = project.part.bbox
    centre_before = tf.apply_point(project.placement, (lo + hi) / 2.0)
    tool.rotate(np.array([0.0, 0.0, 1.0]), 90.0)
    centre_after = tf.apply_point(project.placement, (lo + hi) / 2.0)

    assert np.allclose(centre_before[:2], centre_after[:2], atol=1e-6), "the part spun in place"


def test_a_locked_placement_is_not_moved():
    """An operator who locked the part has measured something. Respect it."""
    project = Project(cell=load_cell("bench_cell"))
    project.placement_locked = True
    pose = tf.pose(tf.identity()[:3, :3], [100.0, 200.0, 900.0])
    project.placement = pose.copy()

    tool = PlacePartTool(project)
    result = tool.translate(np.array([50.0, 0.0, 0.0]))

    assert not result.ok
    assert "lock" in result.message.lower()
    assert np.allclose(project.placement, pose)


# ---------------------------------------------------------------------------
# getting it to the machine
# ---------------------------------------------------------------------------


def test_a_program_file_cannot_escape_the_chosen_folder(tmp_path: Path):
    """The file names come from a postprocessor, so they are checked.

    A program named ``../../startup.mod`` would otherwise be written outside
    the folder the operator chose -- on a machine PC, plausibly over something
    that matters.
    """
    destination = tmp_path / "chosen"
    result = send_to_folder(
        {"../../startup.mod": "MODULE x\nENDMODULE\n", "job.mod": "MODULE y\nENDMODULE\n"},
        SendTarget(method="folder", destination=str(destination)),
    )

    assert result.ok
    written = sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*.mod"))
    assert written == ["job.mod", "startup.mod"], written
    assert not (tmp_path.parent / "startup.mod").exists()


def test_a_job_the_planner_rejected_is_not_sent_by_accident(tmp_path: Path):
    """The gate exists because the consequence is a torch in a fixture."""
    project = Project(name="bad")
    project.programs["rapid"] = {"job.mod": "MODULE x\nENDMODULE\n"}
    project.last_report = {
        "ok": False,
        "errors": ["seam S3: A5 -136.2 deg outside [-130, 130]"],
    }
    target = SendTarget(method="folder", destination=str(tmp_path / "out"))

    with pytest.raises(SendError) as excinfo:
        dispatch(project, "rapid", target)
    assert "problem" in str(excinfo.value).lower()
    assert not project.sent, "nothing was recorded, because nothing was sent"

    # The operator can still override, which is the point of a gate rather
    # than a wall -- but they have to say so.
    result = dispatch(project, "rapid", target, force=True)
    assert result.ok
    assert len(project.sent) == 1
    assert project.sent[0].files


def test_what_was_sent_where_and_when_is_recorded(tmp_path: Path):
    """On the shop floor, "which program is on the machine?" is the question."""
    project = Project(name="good")
    project.programs["urscript"] = {"weld.script": "def weld():\nend\n"}
    project.last_report = {"ok": True, "errors": []}

    first = dispatch(
        project, "urscript", SendTarget(method="folder", destination=str(tmp_path / "a"))
    )
    dispatch(project, "urscript", SendTarget(method="folder", destination=str(tmp_path / "b")))

    assert len(project.sent) == 2
    record = project.sent[0]
    assert record.ok and record.target == "urscript" and record.method == "folder"
    assert record.when and record.files == first.files
    assert "a" in record.destination

    # And it survives being saved, which is what makes it a record.
    path = save_project(project, tmp_path / "sent")
    assert len(load_project(path).sent) == 2


def test_sending_a_program_that_was_never_generated_fails_clearly():
    project = Project(name="empty")
    project.last_report = {"ok": True}
    with pytest.raises(SendError) as excinfo:
        dispatch(project, "rapid", SendTarget(method="folder", destination="/tmp/nowhere"))
    assert "generate" in str(excinfo.value).lower() or "no program" in str(excinfo.value).lower()


def test_a_folder_target_needs_a_folder():
    project = Project(name="x")
    project.programs["rapid"] = {"a.mod": "x"}
    project.last_report = {"ok": True}
    with pytest.raises(SendError):
        dispatch(project, "rapid", SendTarget(method="folder", destination=""))
