"""Editing the cell, importing a bench, and getting both back from disk.

The requirement these cover: an operator measures their bench once and never
describes it again. Three things have to hold.

* The bench numbers are editable, and changing them changes the cell -- not
  just a label.
* A bench imported as a 3-D model is positioned by those same numbers, because
  an operator who moves the bench expects the bench to move.
* A cell survives being saved, both into a project and into the library, with
  the bench model it was given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weldforge.app.project import (
    Project,
    attach_bench,
    cell_for,
    clear_bench,
    load_project,
    save_project,
)
from weldforge.fixture import (
    available_cells,
    bench_model_from_file,
    cell_from_config,
    default_cell_config,
    read_cell_config,
    save_cell_config,
)
from weldforge.geom import transform as tf

pytest.importorskip("trimesh", reason="building the bench specimen needs trimesh")

BENCH_LENGTH = 1600.0
BENCH_WIDTH = 900.0
BENCH_TOP = 800.0


@pytest.fixture(scope="session")
def bench_stl(tmp_path_factory) -> Path:
    """A bench with legs: a top slab at 800 mm on four 740 mm legs.

    Built rather than committed, for the same reason the weld specimens are:
    the suite must not depend on a file in someone's home directory.
    """
    import trimesh

    def box(extents, centre):
        return trimesh.creation.box(
            extents=extents, transform=trimesh.transformations.translation_matrix(centre)
        )

    parts = [box([BENCH_LENGTH, BENCH_WIDTH, 60.0], [0.0, 0.0, BENCH_TOP - 30.0])]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(box([80.0, 80.0, 740.0], [sx * 720.0, sy * 370.0, 370.0]))
    out = tmp_path_factory.mktemp("bench") / "bench.stl"
    trimesh.util.concatenate(parts).export(str(out))
    return out


# ---------------------------------------------------------------------------
# the numbers
# ---------------------------------------------------------------------------


def test_the_bench_numbers_are_editable():
    """Changing the description changes the cell it builds."""
    raw = read_cell_config("bench_cell")
    before = cell_from_config(raw, "bench_cell")

    raw["fixture"]["table_size_mm"] = [1800.0, 1000.0]
    raw["fixture"]["table_z_mm"] = 900.0
    raw["fixture"]["table_origin"] = [1300.0, -50.0, 0.0]
    after = cell_from_config(raw, "bench_cell")

    assert before.fixture.table_size_mm != after.fixture.table_size_mm
    assert after.fixture.table_size_mm == (1800.0, 1000.0)
    assert after.fixture.table_z_mm == 900.0
    assert np.allclose(after.fixture.table_origin, [1300.0, -50.0, 0.0])


def test_the_robot_can_be_changed():
    """Which arm is in the cell has to be part of the cell's description.

    It lived only in the shipped YAML before, so replacing the robot meant
    editing a file -- and the robot is the first thing anyone wants to change.
    """
    from weldforge.fixture import cell_robot

    raw = read_cell_config("bench_cell")
    assert cell_from_config(raw, "bench_cell").robot_name == "kuka_kr16_2"

    raw["robot"] = "abb_irb1520id"
    cell = cell_from_config(raw, "swapped")
    assert cell.robot_name == "abb_irb1520id"
    assert cell_robot(cell).n_joints == 6


def test_the_robot_base_is_editable():
    """A robot on a pedestal, and one turned away from the bench."""
    raw = default_cell_config()
    raw["robot_base"] = {"xyz": [0.0, 0.0, 400.0], "rpy_deg": [0.0, 0.0, 90.0]}
    cell = cell_from_config(raw, "riser")

    assert np.isclose(tf.translation_of(cell.robot_base)[2], 400.0)
    # Turned a quarter turn about Z: the robot's X now points along cell Y.
    assert np.allclose(cell.robot_base[:3, 0], [0.0, 1.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# the 3-D bench
# ---------------------------------------------------------------------------


def test_a_bench_model_loads_as_a_fixture_obstacle(bench_stl: Path):
    """The model has to reach collision checking and the 3-D view.

    Both walk ``all_obstacles()``, so belonging to the fixture is what makes
    an imported bench something the robot will actually avoid.
    """
    raw = default_cell_config()
    raw["fixture"]["model"] = str(bench_stl)
    cell = cell_from_config(raw, "with_bench")

    bench = [o for o in cell.all_obstacles() if o.kind == "bench"]
    assert len(bench) == 1
    assert bench[0].has_mesh
    assert cell.fixture.model_source == str(bench_stl)


def test_importing_a_bench_fits_the_box_to_it(bench_stl: Path):
    """Placement measures against the box, so the box must match the model."""
    project = Project()
    attach_bench(project, bench_stl)
    cell = cell_for(project)

    assert cell.fixture.table_z_mm == pytest.approx(BENCH_TOP)
    assert cell.fixture.table_size_mm[0] == pytest.approx(BENCH_LENGTH)
    assert cell.fixture.table_size_mm[1] == pytest.approx(BENCH_WIDTH)


def test_moving_the_bench_moves_the_model(bench_stl: Path):
    """The three bench numbers position the model, not only the box.

    This is the behaviour an operator asks for by name: nudge the bench
    further from the robot and the bench moves.
    """
    project = Project()
    attach_bench(project, bench_stl)
    project.cell_config["fixture"]["table_origin"] = [1500.0, 200.0, 0.0]
    project.cell_config["fixture"]["table_z_mm"] = 950.0

    cell = cell_for(project)
    bench = next(o for o in cell.all_obstacles() if o.kind == "bench")
    points = tf.apply_points(bench.pose, bench.vertices)

    assert (points[:, 0].min() + points[:, 0].max()) / 2.0 == pytest.approx(1500.0)
    assert (points[:, 1].min() + points[:, 1].max()) / 2.0 == pytest.approx(200.0)
    assert points[:, 2].max() == pytest.approx(950.0)


def test_a_bench_model_that_went_missing_does_not_lose_the_cell(tmp_path: Path):
    """A broken reference costs the model, not the rest of the bench."""
    raw = default_cell_config()
    raw["fixture"]["model"] = str(tmp_path / "not_here.stl")
    cell = cell_from_config(raw, "broken")

    assert cell.fixture.model_source == ""
    assert cell.fixture.table_z_mm == 750.0
    assert not [o for o in cell.all_obstacles() if o.kind == "bench"]


def test_something_with_no_surface_is_refused_as_a_bench(tmp_path: Path):
    """A file that yields no triangles cannot be a bench, and says so."""
    empty = tmp_path / "nothing.stl"
    empty.write_text("not an stl")
    with pytest.raises(Exception):  # noqa: B017 - any read failure is a refusal
        bench_model_from_file(empty)


def test_removing_the_model_keeps_the_numbers(bench_stl: Path):
    project = Project()
    attach_bench(project, bench_stl)
    clear_bench(project)
    cell = cell_for(project)

    assert cell.fixture.model_source == ""
    assert project.bench_bytes is None
    assert cell.fixture.table_z_mm == pytest.approx(BENCH_TOP)
    assert cell.fixture.table_size_mm[0] == pytest.approx(BENCH_LENGTH)


# ---------------------------------------------------------------------------
# the clamps
# ---------------------------------------------------------------------------


def test_clamps_are_editable():
    """Moving a clamp moves it everywhere: view, collision, placement, URDF."""
    raw = read_cell_config("bench_cell")
    raw["fixture"]["clamps"] = [
        {
            "id": "c1",
            "name": "morsetto anteriore",
            "position": [700.0, -300.0, 760.0],
            "approach": [1.0, 0.0, 0.0],
            "clearance_mm": 60.0,
            "height_mm": 150.0,
            "retractable": True,
        }
    ]
    cell = cell_from_config(raw, "edited")

    assert len(cell.fixture.clamps) == 1
    clamp = cell.fixture.clamps[0]
    assert clamp.name == "morsetto anteriore"
    assert np.allclose(clamp.position, [700.0, -300.0, 760.0])
    assert np.allclose(clamp.approach, [1.0, 0.0, 0.0])
    assert clamp.clearance_mm == 60.0
    assert clamp.height_mm == 150.0
    assert clamp.retractable


def test_clamps_can_be_removed_entirely():
    """A bare bench with no clamps is a cell, not an error."""
    raw = read_cell_config("bench_cell")
    assert cell_from_config(raw, "bench_cell").fixture.clamps

    raw["fixture"]["clamps"] = []
    cell = cell_from_config(raw, "bare")
    assert cell.fixture.clamps == []


def test_an_edited_clamp_reaches_the_collision_scene():
    """The real test of "editable": the planner has to see it.

    A clamp the operator moved that the collision scene still has at the old
    place is worse than no clamp at all.
    """
    from weldforge.collide import scene_from_cell

    raw = read_cell_config("bench_cell")
    raw["fixture"]["clamps"] = [
        {"id": "solo", "position": [1234.0, 56.0, 800.0], "clearance_mm": 50.0}
    ]
    scene = scene_from_cell(cell_from_config(raw, "edited"))
    assert "clamp:solo" in scene.names


def test_clamps_survive_being_saved_in_a_project(tmp_path: Path):
    project = Project()
    project.cell_config = read_cell_config("bench_cell")
    project.cell_config["fixture"]["clamps"][0]["position"] = [742.0, -301.0, 755.0]
    project.cell_config["fixture"]["clamps"][0]["retractable"] = True

    reopened = load_project(save_project(project, tmp_path / "job.weldforge"), load_cad=False)
    clamp = cell_for(reopened).fixture.clamps[0]
    assert np.allclose(clamp.position, [742.0, -301.0, 755.0])
    assert clamp.retractable


def test_clamps_survive_being_saved_in_the_library():
    raw = default_cell_config()
    raw["fixture"]["clamps"] = [
        {"id": "a", "position": [100.0, 0.0, 700.0], "clearance_mm": 33.0},
        {"id": "b", "position": [200.0, 0.0, 700.0], "clearance_mm": 44.0},
    ]
    save_cell_config(raw, "two_clamps")

    clamps = cell_from_config(read_cell_config("two_clamps"), "two_clamps").fixture.clamps
    assert [c.id for c in clamps] == ["a", "b"]
    assert [c.clearance_mm for c in clamps] == [33.0, 44.0]


# ---------------------------------------------------------------------------
# dragging, which edits the same description the panel does
# ---------------------------------------------------------------------------


def _drag(tool, key: str, grab, to):
    """One press-move-release, the way the window drives the tool."""
    from weldforge.app.scene3d import PickResult

    held = tool.on_pick(PickResult(kind="cell", id=key, point=np.asarray(grab, dtype=float)))
    assert held.ok, held.message
    tool.on_drag(np.asarray(to, dtype=float))
    return tool.on_drop()


@pytest.fixture
def jig_project() -> Project:
    project = Project()
    project.cell_config = read_cell_config("jig_cell")
    project.cell = cell_for(project)
    return project


def test_dragging_the_bench_moves_it(jig_project: Project):
    from weldforge.app.tools import MoveCellTool

    before = list(jig_project.cell_config["fixture"]["table_origin"])
    result = _drag(MoveCellTool(jig_project), "table", [850.0, 0.0, 800.0], [1100.0, 120.0, 800.0])

    assert result.changed
    after = jig_project.cell_config["fixture"]["table_origin"]
    assert after[0] == pytest.approx(before[0] + 250.0)
    assert after[1] == pytest.approx(before[1] + 120.0)
    assert np.allclose(cell_for(jig_project).fixture.table_origin[:2], after[:2])


def test_dragging_the_bench_model_moves_the_bench(bench_stl: Path):
    """Grabbing the imported model is grabbing the bench, not a separate thing."""
    from weldforge.app.tools import MoveCellTool

    project = Project()
    attach_bench(project, bench_stl)
    before = list(project.cell_config["fixture"]["table_origin"])

    result = _drag(
        MoveCellTool(project), "bench:bench_model", [0.0, 0.0, BENCH_TOP], [300.0, 0.0, BENCH_TOP]
    )
    assert result.changed
    assert project.cell_config["fixture"]["table_origin"][0] == pytest.approx(before[0] + 300.0)


def test_dragging_a_clamp_moves_that_clamp(jig_project: Project):
    from weldforge.app.tools import MoveCellTool

    clamps = jig_project.cell_config["fixture"]["clamps"]
    identifier = clamps[0]["id"]
    before = list(clamps[0]["position"])
    others = [list(c["position"]) for c in clamps[1:]]

    result = _drag(
        MoveCellTool(jig_project),
        f"clamp:{identifier}",
        before,
        [before[0] + 75.0, before[1] - 40.0, before[2]],
    )
    assert result.changed
    assert clamps[0]["position"][0] == pytest.approx(before[0] + 75.0)
    assert clamps[0]["position"][1] == pytest.approx(before[1] - 40.0)
    assert [list(c["position"]) for c in clamps[1:]] == others, "the others stayed put"


def test_dragging_an_obstacle_moves_it(jig_project: Project):
    from weldforge.app.tools import MoveCellTool

    obstacle = jig_project.cell_config["obstacles"][0]
    before = list(obstacle["centre"])

    result = _drag(
        MoveCellTool(jig_project),
        f"{obstacle.get('kind', 'jig')}:{obstacle['id']}",
        before,
        [before[0] - 200.0, before[1], before[2]],
    )
    assert result.changed
    assert obstacle["centre"][0] == pytest.approx(before[0] - 200.0)


def test_a_drag_never_changes_height(jig_project: Project):
    """Height is a measurement. A bench dragged into the floor describes nothing."""
    from weldforge.app.tools import MoveCellTool

    tool = MoveCellTool(jig_project)
    before = list(jig_project.cell_config["fixture"]["table_origin"])
    _drag(tool, "table", [850.0, 0.0, 800.0], [900.0, 0.0, 5000.0])

    assert jig_project.cell_config["fixture"]["table_origin"][2] == pytest.approx(before[2])


def test_a_click_that_does_not_move_is_not_an_edit(jig_project: Project):
    """Otherwise every stray click would dirty the project."""
    from weldforge.app.tools import MoveCellTool

    result = _drag(
        MoveCellTool(jig_project), "table", [850.0, 0.0, 800.0], [850.2, 0.1, 800.0]
    )
    assert not result.changed


def test_escape_puts_a_dragged_element_back(jig_project: Project):
    from weldforge.app.scene3d import PickResult
    from weldforge.app.tools import MoveCellTool

    tool = MoveCellTool(jig_project)
    before = list(jig_project.cell_config["fixture"]["table_origin"])

    tool.on_pick(PickResult(kind="cell", id="table", point=np.array([850.0, 0.0, 800.0])))
    tool.on_drag(np.array([3000.0, 0.0, 800.0]))
    assert tool.dragging()
    tool.on_command("cancel")

    assert not tool.dragging()
    assert jig_project.cell_config["fixture"]["table_origin"] == before


def test_things_that_cannot_be_moved_say_so(jig_project: Project):
    from weldforge.app.scene3d import PickResult
    from weldforge.app.tools import MoveCellTool

    tool = MoveCellTool(jig_project)
    result = tool.on_pick(PickResult(kind="cell", id="robot:link3", point=np.zeros(3)))

    # The behaviour is asserted, not the wording: the message is shown to an
    # operator and will be translated.
    assert not result.ok
    assert result.message
    assert not tool.dragging()
    assert tool.drag_preview() is None


def test_dragging_reports_an_offset_to_draw(jig_project: Project):
    """The window shifts the actor by this while the mouse is down."""
    from weldforge.app.scene3d import PickResult
    from weldforge.app.tools import MoveCellTool

    tool = MoveCellTool(jig_project)
    assert tool.drag_preview() is None

    tool.on_pick(PickResult(kind="cell", id="table", point=np.array([850.0, 0.0, 800.0])))
    tool.on_drag(np.array([950.0, 30.0, 800.0]))
    key, offset = tool.drag_preview()

    assert key == "table"
    assert np.allclose(offset, [100.0, 30.0, 0.0])
    assert tool.drag_plane_z() == pytest.approx(800.0)


# ---------------------------------------------------------------------------
# obstacles imported as models: a welding jig, a positioner housing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def jig_stl(tmp_path_factory) -> Path:
    """A jig modelled far from the origin, as a vendor file would be."""
    import trimesh

    jig = trimesh.creation.box(extents=[600.0, 400.0, 700.0])
    jig.apply_translation([4000.0, -3000.0, 350.0])
    out = tmp_path_factory.mktemp("jig") / "Maschera Saldatura.stl"
    jig.export(str(out))
    return out


def test_an_imported_obstacle_lands_on_the_bench_with_its_feet_on_the_floor(jig_stl: Path):
    from weldforge.app.project import attach_obstacle

    project = Project()
    project.cell_config = read_cell_config("bench_cell")
    entry = attach_obstacle(project, jig_stl)

    cell = cell_for(project)
    obstacle = next(o for o in cell.all_obstacles() if o.id == entry["id"])
    assert obstacle.has_mesh
    placed = tf.apply_points(obstacle.pose, obstacle.vertices)
    bench = cell.fixture.table_origin
    assert (placed[:, 0].min() + placed[:, 0].max()) / 2.0 == pytest.approx(bench[0])
    assert (placed[:, 1].min() + placed[:, 1].max()) / 2.0 == pytest.approx(bench[1])
    assert placed[:, 2].min() == pytest.approx(0.0), "it stands on the floor, not in it"


def test_an_imported_obstacle_travels_with_the_project(tmp_path: Path, jig_stl: Path):
    from weldforge.app.project import attach_obstacle

    project = Project()
    project.cell_config = read_cell_config("bench_cell")
    entry = attach_obstacle(project, jig_stl)
    project.cell_config["obstacles"][-1]["centre"] = [1300.0, 250.0, 0.0]

    reopened = load_project(save_project(project, tmp_path / "job.weldforge"), load_cad=False)
    assert entry["id"] in reopened.obstacle_models
    obstacle = next(o for o in cell_for(reopened).all_obstacles() if o.id == entry["id"])
    placed = tf.apply_points(obstacle.pose, obstacle.vertices)
    assert (placed[:, 0].min() + placed[:, 0].max()) / 2.0 == pytest.approx(1300.0)


def test_an_imported_obstacle_can_be_removed(jig_stl: Path):
    from weldforge.app.project import attach_obstacle, remove_obstacle

    project = Project()
    project.cell_config = read_cell_config("bench_cell")
    entry = attach_obstacle(project, jig_stl)
    assert remove_obstacle(project, entry["id"])
    assert entry["id"] not in {o["id"] for o in project.cell_config["obstacles"]}
    assert entry["id"] not in project.obstacle_models
    assert not remove_obstacle(project, entry["id"]), "removing twice is a no-op, not an error"


def test_a_saved_cell_keeps_its_obstacle_models(jig_stl: Path):
    """The model is copied beside the YAML, like the bench model is."""
    raw = default_cell_config()
    raw["obstacles"] = [
        {"id": "jig", "name": "jig", "kind": "jig", "model": str(jig_stl), "centre": [900.0, 0, 0]}
    ]
    path = save_cell_config(raw, "cella_con_maschera")
    assert any(p.suffix == ".stl" for p in path.parent.iterdir())
    reloaded = cell_from_config(read_cell_config("cella_con_maschera"), "cella_con_maschera")
    assert next(o for o in reloaded.all_obstacles() if o.id == "jig").has_mesh


def test_a_dragging_tool_says_so_and_gives_its_grab_point(jig_project: Project):
    """The window routes Enter/Esc and picks the drag plane from these."""
    from weldforge.app.scene3d import PickResult
    from weldforge.app.tools import MoveCellTool

    tool = MoveCellTool(jig_project)
    assert not tool.busy() and tool.drag_through() is None
    tool.on_pick(PickResult(kind="cell", id="table", point=np.array([850.0, 0.0, 800.0])))
    assert tool.busy()
    assert np.allclose(tool.drag_through(), [850.0, 0.0, 800.0])
    tool.on_command("cancel")
    assert not tool.busy() and tool.drag_through() is None


# ---------------------------------------------------------------------------
# reading the bench model
# ---------------------------------------------------------------------------


def test_the_bench_mesh_is_read_once_per_file(bench_stl: Path):
    """The cell is rebuilt on every nudge of a bench dimension.

    Re-reading a STEP of a real bench each time would make the spin boxes
    unusable, so the mesh is cached by file identity.
    """
    from weldforge.fixture import cell as cell_module

    cell_module._MESH_CACHE.clear()
    raw = default_cell_config()
    raw["fixture"]["model"] = str(bench_stl)

    for step in range(5):
        raw["fixture"]["table_origin"] = [1000.0 + step, 0.0, 0.0]
        cell_from_config(raw, "repeated")

    assert len(cell_module._MESH_CACHE) == 1


def test_a_bench_model_edited_in_cad_is_read_again(tmp_path: Path, bench_stl: Path):
    """Caching must not mean showing yesterday's bench."""
    import trimesh

    from weldforge.fixture import cell as cell_module

    cell_module._MESH_CACHE.clear()
    target = tmp_path / "bench.stl"
    target.write_bytes(bench_stl.read_bytes())
    before, _ = bench_model_from_file(target)

    trimesh.creation.box(extents=[10.0, 10.0, 10.0]).export(str(target))
    after, _ = bench_model_from_file(target)

    assert len(after) != len(before)


# ---------------------------------------------------------------------------
# persistence: the point of the whole thing
# ---------------------------------------------------------------------------


def test_a_project_carries_its_cell_and_its_bench(tmp_path: Path, bench_stl: Path):
    """Reopening a job must not mean describing the bench again."""
    project = Project()
    attach_bench(project, bench_stl)
    project.cell_config["fixture"]["table_origin"] = [1250.0, 0.0, 0.0]
    project.cell_config["name"] = "Shop floor bench"

    saved = save_project(project, tmp_path / "job.weldforge")
    reopened = load_project(saved, load_cad=False)

    assert reopened.cell_config is not None
    assert reopened.bench_bytes == bench_stl.read_bytes()

    cell = cell_for(reopened)
    assert cell.name == "Shop floor bench"
    assert np.allclose(cell.fixture.table_origin, [1250.0, 0.0, 0.0])
    assert next(o for o in cell.all_obstacles() if o.kind == "bench").has_mesh


def test_a_project_without_a_cell_config_still_opens(tmp_path: Path):
    """Projects written before the cell travelled with them must keep working."""
    project = Project()
    project.cell_name = "bench_cell"
    saved = save_project(project, tmp_path / "old.weldforge")

    reopened = load_project(saved, load_cad=False)
    reopened.cell_config = None  # as a format-1 project arrives
    cell = cell_for(reopened)
    assert cell.fixture.table_z_mm == 750.0


def test_the_cell_does_not_leak_the_machine_it_was_read_on(tmp_path: Path):
    """``_dir`` records a local folder, so it must not reach the project file."""
    project = Project()
    project.cell_config = read_cell_config("bench_cell")
    assert "_dir" in project.cell_config

    saved = save_project(project, tmp_path / "job.weldforge")
    reopened = load_project(saved, load_cad=False)
    assert "_dir" not in (reopened.cell_config or {})


def test_a_saved_cell_joins_the_library_and_reloads(bench_stl: Path):
    """"Save cell" has to produce something the next job can simply pick."""
    project = Project()
    attach_bench(project, bench_stl)
    raw = {**project.cell_config, "name": "Banco officina"}
    raw["fixture"] = {**raw["fixture"], "model": str(bench_stl)}

    path = save_cell_config(raw, "banco_officina")
    assert path.exists()
    assert "banco_officina" in available_cells()

    # The model is copied in beside the YAML, so the cell stands on its own.
    reloaded = cell_from_config(read_cell_config("banco_officina"), "banco_officina")
    assert reloaded.name == "Banco officina"
    assert reloaded.fixture.model_source
    assert next(o for o in reloaded.all_obstacles() if o.kind == "bench").has_mesh


def test_saving_a_cell_whose_model_is_gone_keeps_the_cell(tmp_path: Path):
    """Losing the model must not mean losing the bench you measured."""
    raw = default_cell_config()
    raw["fixture"]["model"] = str(tmp_path / "vanished.stl")
    raw["fixture"]["table_z_mm"] = 880.0

    save_cell_config(raw, "no_model")
    reloaded = cell_from_config(read_cell_config("no_model"), "no_model")
    assert reloaded.fixture.table_z_mm == 880.0
    assert reloaded.fixture.model_source == ""
