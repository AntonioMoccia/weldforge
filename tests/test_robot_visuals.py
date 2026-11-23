"""Drawing a robot from its link geometry instead of as a stick figure.

The arm used to be one tube through the joint origins. That is honest about
what published link lengths support, but nobody recognises their own cell in
it, and an operator judging whether a part is too close to the robot has
nothing to judge against. So a robot file may now carry one mesh per link.

What these cover: the meshes are read, each link ends up at its own forward
kinematics frame, playing a trajectory moves them, and a robot with no meshes
still gets the stick figure rather than nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from weldforge.kin import LinkVisual, fk_all, load_robot
from weldforge.kin.library import clear_cache

pytest.importorskip("trimesh", reason="building the link specimens needs trimesh")

N_JOINTS = 6
SHIPPED_ROBOTS = Path(__file__).resolve().parents[1] / "data" / "robots"


@pytest.fixture
def arm_files(tmp_path: Path) -> Path:
    """A robot file with one box per link, in its own data directory.

    Written rather than committed, and pointed at by ``data_dir``, so the test
    says nothing about which robots happen to ship with the application.
    """
    import trimesh

    robots = tmp_path / "robots"
    meshes = robots / "meshes" / "testarm"
    meshes.mkdir(parents=True)
    for index in range(N_JOINTS + 1):
        trimesh.creation.box(extents=[100.0, 100.0, 100.0 + index]).export(
            str(meshes / f"link{index}.stl")
        )

    source = yaml.safe_load((SHIPPED_ROBOTS / "kuka_kr16_2.yaml").read_text(encoding="utf-8"))
    source["name"] = "Test arm"
    source["visual"] = {
        "meshes_dir": "testarm",
        "links": [{"mesh": f"link{i}.stl"} for i in range(N_JOINTS + 1)],
    }
    (robots / "testarm.yaml").write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    clear_cache()
    yield tmp_path
    clear_cache()


def _rewrite(arm_files: Path, change) -> None:
    """Edit the test robot's visual block and drop the load cache."""
    path = arm_files / "robots" / "testarm.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    clear_cache()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_a_robot_without_meshes_has_no_visuals():
    clear_cache()
    robot = load_robot("kuka_kr16_2")
    assert robot.link_visuals == []
    assert not robot.has_visuals


def test_link_meshes_are_read(arm_files: Path):
    robot = load_robot("testarm", data_dir=arm_files)
    assert robot.has_visuals
    assert len(robot.link_visuals) == N_JOINTS + 1
    assert all(visual.has_mesh for visual in robot.link_visuals)


def test_there_is_one_visual_per_forward_kinematics_frame(arm_files: Path):
    """An off-by-one here assembles the arm out of the wrong links."""
    robot = load_robot("testarm", data_dir=arm_files)
    frames = fk_all(robot, robot.home_joints())

    # fk_all gives the base, one frame per joint, then the tool; the visuals
    # cover everything up to the flange.
    assert len(frames) == N_JOINTS + 2
    assert len(robot.link_visuals) == len(frames) - 1


def test_a_link_offset_is_kept(arm_files: Path):
    """Vendor meshes are not modelled in DH frames; the offset is the fix."""
    _rewrite(
        arm_files,
        lambda raw: raw["visual"]["links"].__setitem__(
            2, {"mesh": "link2.stl", "xyz": [10.0, 20.0, 30.0], "rpy_deg": [0.0, 0.0, 90.0]}
        ),
    )
    visual = load_robot("testarm", data_dir=arm_files).link_visuals[2]
    assert np.allclose(visual.offset[:3, 3], [10.0, 20.0, 30.0])
    assert np.allclose(visual.offset[:3, 0], [0.0, 1.0, 0.0], atol=1e-9)


def test_a_missing_link_mesh_costs_that_link_only(arm_files: Path):
    """Five of six links drawn beats refusing to load the robot."""
    (arm_files / "robots" / "meshes" / "testarm" / "link3.stl").unlink()
    clear_cache()

    robot = load_robot("testarm", data_dir=arm_files)
    assert robot.has_visuals, "the rest of the arm is still drawn"
    assert not robot.link_visuals[3].has_mesh
    assert robot.n_joints == N_JOINTS, "kinematics do not depend on the meshes"


def test_a_link_entry_may_name_no_mesh(arm_files: Path):
    _rewrite(arm_files, lambda raw: raw["visual"]["links"].__setitem__(0, {}))
    robot = load_robot("testarm", data_dir=arm_files)
    assert not robot.link_visuals[0].has_mesh
    assert robot.link_visuals[1].has_mesh


def test_an_empty_visual_block_is_not_an_error(arm_files: Path):
    _rewrite(arm_files, lambda raw: raw.__setitem__("visual", {}))
    assert load_robot("testarm", data_dir=arm_files).link_visuals == []


def test_more_links_than_joints_are_ignored(arm_files: Path):
    """A file that lists eight links for six joints must not grow the arm."""
    _rewrite(
        arm_files,
        lambda raw: raw["visual"].__setitem__(
            "links", [{"mesh": "link0.stl"}] * (N_JOINTS + 4)
        ),
    )
    assert len(load_robot("testarm", data_dir=arm_files).link_visuals) == N_JOINTS + 1


def test_link_visual_reports_whether_it_has_geometry():
    assert not LinkVisual().has_mesh
    assert not LinkVisual(vertices=np.zeros((3, 3)), faces=np.zeros((0, 3))).has_mesh
    assert LinkVisual(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]])).has_mesh


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


@pytest.fixture
def scene():
    """An offscreen renderer, which is all the drawing tests need."""
    vtk = pytest.importorskip("vtk", reason="the 3-D view needs VTK")
    from weldforge.app.scene3d import Scene3D

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(640, 480)
    renderer = vtk.vtkRenderer()
    window.AddRenderer(renderer)
    return Scene3D(renderer)


def _bench_project():
    from weldforge.app.project import Project
    from weldforge.fixture import cell_from_config, read_cell_config

    raw = read_cell_config("bench_cell")
    project = Project()
    project.cell_config = raw
    project.cell = cell_from_config(raw, "test")
    return project


def test_each_link_is_drawn_at_its_own_frame(arm_files: Path, scene):
    robot = load_robot("testarm", data_dir=arm_files)
    scene.set_project(_bench_project(), robot)

    assert len(scene._link_actors) == N_JOINTS + 1
    frames = fk_all(robot, robot.home_joints())
    for index, actor, offset in scene._link_actors:
        matrix = actor.GetUserMatrix()
        placed = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
        assert np.allclose(placed, frames[index] @ offset), f"link {index} is misplaced"


def test_playing_a_trajectory_moves_the_links(arm_files: Path, scene):
    """Matrices per frame, not a re-tessellated robot per frame."""
    robot = load_robot("testarm", data_dir=arm_files)
    scene.set_project(_bench_project(), robot)

    def x_of(actor):
        return actor.GetUserMatrix().GetElement(0, 3)

    before = [x_of(actor) for _, actor, _ in scene._link_actors]
    scene._frames = [
        {
            "arm": np.array([np.zeros(3), np.ones(3)]),
            "poses": fk_all(robot, robot.home_joints() + 0.4),
            "torch": np.array([np.zeros(3), np.ones(3)]),
            "weld": True,
        }
    ]
    scene.set_frame(0)

    after = [x_of(actor) for _, actor, _ in scene._link_actors]
    assert any(abs(a - b) > 1e-6 for a, b in zip(before, after, strict=True))


def test_a_robot_without_meshes_keeps_the_stick_figure(scene):
    """The fallback has to stay: most robots ship no geometry."""
    from weldforge.fixture import cell_robot

    clear_cache()
    project = _bench_project()
    scene.set_project(project, cell_robot(project.cell))

    assert scene._link_actors == []
    assert scene._robot_actor is not None


# ---------------------------------------------------------------------------
# the demonstration robot that ships with the application
# ---------------------------------------------------------------------------


def test_the_demo_arm_says_it_is_a_demonstration():
    """It exists to show the mechanism, and must not be mistaken for a robot."""
    raw = yaml.safe_load((SHIPPED_ROBOTS / "demo_arm.yaml").read_text(encoding="utf-8"))
    notes = " ".join(raw.get("notes", [])).lower()
    assert "demonstration" in notes
    assert "do not use it to judge clearances" in notes

    clear_cache()
    robot = load_robot("demo_arm")
    assert robot.has_visuals, "the point of it is that it is drawn as solid links"
    assert robot.n_joints == N_JOINTS
