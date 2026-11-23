"""Shared fixtures.

Synthetic weld specimens are built with OpenCASCADE primitives and written to
STEP in a session-scoped temp directory. The point is that the whole test
suite is reproducible on any machine: it must not depend on a customer part
sitting in someone's home directory.

Each specimen is the canonical form of one joint type, so the seam classifier
can be tested against geometry whose right answer is known by construction.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from weldforge.cad.backend import CadBackendError, backend_available, get_backend

requires_cad = pytest.mark.skipif(
    not backend_available(), reason="no OpenCASCADE backend (pip install cadquery-ocp)"
)


# ---------------------------------------------------------------------------
# OCCT construction helpers
# ---------------------------------------------------------------------------


def _box(backend, dx: float, dy: float, dz: float, origin=(0.0, 0.0, 0.0)):
    """Axis-aligned box with its min corner at *origin*."""
    gp = backend.module("gp")
    make_box = backend.sym("BRepPrimAPI", "BRepPrimAPI_MakeBox")
    return make_box(gp.gp_Pnt(*[float(v) for v in origin]), float(dx), float(dy), float(dz)).Shape()


def _rotate(backend, shape, axis_dir, angle_rad, axis_point=(0.0, 0.0, 0.0)):
    gp = backend.module("gp")
    trsf = gp.gp_Trsf()
    trsf.SetRotation(
        gp.gp_Ax1(
            gp.gp_Pnt(*[float(v) for v in axis_point]),
            gp.gp_Dir(*[float(v) for v in axis_dir]),
        ),
        float(angle_rad),
    )
    transform = backend.sym("BRepBuilderAPI", "BRepBuilderAPI_Transform")
    return transform(shape, trsf, True).Shape()


def _compound(backend, shapes):
    """Pack several solids into one compound -- an assembly, not a fused body."""
    builder = backend.sym("BRep", "BRep_Builder")()
    compound = backend.sym("TopoDS", "TopoDS_Compound")()
    builder.MakeCompound(compound)
    for s in shapes:
        builder.Add(compound, s)
    return compound


def _fuse(backend, a, b):
    """Boolean union -- a single body with shared topology at the joint."""
    fuse = backend.sym("BRepAlgoAPI", "BRepAlgoAPI_Fuse")(a, b)
    fuse.Build()
    return fuse.Shape()


def _write_step(backend, shape, path: Path) -> Path:
    writer = backend.sym("STEPControl", "STEPControl_Writer")()
    as_is = backend.sym("STEPControl", "STEPControl_StepModelType").STEPControl_AsIs
    writer.Transfer(shape, as_is)
    status = writer.Write(str(path))
    ret_done = backend.sym("IFSelect", "IFSelect_ReturnStatus").IFSelect_RetDone
    if status != ret_done:
        raise CadBackendError(f"STEP write failed for {path} (status {status})")
    return path


# ---------------------------------------------------------------------------
# specimen geometry
# ---------------------------------------------------------------------------

# One size for every specimen keeps the expected numbers easy to reason about.
PLATE_LEN = 200.0
PLATE_WIDTH = 100.0
PLATE_THICK = 8.0
WEB_HEIGHT = 60.0
WEB_THICK = 6.0


@pytest.fixture(scope="session")
def specimen_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("weld_specimens")


@pytest.fixture(scope="session")
def tee_assembly_step(specimen_dir: Path) -> Path:
    """T-joint as an **assembly**: a web standing on a base, two separate solids.

    This is what a real STEP export of a weldment looks like -- the two parts
    touch but share no topology, so the only way to find the two fillet seams
    is contact detection. 8 mm base, 6 mm web, two seams of 200 mm each.
    """
    b = get_backend()
    base = _box(b, PLATE_LEN, PLATE_WIDTH, PLATE_THICK)
    web = _box(
        b,
        PLATE_LEN,
        WEB_THICK,
        WEB_HEIGHT,
        origin=(0.0, (PLATE_WIDTH - WEB_THICK) / 2.0, PLATE_THICK),
    )
    return _write_step(b, _compound(b, [base, web]), specimen_dir / "tee_assembly.step")


@pytest.fixture(scope="session")
def tee_fused_step(specimen_dir: Path) -> Path:
    """The same T-joint **fused** into one solid.

    Here the joint shows up as two concave manifold edges with a 90 deg
    dihedral, which is what the topological detector is for.
    """
    b = get_backend()
    base = _box(b, PLATE_LEN, PLATE_WIDTH, PLATE_THICK)
    web = _box(
        b,
        PLATE_LEN,
        WEB_THICK,
        WEB_HEIGHT,
        origin=(0.0, (PLATE_WIDTH - WEB_THICK) / 2.0, PLATE_THICK),
    )
    return _write_step(b, _fuse(b, base, web), specimen_dir / "tee_fused.step")


@pytest.fixture(scope="session")
def lap_assembly_step(specimen_dir: Path) -> Path:
    """Lap joint: two 8 mm plates overlapping by 40 mm, welded along the top edge."""
    b = get_backend()
    lower = _box(b, PLATE_LEN, PLATE_WIDTH, PLATE_THICK)
    upper = _box(
        b,
        PLATE_LEN,
        PLATE_WIDTH,
        PLATE_THICK,
        origin=(0.0, PLATE_WIDTH - 40.0, PLATE_THICK),
    )
    return _write_step(b, _compound(b, [lower, upper]), specimen_dir / "lap_assembly.step")


@pytest.fixture(scope="session")
def butt_assembly_step(specimen_dir: Path) -> Path:
    """Square butt joint: two coplanar 8 mm plates with a 2 mm root gap."""
    b = get_backend()
    left = _box(b, PLATE_LEN, PLATE_WIDTH, PLATE_THICK)
    right = _box(b, PLATE_LEN, PLATE_WIDTH, PLATE_THICK, origin=(0.0, PLATE_WIDTH + 2.0, 0.0))
    return _write_step(b, _compound(b, [left, right]), specimen_dir / "butt_assembly.step")


@pytest.fixture(scope="session")
def corner_assembly_step(specimen_dir: Path) -> Path:
    """Outside corner joint: two plates forming an L, outer corner up."""
    b = get_backend()
    bottom = _box(b, PLATE_LEN, PLATE_WIDTH, PLATE_THICK)
    side = _box(b, PLATE_LEN, PLATE_THICK, WEB_HEIGHT, origin=(0.0, 0.0, PLATE_THICK))
    return _write_step(b, _compound(b, [bottom, side]), specimen_dir / "corner_assembly.step")


@pytest.fixture(scope="session")
def plate_with_hole_step(specimen_dir: Path) -> Path:
    """A 10 mm plate with a through hole: exercises periodic seam rejection.

    The cylindrical hole wall closes on itself along a seam edge that is a
    topological artefact. Any detector that treats it as a joint invents a
    phantom weld down the inside of every hole.
    """
    b = get_backend()
    gp = b.module("gp")
    plate = _box(b, 120.0, 120.0, 10.0)
    cyl = b.sym("BRepPrimAPI", "BRepPrimAPI_MakeCylinder")(
        gp.gp_Ax2(gp.gp_Pnt(60.0, 60.0, -5.0), gp.gp_Dir(0.0, 0.0, 1.0)), 15.0, 20.0
    ).Shape()
    cut = b.sym("BRepAlgoAPI", "BRepAlgoAPI_Cut")(plate, cyl)
    cut.Build()
    return _write_step(b, cut.Shape(), specimen_dir / "plate_with_hole.step")


@pytest.fixture(scope="session")
def curved_tee_step(specimen_dir: Path) -> Path:
    """A web standing on a base along a 90 deg arc -- a genuinely curved seam.

    Built by rotating a thin box about Z so the contact line follows an arc;
    the point is to prove arc length, tangents and bisectors stay right when
    the seam is not straight.
    """
    b = get_backend()
    gp = b.module("gp")
    base = _box(b, 300.0, 300.0, PLATE_THICK, origin=(-150.0, -150.0, 0.0))
    # A curved web: a cylindrical shell slice standing on the base.
    outer = b.sym("BRepPrimAPI", "BRepPrimAPI_MakeCylinder")(
        gp.gp_Ax2(gp.gp_Pnt(0.0, 0.0, PLATE_THICK), gp.gp_Dir(0.0, 0.0, 1.0)),
        100.0 + WEB_THICK / 2.0,
        WEB_HEIGHT,
        np.pi / 2.0,
    ).Shape()
    inner = b.sym("BRepPrimAPI", "BRepPrimAPI_MakeCylinder")(
        gp.gp_Ax2(gp.gp_Pnt(0.0, 0.0, PLATE_THICK - 1.0), gp.gp_Dir(0.0, 0.0, 1.0)),
        100.0 - WEB_THICK / 2.0,
        WEB_HEIGHT + 2.0,
        np.pi / 2.0,
    ).Shape()
    cut = b.sym("BRepAlgoAPI", "BRepAlgoAPI_Cut")(outer, inner)
    cut.Build()
    return _write_step(b, _compound(b, [base, cut.Shape()]), specimen_dir / "curved_tee.step")


@pytest.fixture(scope="session")
def stl_tee(specimen_dir: Path, tee_assembly_step: Path) -> Path:
    """The T-joint assembly as an STL, for the no-OCCT fallback path."""
    import trimesh

    b = get_backend()
    shape = b.read_step(str(tee_assembly_step))
    verts, tris = b.tessellate(shape, 0.3)
    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    out = specimen_dir / "tee.stl"
    mesh.export(str(out))
    return out


@pytest.fixture(scope="session")
def company_step() -> Path | None:
    """A real production weldment, if one happens to be present.

    Point ``WELDFORGE_TEST_PART`` at a STEP file to run the tests that need a
    real part. Tests that use it are skipped when it is absent -- they are a
    bonus signal, never a requirement, because the repository must not depend
    on a part that cannot be published with it.
    """
    from_environment = os.environ.get("WELDFORGE_TEST_PART")
    if from_environment:
        candidate = Path(from_environment).expanduser()
        return candidate if candidate.exists() else None

    # Otherwise the folder above the repository, which is where the file sits
    # during development.
    parent = Path(__file__).resolve().parents[2]
    return next(
        (f for pattern in ("*.stp", "*.step") for f in sorted(parent.glob(pattern))),
        None,
    )


# ---------------------------------------------------------------------------
# synthetic seams (no CAD required)
# ---------------------------------------------------------------------------


def make_seam(
    joint_type,
    thickness_a: float,
    thickness_b: float,
    length: float = 200.0,
    gap: float = 0.0,
    bisector: tuple[float, float, float] = (0.0, 0.7071, -0.7071),
    n_points: int = 21,
):
    """A straight seam with explicit flank data, for testing without CAD.

    The default bisector points down and sideways at 45 deg -- the wire
    direction of an ordinary horizontal-vertical fillet (PB). Pass a different
    one to place the seam in another welding position: ``(0, 0, -1)`` for a flat
    butt, ``(0, 0.7071, 0.7071)`` for overhead, ``(0, 1, 0)`` for horizontal.
    """
    from weldforge.model import JointType, Seam

    points = np.column_stack(
        [np.linspace(0.0, length, n_points), np.zeros(n_points), np.zeros(n_points)]
    )
    groove = np.pi if joint_type is JointType.BUTT else np.pi / 2.0
    return Seam(
        joint_type=joint_type,
        points=points,
        tangents=np.tile([1.0, 0.0, 0.0], (n_points, 1)),
        normal_a=np.tile([0.0, 0.0, 1.0], (n_points, 1)),
        normal_b=np.tile([0.0, -1.0, 0.0], (n_points, 1)),
        bisector=np.tile(bisector, (n_points, 1)),
        dihedral=np.full(n_points, groove),
        thickness_a_mm=thickness_a,
        thickness_b_mm=thickness_b,
        gap_mm=gap,
        confidence=0.85,
    )
