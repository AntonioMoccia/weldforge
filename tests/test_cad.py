"""Tests for the CAD layer: backend adapter, topology, loader.

The invariants checked here are the ones whose silent failure would wreck
every downstream stage:

* the backend resolves OCCT's differently-named statics and template
  instantiations;
* shape identity uses OCCT's hash, not Python ``id()``;
* face normals point **out of** the solid, including on REVERSED faces;
* an edge's length is its arc length, not its chord;
* the closing seam of a periodic face is not mistaken for a joint.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import PLATE_THICK, WEB_THICK, requires_cad

from weldforge.cad import backend as be
from weldforge.cad import load_part
from weldforge.cad.loader import LoadOptions
from weldforge.cad.topology import FaceEvaluator, estimate_thickness, extract, sample_edge
from weldforge.geom import polyline as pl

pytestmark = pytest.mark.needs_cad


# ---------------------------------------------------------------------------
# backend adapter
# ---------------------------------------------------------------------------


@requires_cad
def test_backend_resolves():
    b = be.get_backend()
    assert b.name in ("OCP", "pythonocc-core")
    assert be.backend_name() == b.name


@requires_cad
def test_backend_is_cached():
    assert be.get_backend() is be.get_backend()


def test_unknown_backend_preference_raises():
    with pytest.raises(be.CadBackendError):
        be.get_backend(prefer="not-a-binding")


@requires_cad
def test_static_resolution_across_naming_styles():
    """OCP spells it ``SurfaceProperties_s``, pythonocc ``SurfaceProperties``."""
    b = be.get_backend()
    assert callable(b.static("BRepGProp", "BRepGProp", "SurfaceProperties"))
    assert callable(b.static("TopExp", "TopExp", "MapShapesAndAncestors"))
    with pytest.raises(be.CadBackendError):
        b.static("BRepGProp", "BRepGProp", "NoSuchMethod")


@requires_cad
def test_first_sym_finds_the_template_instantiation():
    """The indexed shape map lives under a different name in each binding."""
    b = be.get_backend()
    cls = b.first_sym(
        (
            ("TopTools", "TopTools_IndexedDataMapOfShapeListOfShape"),
            (
                "collections",
                "IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher",
            ),
        )
    )
    assert cls is not None
    with pytest.raises(be.CadBackendError):
        b.first_sym((("TopTools", "Nope"), ("collections", "AlsoNope")))


@requires_cad
def test_shape_enum_rejects_nonsense():
    with pytest.raises(ValueError):
        be.get_backend().shape_enum("BANANA")


@requires_cad
def test_shape_key_is_topological_not_object_identity(tee_assembly_step):
    """Two wrapper objects for the same face must share a key.

    The bindings hand out a fresh Python wrapper per traversal, so keying
    adjacency on ``id()`` yields an empty map -- the bug this guards against.
    """
    b = be.get_backend()
    shape = b.read_step(str(tee_assembly_step))
    first = list(b.explore(shape, "face"))
    second = list(b.explore(shape, "face"))
    assert len(first) == len(second)
    assert any(a is not bb for a, bb in zip(first, second, strict=True))
    assert [b.shape_key(f) for f in first] == [b.shape_key(f) for f in second]


@requires_cad
def test_explore_unique_dedupes(tee_fused_step):
    b = be.get_backend()
    shape = b.read_step(str(tee_fused_step))
    assert len(list(b.explore_unique(shape, "edge"))) < len(list(b.explore(shape, "edge")))


@requires_cad
def test_bbox_of_a_known_box(tee_assembly_step):
    b = be.get_backend()
    lo, hi = b.bbox(b.read_step(str(tee_assembly_step)))
    assert np.allclose(hi - lo, [200.0, 100.0, PLATE_THICK + 60.0], atol=1e-6)


@requires_cad
def test_volume_properties_of_a_known_box(tee_assembly_step):
    b = be.get_backend()
    shape = b.read_step(str(tee_assembly_step))
    volumes = sorted(b.volume_properties(s)[0] for s in b.explore_unique(shape, "solid"))
    expected = sorted([200.0 * 100.0 * PLATE_THICK, 200.0 * WEB_THICK * 60.0])
    assert np.allclose(volumes, expected, rtol=1e-9)


@requires_cad
def test_step_read_failure_is_loud(tmp_path):
    bad = tmp_path / "not-really.step"
    bad.write_text("this is not a STEP file\n")
    with pytest.raises(be.CadBackendError):
        be.get_backend().read_step(str(bad))


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------


@requires_cad
def test_extract_counts_a_fused_tee(tee_fused_step):
    b = be.get_backend()
    topo = extract(b, b.read_step(str(tee_fused_step)))
    assert topo.counts["solids"] == 1
    assert topo.counts["faces"] == 12
    # Every edge of a closed solid is shared by exactly two faces.
    assert all(topo.edge_kind(ek) == "manifold" for ek in topo.edges)


@requires_cad
def test_extract_keeps_solids_separate_in_an_assembly(tee_assembly_step):
    b = be.get_backend()
    topo = extract(b, b.read_step(str(tee_assembly_step)))
    assert topo.counts["solids"] == 2
    # The two bodies touch but share no topology: that is the whole reason
    # contact-based seam detection has to exist.
    assert all(len(topo.edge_solids(ek)) == 1 for ek in topo.edges)


@requires_cad
def test_periodic_seam_edges_are_labelled(plate_with_hole_step):
    """The hole's closing seam must not look like a joint."""
    b = be.get_backend()
    topo = extract(b, b.read_step(str(plate_with_hole_step)))
    kinds = {ek: topo.edge_kind(ek) for ek in topo.edges}
    assert "periodic_seam" in kinds.values()
    assert sum(1 for v in kinds.values() if v == "periodic_seam") == 1


@requires_cad
def test_face_normals_point_out_of_the_solid(tee_fused_step):
    """The invariant that decides whether the torch points at the work or into it.

    Probed with OCCT's own solid classifier from points guaranteed to be inside
    the trimmed face, so a trimmed or annular face cannot produce a false
    failure.
    """
    b = be.get_backend()
    shape = b.read_step(str(tee_fused_step))
    b.tessellate(shape, 0.3)  # populate triangulations for on-face sampling
    topo = extract(b, shape)

    classifier_cls = b.sym("BRepClass3d", "BRepClass3d_SolidClassifier")
    state_in = b.sym("TopAbs", "TopAbs_State").TopAbs_IN
    gp = b.module("gp")

    inward = []
    for sk, solid in topo.solids.items():
        clf = classifier_cls(solid)
        for fk in (k for k, v in topo.face_solid.items() if v == sk):
            ev = FaceEvaluator(b, topo.faces[fk])
            for point in ev.sample_on_face(3):
                probe = point + ev.normal_at_point(point) * 0.2
                clf.Perform(gp.gp_Pnt(*[float(x) for x in probe]), 1e-6)
                if clf.State() == state_in:
                    inward.append(topo.face_ids[fk])
                    break
    assert inward == []


@requires_cad
def test_reversed_faces_exist_so_the_test_above_means_something(tee_fused_step):
    """Guard the guard: if nothing is REVERSED the orientation test is vacuous."""
    b = be.get_backend()
    shape = b.read_step(str(tee_fused_step))
    assert any(b.is_reversed(f) for f in b.explore_unique(shape, "face"))


@requires_cad
def test_sample_edge_of_a_line_is_two_points(tee_fused_step):
    b = be.get_backend()
    shape = b.read_step(str(tee_fused_step))
    for edge in b.explore_unique(shape, "edge"):
        points, curve_type, _ = sample_edge(b, edge, spacing_mm=2.0)
        if curve_type == "line":
            assert len(points) == 2
            return
    pytest.fail("the fused tee should contain straight edges")


@requires_cad
def test_sample_edge_arc_length_beats_the_chord(plate_with_hole_step):
    """An R15 hole rim is 94.2 mm around; its chord is 0. Sampling must know."""
    b = be.get_backend()
    shape = b.read_step(str(plate_with_hole_step))
    circles = []
    for edge in b.explore_unique(shape, "edge"):
        points, curve_type, radius = sample_edge(b, edge, spacing_mm=2.0)
        if curve_type == "circle":
            circles.append((points, radius))
    assert circles, "the plate should have circular hole rims"
    points, radius = circles[0]
    assert np.isclose(radius, 15.0, atol=1e-6)
    assert np.isclose(pl.total_length(points), 2.0 * np.pi * 15.0, rtol=2e-3)
    assert len(points) >= 10


@requires_cad
def test_sample_edge_respects_spacing(curved_tee_step):
    b = be.get_backend()
    shape = b.read_step(str(curved_tee_step))
    for edge in b.explore_unique(shape, "edge"):
        points, curve_type, _ = sample_edge(b, edge, spacing_mm=1.0)
        if curve_type == "circle" and pl.total_length(points) > 50.0:
            steps = pl.segment_lengths(points)
            assert steps.max() <= 1.5
            return
    pytest.fail("the curved tee should contain a long arc")


@requires_cad
def test_estimate_thickness_of_known_plates(tee_assembly_step):
    """8 mm base and 6 mm web, recovered from geometry alone."""
    b = be.get_backend()
    shape = b.read_step(str(tee_assembly_step))
    found = sorted(estimate_thickness(b, solid) for solid in b.explore_unique(shape, "solid"))
    assert np.allclose(found, [WEB_THICK, PLATE_THICK], atol=1e-6)


@requires_cad
def test_face_evaluator_distance_to_plane(tee_fused_step):
    b = be.get_backend()
    shape = b.read_step(str(tee_fused_step))
    topo = extract(b, shape)
    for face in topo.faces.values():
        ev = FaceEvaluator(b, face)
        if ev.is_planar:
            centre = ev.center()
            assert ev.distance_to_point(centre) < 1e-6
            assert np.isclose(ev.distance_to_point(centre + ev.normal_at_point(centre) * 7.0), 7.0)
            return


@requires_cad
def test_cylindrical_normal_is_radial(plate_with_hole_step):
    """A hole wall's outward normal points at the axis, not away from it."""
    b = be.get_backend()
    shape = b.read_step(str(plate_with_hole_step))
    axis_xy = np.array([60.0, 60.0])
    for face in b.explore_unique(shape, "face"):
        ev = FaceEvaluator(b, face)
        if ev.face_type != "cylindrical":
            continue
        point = ev.center()
        normal = ev.normal_at_point(point)
        radial = np.array([point[0] - axis_xy[0], point[1] - axis_xy[1], 0.0])
        radial /= np.linalg.norm(radial)
        # Hole wall: the solid is outside, so its outward normal points inward.
        assert np.isclose(abs(np.dot(normal, radial)), 1.0, atol=1e-6)
        assert np.isclose(normal[2], 0.0, atol=1e-9)
        return
    pytest.fail("no cylindrical face found")


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


@requires_cad
def test_load_part_populates_everything(tee_assembly_step):
    part = load_part(tee_assembly_step)
    assert part.name == "tee_assembly"
    assert len(part.solids) == 2
    assert len(part.faces) == 12
    assert len(part.edges) == 24
    assert part.mesh_vertices is not None and len(part.mesh_vertices) > 0
    assert part.metadata["geometry_kind"] == "brep"
    assert np.allclose(part.bbox_size, [200.0, 100.0, 68.0], atol=1e-6)


@requires_cad
def test_load_part_mass_of_known_plates(tee_assembly_step):
    part = load_part(tee_assembly_step)
    volume_mm3 = 200.0 * 100.0 * PLATE_THICK + 200.0 * WEB_THICK * 60.0
    assert np.isclose(part.mass_kg, volume_mm3 * 7.85e-6, rtol=1e-6)


@requires_cad
def test_load_part_thickness_per_solid(tee_assembly_step):
    part = load_part(tee_assembly_step)
    assert sorted(round(s.thickness_mm, 3) for s in part.solids) == [WEB_THICK, PLATE_THICK]


@requires_cad
def test_load_part_drops_periodic_seams(plate_with_hole_step):
    b = be.get_backend()
    topo = extract(b, b.read_step(str(plate_with_hole_step)))
    part = load_part(plate_with_hole_step)
    n_periodic = sum(1 for ek in topo.edges if topo.edge_kind(ek) == "periodic_seam")
    assert n_periodic >= 1
    assert len(part.edges) == len(topo.edges) - n_periodic


@requires_cad
def test_load_part_edge_arc_lengths(plate_with_hole_step):
    part = load_part(plate_with_hole_step)
    rims = [e for e in part.edges.values() if e.curve_type == "circle"]
    assert len(rims) == 2  # top and bottom of the hole
    for rim in rims:
        assert np.isclose(rim.length_mm, 2.0 * np.pi * 15.0, rtol=2e-3)


@requires_cad
def test_load_part_without_mesh_is_faster_and_meshless(tee_assembly_step):
    part = load_part(tee_assembly_step, LoadOptions(with_mesh=False))
    assert part.mesh_vertices is None
    assert len(part.faces) == 12


@requires_cad
def test_load_mesh_part_from_stl(stl_tee):
    part = load_part(stl_tee)
    assert part.metadata["geometry_kind"] == "mesh"
    assert part.mesh_faces is not None and len(part.mesh_faces) > 0
    assert part.faces == {} and part.edges == {}
    # The two touching plates tessellate into two connected components.
    assert len(part.solids) == 2
    assert np.allclose(part.bbox_size, [200.0, 100.0, 68.0], atol=0.6)


def test_load_part_rejects_unknown_format(tmp_path):
    p = tmp_path / "part.xyz"
    p.write_text("nope")
    with pytest.raises(ValueError, match="unsupported"):
        load_part(p)


def test_load_part_missing_file():
    with pytest.raises(FileNotFoundError):
        load_part("/nonexistent/part.step")


@requires_cad
def test_company_part_loads_if_present(company_step):
    """Bonus signal on the real 58-body weldment, skipped when absent."""
    if company_step is None:
        pytest.skip("customer STEP not available")
    part = load_part(company_step)
    assert len(part.solids) == 58
    assert len(part.faces) == 885
    thicknesses = {round(s.thickness_mm, 1) for s in part.solids}
    # The design uses 6 mm and 8 mm plate; nothing else should be inferred.
    assert thicknesses <= {6.0, 8.0}
    assert 11.0 < part.mass_kg < 13.0
