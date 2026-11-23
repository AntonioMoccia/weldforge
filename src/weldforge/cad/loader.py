"""Turn a CAD file into a :class:`weldforge.model.Part`.

Supported inputs
----------------
============  ===================================================  ============
Extension     Path                                                 Needs OCCT
============  ===================================================  ============
.step .stp    full B-rep: solids, faces, edges, exact normals       yes
.iges .igs    B-rep, usually surfaces only                          yes
.brep         native OpenCASCADE                                    yes
.stl .obj     triangle mesh: no exact geometry, mesh-based seams     no
.ply .3mf     ditto                                                 no
============  ===================================================  ============

The mesh path exists so the tool still works when OpenCASCADE is unavailable
or when the only export an operator can get out of their CAD seat is an STL.
Seam recognition then runs on the mesh, with lower confidence -- the
:class:`Part` records which it was in ``metadata['geometry_kind']``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..model import BrepHandles, EdgeRef, FaceRef, Part, SolidInfo
from .backend import CadBackend, CadBackendError, get_backend
from .topology import FaceEvaluator, estimate_thickness, extract, sample_edge

logger = logging.getLogger(__name__)

__all__ = [
    "LoadOptions",
    "UnsupportedCadFormat",
    "load_part",
    "load_brep_part",
    "load_mesh_part",
    "BREP_SUFFIXES",
    "MESH_SUFFIXES",
    "NATIVE_SUFFIXES",
    "ALL_SUFFIXES",
]


class UnsupportedCadFormat(ValueError):
    """The file is not in a format this installation can read.

    A ``ValueError`` subclass rather than a new exception root, so that every
    caller already written to catch ``ValueError`` around ``load_part`` keeps
    working. What it adds is a message an operator can act on without reading
    the source.
    """

BREP_SUFFIXES = {".step", ".stp", ".iges", ".igs", ".brep"}
MESH_SUFFIXES = {".stl", ".obj", ".ply", ".3mf", ".off", ".glb", ".gltf"}

#: Native CAD formats, which are closed and readable only by the seat that
#: wrote them. Recognising them by name is worth the dictionary: an operator
#: who picks a .sldprt deserves "export it as STEP" rather than a list of
#: extensions they have to diff against their own by eye.
NATIVE_SUFFIXES: dict[str, str] = {
    ".sldprt": "SolidWorks part",
    ".sldasm": "SolidWorks assembly",
    ".ipt": "Inventor part",
    ".iam": "Inventor assembly",
    ".catpart": "CATIA part",
    ".catproduct": "CATIA product",
    ".prt": "NX / Creo part",
    ".asm": "Creo assembly",
    ".par": "Solid Edge part",
    ".psm": "Solid Edge sheet metal",
    ".x_t": "Parasolid text",
    ".x_b": "Parasolid binary",
    ".xmt_txt": "Parasolid text",
    ".xmt_bin": "Parasolid binary",
    ".sat": "ACIS",
    ".sab": "ACIS binary",
    ".f3d": "Fusion 360 archive",
    ".3dm": "Rhino",
    ".dwg": "AutoCAD drawing",
    ".dxf": "DXF drawing",
    ".fcstd": "FreeCAD document",
    ".jt": "JT",
    ".3dxml": "3DXML",
}

#: Every extension the file dialog should offer.
ALL_SUFFIXES = BREP_SUFFIXES | MESH_SUFFIXES


class LoadOptions:
    """Tuning knobs for CAD import.

    Defaults are chosen for a steel weldment of a few hundred millimetres:

    * ``edge_spacing_mm = 2.0`` -- one seam station every 2 mm is finer than
      any weld pool, and gives a torch path that no controller chokes on.
    * ``mesh_deflection_mm = 0.5`` -- plenty for collision checking; the exact
      B-rep is still used for all seam geometry.
    * ``estimate_thickness = True`` -- costs O(planar faces squared) per solid
      and is worth it, because thickness drives the whole welding procedure.
    """

    def __init__(
        self,
        edge_spacing_mm: float = 2.0,
        edge_deflection_mm: float = 0.05,
        mesh_deflection_mm: float = 0.5,
        mesh_angular_deflection: float = 0.35,
        with_mesh: bool = True,
        estimate_thickness: bool = True,
        read_names: bool = True,
        max_faces_for_thickness: int = 400,
    ) -> None:
        self.edge_spacing_mm = edge_spacing_mm
        self.edge_deflection_mm = edge_deflection_mm
        self.mesh_deflection_mm = mesh_deflection_mm
        self.mesh_angular_deflection = mesh_angular_deflection
        self.with_mesh = with_mesh
        self.estimate_thickness = estimate_thickness
        self.read_names = read_names
        self.max_faces_for_thickness = max_faces_for_thickness


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def load_part(
    path: str | Path,
    options: LoadOptions | None = None,
    name: str | None = None,
) -> Part:
    """Load any supported CAD file into a :class:`Part`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CAD file not found: {path}")
    suffix = path.suffix.lower()

    if suffix in BREP_SUFFIXES:
        return load_brep_part(path, options, name)
    if suffix in MESH_SUFFIXES:
        return load_mesh_part(path, name)
    if suffix in NATIVE_SUFFIXES:
        seat = NATIVE_SUFFIXES[suffix].split()[0]
        raise UnsupportedCadFormat(
            f"{path.name} is a {NATIVE_SUFFIXES[suffix]} file -- an unsupported, closed "
            f"format that only {seat} itself can read.\n\n"
            "In your CAD seat use File > Save as (or Export) and pick STEP "
            "(AP214 or AP242, .step / .stp), then import that. STEP carries the exact "
            "solid geometry, which is what seam recognition works on."
        )
    raise UnsupportedCadFormat(
        f"unsupported CAD format {suffix or '(no extension)'!r}.\n\n"
        f"Exact geometry : {', '.join(sorted(BREP_SUFFIXES))}\n"
        f"Meshes         : {', '.join(sorted(MESH_SUFFIXES))}\n\n"
        "STEP is the format to ask for: every CAD system exports it, and it keeps "
        "the solids, faces and edges that seam recognition needs."
    )


# ---------------------------------------------------------------------------
# B-rep path
# ---------------------------------------------------------------------------


def load_brep_part(
    path: str | Path,
    options: LoadOptions | None = None,
    name: str | None = None,
) -> Part:
    """Load a STEP/IGES/BREP file with exact geometry."""
    opts = options or LoadOptions()
    path = Path(path)
    backend = get_backend()
    t_start = time.perf_counter()

    suffix = path.suffix.lower()
    if suffix in (".step", ".stp"):
        shape = backend.read_step(str(path))
    elif suffix in (".iges", ".igs"):
        shape = backend.read_iges(str(path))
    elif suffix == ".brep":
        shape = _read_brep(backend, str(path))
    else:  # pragma: no cover - guarded by load_part
        raise ValueError(f"not a B-rep format: {suffix}")

    topo = extract(backend, shape)
    part = Part(
        name=name or path.stem,
        source_path=str(path),
        metadata={"geometry_kind": "brep", "cad_backend": backend.name},
    )

    # -- mesh first: it populates the face triangulations that the
    #    evaluators' on-face sampling relies on.
    if opts.with_mesh:
        verts, tris = backend.tessellate(
            shape, opts.mesh_deflection_mm, opts.mesh_angular_deflection
        )
        part.mesh_vertices = verts
        part.mesh_faces = tris
        logger.info("load_brep_part: mesh %d verts / %d tris", len(verts), len(tris))

    evaluators: dict[int, FaceEvaluator] = {}
    names = _read_solid_names(backend, str(path)) if opts.read_names else {}

    # -- faces ---------------------------------------------------------------
    for fk, face in topo.faces.items():
        ev = FaceEvaluator(backend, face)
        evaluators[fk] = ev
        area, centroid = backend.surface_properties(face)
        normal = ev.normal_at_point(centroid)
        part.faces[topo.face_ids[fk]] = FaceRef(
            id=topo.face_ids[fk],
            solid_id=topo.solid_ids.get(topo.face_solid.get(fk, -1), ""),
            face_type=ev.face_type,
            area_mm2=area,
            normal=normal,
            centroid=centroid,
            plane_offset_mm=ev.plane_offset_mm,
            radius_mm=ev.radius_mm,
        )

    # -- edges ---------------------------------------------------------------
    for ek, edge in topo.edges.items():
        kind = topo.edge_kind(ek)
        if kind == "periodic_seam":
            # The closing seam of a periodic surface. It is a topological
            # artefact, never a weld, and feeding it on would create a phantom
            # seam down the inside of every hole.
            continue
        points, curve_type, radius = sample_edge(
            backend, edge, opts.edge_spacing_mm, opts.edge_deflection_mm
        )
        face_ids = tuple(topo.face_ids[fk] for fk in topo.edge_faces.get(ek, ()))
        solid_ids = tuple(topo.solid_ids[sk] for sk in topo.edge_solids(ek) if sk in topo.solid_ids)
        part.edges[topo.edge_ids[ek]] = EdgeRef(
            id=topo.edge_ids[ek],
            points=points,
            curve_type=curve_type,
            face_ids=face_ids,
            solid_ids=solid_ids,
            radius_mm=radius,
        )

    # -- solids --------------------------------------------------------------
    for sk, solid in topo.solids.items():
        solid_id = topo.solid_ids[sk]
        volume, centroid = backend.volume_properties(solid)
        lo, hi = backend.bbox(solid)
        own_faces = [fk for fk, owner in topo.face_solid.items() if owner == sk]
        area = sum(part.faces[topo.face_ids[fk]].area_mm2 for fk in own_faces)

        thickness = 0.0
        if opts.estimate_thickness:
            if len(own_faces) <= opts.max_faces_for_thickness:
                thickness = estimate_thickness(backend, solid, [evaluators[fk] for fk in own_faces])
            else:
                thickness = float(np.min(hi - lo))
                logger.debug(
                    "%s: %d faces, skipping pairwise thickness search",
                    solid_id,
                    len(own_faces),
                )

        part.solids.append(
            SolidInfo(
                id=solid_id,
                name=names.get(solid_id, ""),
                volume_mm3=volume,
                area_mm2=area,
                centroid=centroid,
                bbox_min=lo,
                bbox_max=hi,
                thickness_mm=thickness,
            )
        )

    part.metadata["load_seconds"] = round(time.perf_counter() - t_start, 3)
    part.metadata["edge_kinds"] = topo.counts
    # Live OCCT handles for the stages that need exact geometry: contact-based
    # seam detection re-queries the surfaces and casts rays into the solids.
    part.brep = BrepHandles(backend=backend, shape=shape, topology=topo, evaluators=evaluators)

    logger.info(
        "load_brep_part: %s -> %d solids, %d faces, %d edges in %.2fs",
        path.name,
        len(part.solids),
        len(part.faces),
        len(part.edges),
        part.metadata["load_seconds"],
    )
    return part


def _read_brep(backend: CadBackend, path: str) -> Any:
    builder = backend.sym("BRep", "BRep_Builder")()
    shape_cls = backend.sym("TopoDS", "TopoDS_Shape")
    shape = shape_cls()
    read_fn = backend.static("BRepTools", "BRepTools", "Read")
    if not read_fn(shape, path, builder):
        raise CadBackendError(f"BREP read failed for {path!r}")
    return shape


def _read_solid_names(backend: CadBackend, path: str) -> dict[str, str]:
    """Best-effort component names from the STEP assembly structure.

    Names let an operator recognise which body a seam belongs to
    ("piastra_laterale" rather than "solid_37"), so they are worth reading --
    but they are cosmetic, and the XCAF API differs enough between bindings
    that any failure here is swallowed.
    """
    try:
        doc_cls = backend.sym("TDocStd", "TDocStd_Document")
        ext_str = backend.sym("TCollection", "TCollection_ExtendedString")
        reader_cls = backend.sym("STEPCAFControl", "STEPCAFControl_Reader")
        tool = backend.sym("XCAFDoc", "XCAFDoc_DocumentTool")
        name_cls = backend.sym("TDataStd", "TDataStd_Name")
        seq_cls = backend.first_sym(
            (
                ("TDF", "TDF_LabelSequence"),
                ("collections", "Sequence_TDF_Label"),
            )
        )

        doc = doc_cls(ext_str("weldforge"))
        reader = reader_cls()
        reader.SetNameMode(True)
        reader.ReadFile(path)
        reader.Transfer(doc)

        shape_tool_fn = getattr(tool, "ShapeTool_s", None) or tool.ShapeTool
        shape_tool = shape_tool_fn(doc.Main())
        labels = seq_cls()
        shape_tool.GetFreeShapes(labels)

        # OCCT 8 turned GetComponents into a static, so the bound method is
        # gone. Resolve whichever form this binding has, and call it with the
        # same (root, children) arguments either way.
        static_components = getattr(type(shape_tool), "GetComponents_s", None)
        if static_components is not None:
            def get_components(root: Any, children: Any) -> None:
                static_components(root, children)
        else:
            def get_components(root: Any, children: Any) -> None:
                shape_tool.GetComponents(root, children)

        get_id = getattr(name_cls, "GetID_s", None) or name_cls.GetID

        def label_name(label: Any) -> str:
            attr = name_cls()
            if label.FindAttribute(get_id(), attr):
                return str(attr.Get().ToExtString())
            return ""

        out: dict[str, str] = {}
        index = 0
        for i in range(1, labels.Length() + 1):
            root = labels.Value(i)
            children = seq_cls()
            get_components(root, children)
            if children.Length() == 0:
                out[f"solid_{index}"] = label_name(root)
                index += 1
                continue
            for j in range(1, children.Length() + 1):
                out[f"solid_{index}"] = label_name(children.Value(j))
                index += 1
        return {k: v for k, v in out.items() if v}
    except Exception as exc:
        logger.debug("_read_solid_names unavailable: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# mesh path
# ---------------------------------------------------------------------------


def load_mesh_part(path: str | Path, name: str | None = None) -> Part:
    """Load a triangle mesh; no exact geometry, so no faces or edges.

    Connected components become pseudo-solids, which is enough for the
    contact-based seam finder and for collision checking. Seam confidence is
    capped lower downstream because a mesh cannot tell a 90.0 deg joint from an
    89.3 deg one.
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise ImportError("trimesh is required to load mesh files") from exc

    path = Path(path)
    loaded = trimesh.load(str(path), force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):  # pragma: no cover - force='mesh' handles it
        loaded = loaded.to_mesh()

    part = Part(
        name=name or path.stem,
        source_path=str(path),
        mesh_vertices=np.asarray(loaded.vertices, dtype=float),
        mesh_faces=np.asarray(loaded.faces, dtype=np.int64),
        metadata={"geometry_kind": "mesh", "cad_backend": "trimesh"},
    )

    components = loaded.split(only_watertight=False) if len(loaded.faces) else []
    if not components:
        components = [loaded]

    for i, comp in enumerate(components):
        verts = np.asarray(comp.vertices, dtype=float)
        lo = verts.min(axis=0) if len(verts) else np.zeros(3)
        hi = verts.max(axis=0) if len(verts) else np.zeros(3)
        volume = float(abs(comp.volume)) if comp.is_watertight else 0.0
        part.solids.append(
            SolidInfo(
                id=f"solid_{i}",
                name=f"component_{i}",
                volume_mm3=volume,
                area_mm2=float(comp.area),
                centroid=np.asarray(comp.centroid, dtype=float),
                bbox_min=lo,
                bbox_max=hi,
                thickness_mm=float(np.min(hi - lo)) if len(verts) else 0.0,
            )
        )

    logger.info(
        "load_mesh_part: %s -> %d components, %d tris",
        path.name,
        len(part.solids),
        len(part.mesh_faces) if part.mesh_faces is not None else 0,
    )
    return part
