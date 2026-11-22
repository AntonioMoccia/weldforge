"""OpenCASCADE backend adapter: one API over ``OCP`` and ``pythonocc-core``.

Why this module exists
----------------------
There are two maintained Python bindings for OpenCASCADE and we need both:

* **OCP** (``cadquery-ocp`` wheels) installs with plain ``pip`` on macOS,
  Windows and Linux. That is what an engineer running the FreeCAD plugin or
  the API on a workstation will have.
* **pythonocc-core** is conda-only but is what the ROS 2 / ROS-Industrial
  images ship, so it is what runs inside the cell container.

The two bindings wrap the same C++ library but differ in ways that break code
silently rather than loudly:

* module paths -- ``OCP.TopExp`` vs ``OCC.Core.TopExp``;
* static methods -- OCP appends ``_s`` (``BRepGProp.SurfaceProperties_s``)
  while pythonocc exposes free functions or lowercase proxies
  (``brepgprop.SurfaceProperties``);
* ``TopoDS`` downcasts -- ``TopoDS.Face`` vs ``topods.Face``;
* ``Bnd_Box.Get()`` returns a 6-tuple in pythonocc but an unregistered C++
  type in OCP, where the per-axis getters must be used instead.

Everything below funnels those differences into one place. If a third binding
ever appears, this is the only file that changes.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CadBackendError",
    "CadBackend",
    "get_backend",
    "backend_available",
    "backend_name",
]


class CadBackendError(RuntimeError):
    """No usable OpenCASCADE binding, or a call into it failed."""


_BINDINGS = (
    # (distribution name, module prefix)
    ("OCP", "OCP"),
    ("pythonocc-core", "OCC.Core"),
)


def _try_import(prefix: str) -> bool:
    try:
        importlib.import_module(f"{prefix}.TopoDS")
        return True
    except Exception:
        return False


@dataclass
class CadBackend:
    """Resolved handles into one OpenCASCADE binding.

    Attribute access is deliberately explicit rather than ``__getattr__``
    magic: a typo should fail at construction, not deep inside a 900-face
    traversal.
    """

    name: str
    prefix: str
    _modules: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- module / symbol resolution -----------------------------------------

    def module(self, name: str) -> Any:
        """Import ``<prefix>.<name>`` once and cache it."""
        if name not in self._modules:
            try:
                self._modules[name] = importlib.import_module(f"{self.prefix}.{name}")
            except ImportError as exc:  # pragma: no cover - environment specific
                raise CadBackendError(
                    f"{self.name}: cannot import module {name!r} ({exc})"
                ) from exc
        return self._modules[name]

    def sym(self, module: str, symbol: str) -> Any:
        """Fetch a class or function from a backend module."""
        mod = self.module(module)
        try:
            return getattr(mod, symbol)
        except AttributeError as exc:  # pragma: no cover - environment specific
            raise CadBackendError(
                f"{self.name}: {module}.{symbol} not found in this binding"
            ) from exc

    def first_sym(self, candidates: tuple[tuple[str, str], ...]) -> Any:
        """First resolvable ``(module, symbol)`` pair.

        Needed for the OCCT template instantiations, which the two bindings
        name completely differently: pythonocc keeps the C++ typedef
        (``TopTools.TopTools_IndexedDataMapOfShapeListOfShape``) while OCP
        exposes the mangled template under ``collections``
        (``IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher``).
        """
        tried: list[str] = []
        for module, symbol in candidates:
            try:
                return self.sym(module, symbol)
            except CadBackendError:
                tried.append(f"{module}.{symbol}")
        raise CadBackendError(f"{self.name}: none of {tried} is available")

    def static(self, module: str, cls: str, method: str) -> Any:
        """Resolve a C++ static method across both naming styles.

        OCP exposes ``Cls.method_s``; pythonocc exposes ``Cls.method``, and for
        some packages only a lowercase free-function proxy
        (``brepgprop.SurfaceProperties``). All three are tried.
        """
        mod = self.module(module)
        owner = getattr(mod, cls, None)
        if owner is not None:
            for candidate in (f"{method}_s", method):
                fn = getattr(owner, candidate, None)
                if callable(fn):
                    return fn
        # pythonocc lowercase proxy object, e.g. OCC.Core.BRepGProp.brepgprop
        proxy = getattr(mod, cls.lower(), None)
        if proxy is not None:
            fn = getattr(proxy, method, None)
            if callable(fn):
                return fn
        # pythonocc flattened free function, e.g. brepgprop_SurfaceProperties
        fn = getattr(mod, f"{cls.lower()}_{method}", None)
        if callable(fn):
            return fn
        raise CadBackendError(f"{self.name}: no static {cls}.{method} in {module}")

    # -- shape enumeration ---------------------------------------------------

    _SHAPE_KINDS = (
        "COMPOUND",
        "COMPSOLID",
        "SOLID",
        "SHELL",
        "FACE",
        "WIRE",
        "EDGE",
        "VERTEX",
    )

    def shape_enum(self, kind: str) -> Any:
        kind = kind.upper()
        if kind not in self._SHAPE_KINDS:
            raise ValueError(f"unknown shape kind {kind!r}; expected one of {self._SHAPE_KINDS}")
        enum = self.sym("TopAbs", "TopAbs_ShapeEnum")
        return getattr(enum, f"TopAbs_{kind}")

    def downcast(self, kind: str, shape: Any) -> Any:
        """``TopoDS_Shape`` -> concrete ``TopoDS_Face``/``_Edge``/``_Solid``/..."""
        name = kind.capitalize()
        topods_mod = self.module("TopoDS")
        for owner_name in ("TopoDS", "topods"):
            owner = getattr(topods_mod, owner_name, None)
            if owner is None:
                continue
            for candidate in (name, f"{name}_s"):
                fn = getattr(owner, candidate, None)
                if callable(fn):
                    return fn(shape)
        fn = getattr(topods_mod, f"topods_{name}", None)
        if callable(fn):
            return fn(shape)
        raise CadBackendError(f"{self.name}: no TopoDS downcast for {kind!r}")

    def explore(self, shape: Any, kind: str, downcast: bool = True) -> Iterator[Any]:
        """Yield every sub-shape of *kind*, duplicates included.

        OCCT's explorer visits a shape once per parent, so a face shared by two
        shells comes out twice. Callers that need uniqueness key on
        :meth:`shape_key`.
        """
        explorer_cls = self.sym("TopExp", "TopExp_Explorer")
        exp = explorer_cls(shape, self.shape_enum(kind))
        while exp.More():
            current = exp.Current()
            yield self.downcast(kind, current) if downcast else current
            exp.Next()

    def explore_unique(self, shape: Any, kind: str) -> Iterator[Any]:
        """Like :meth:`explore` but skips shapes already seen (by location+TShape)."""
        seen: set[Any] = set()
        for sub in self.explore(shape, kind):
            key = self.shape_key(sub)
            if key in seen:
                continue
            seen.add(key)
            yield sub

    @staticmethod
    def shape_key(shape: Any) -> int:
        """Stable identity for a ``TopoDS_Shape``.

        ``TopoDS_Shape.__hash__`` hashes the underlying TShape *and* location in
        both bindings, which is exactly the identity OCCT's own maps use. Python
        ``id()`` must **not** be used: the bindings hand out a fresh wrapper
        object per call, so ``id()`` differs for the same topological entity.
        That mistake silently empties any adjacency map built on it.
        """
        return hash(shape)

    # -- diagnostics ---------------------------------------------------------

    def silence(self) -> bool:
        """Mute OCCT's own console chatter.

        OpenCASCADE writes transfer statistics and warnings straight to stdout
        through its global messenger. In a CLI, an API worker or a ROS node
        that output is noise at best and corrupts machine-readable stdout at
        worst, so the default printers are removed. Returns True if the
        messenger was found and cleared.
        """
        try:
            message_cls = self.sym("Message", "Message")
            messenger_fn = getattr(message_cls, "DefaultMessenger_s", None) or (
                message_cls.DefaultMessenger
            )
            messenger = messenger_fn()
            printers = messenger.ChangePrinters()
            printers.Clear()
            return True
        except Exception as exc:  # pragma: no cover - binding specific
            logger.debug("silence() unavailable: %s", exc)
            return False

    # -- geometry primitives -------------------------------------------------

    def read_step(self, path: str) -> Any:
        """Read a STEP file and return the transferred shape (millimetres).

        OCCT converts the file's length unit to its system unit, which defaults
        to millimetres -- so a model authored in metres or inches arrives in mm
        without any scaling on our side.
        """
        reader_cls = self.sym("STEPControl", "STEPControl_Reader")
        ret_done = self.sym("IFSelect", "IFSelect_ReturnStatus").IFSelect_RetDone
        reader = reader_cls()
        status = reader.ReadFile(str(path))
        if status != ret_done:
            raise CadBackendError(f"STEP read failed for {path!r} (status {status})")
        n_roots = reader.NbRootsForTransfer()
        if n_roots <= 0:
            raise CadBackendError(f"STEP file {path!r} contains no transferable roots")
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape is None or shape.IsNull():
            raise CadBackendError(f"STEP file {path!r} transferred to a null shape")
        return shape

    def read_iges(self, path: str) -> Any:
        reader_cls = self.sym("IGESControl", "IGESControl_Reader")
        ret_done = self.sym("IFSelect", "IFSelect_ReturnStatus").IFSelect_RetDone
        reader = reader_cls()
        if reader.ReadFile(str(path)) != ret_done:
            raise CadBackendError(f"IGES read failed for {path!r}")
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape is None or shape.IsNull():
            raise CadBackendError(f"IGES file {path!r} transferred to a null shape")
        return shape

    def bbox(self, shape: Any, optimal: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as ``(min, max)`` in mm.

        ``Bnd_Box.Get()`` is unusable under OCP (unregistered return type), so
        the per-axis getters -- present in both bindings -- are used instead.
        """
        box_cls = self.sym("Bnd", "Bnd_Box")
        box = box_cls()
        adder = self.static("BRepBndLib", "BRepBndLib", "AddOptimal" if optimal else "Add")
        adder(shape, box)
        if box.IsVoid():
            return np.zeros(3), np.zeros(3)
        lo = np.array([box.GetXMin(), box.GetYMin(), box.GetZMin()], dtype=float)
        hi = np.array([box.GetXMax(), box.GetYMax(), box.GetZMax()], dtype=float)
        return lo, hi

    def surface_properties(self, shape: Any) -> tuple[float, np.ndarray]:
        """``(area_mm2, centroid_mm)`` of a face or shell."""
        props_cls = self.sym("GProp", "GProp_GProps")
        props = props_cls()
        self.static("BRepGProp", "BRepGProp", "SurfaceProperties")(shape, props)
        com = props.CentreOfMass()
        return float(props.Mass()), np.array([com.X(), com.Y(), com.Z()], dtype=float)

    def volume_properties(self, shape: Any) -> tuple[float, np.ndarray]:
        """``(volume_mm3, centroid_mm)`` of a solid."""
        props_cls = self.sym("GProp", "GProp_GProps")
        props = props_cls()
        self.static("BRepGProp", "BRepGProp", "VolumeProperties")(shape, props)
        com = props.CentreOfMass()
        return float(props.Mass()), np.array([com.X(), com.Y(), com.Z()], dtype=float)

    def linear_properties(self, shape: Any) -> float:
        """Arc length of an edge or wire, in mm."""
        props_cls = self.sym("GProp", "GProp_GProps")
        props = props_cls()
        self.static("BRepGProp", "BRepGProp", "LinearProperties")(shape, props)
        return float(props.Mass())

    # -- surfaces ------------------------------------------------------------

    _SURFACE_TYPES = {
        "GeomAbs_Plane": "planar",
        "GeomAbs_Cylinder": "cylindrical",
        "GeomAbs_Cone": "conical",
        "GeomAbs_Sphere": "spherical",
        "GeomAbs_Torus": "toroidal",
        "GeomAbs_BezierSurface": "bezier",
        "GeomAbs_BSplineSurface": "bspline",
        "GeomAbs_SurfaceOfRevolution": "revolution",
        "GeomAbs_SurfaceOfExtrusion": "extrusion",
        "GeomAbs_OffsetSurface": "offset",
        "GeomAbs_OtherSurface": "other",
    }

    def surface_adaptor(self, face: Any) -> Any:
        cls = self.sym("BRepAdaptor", "BRepAdaptor_Surface")
        return cls(face, True)

    def surface_type(self, adaptor: Any) -> str:
        """Backend-independent surface-type label."""
        enum = self.sym("GeomAbs", "GeomAbs_SurfaceType")
        value = adaptor.GetType()
        for occ_name, label in self._SURFACE_TYPES.items():
            member = getattr(enum, occ_name, None)
            if member is not None and value == member:
                return label
        return "other"

    _CURVE_TYPES = {
        "GeomAbs_Line": "line",
        "GeomAbs_Circle": "circle",
        "GeomAbs_Ellipse": "ellipse",
        "GeomAbs_Hyperbola": "hyperbola",
        "GeomAbs_Parabola": "parabola",
        "GeomAbs_BezierCurve": "bezier",
        "GeomAbs_BSplineCurve": "bspline",
        "GeomAbs_OffsetCurve": "offset",
        "GeomAbs_OtherCurve": "other",
    }

    def curve_adaptor(self, edge: Any) -> Any:
        cls = self.sym("BRepAdaptor", "BRepAdaptor_Curve")
        return cls(edge)

    def curve_type(self, adaptor: Any) -> str:
        enum = self.sym("GeomAbs", "GeomAbs_CurveType")
        value = adaptor.GetType()
        for occ_name, label in self._CURVE_TYPES.items():
            member = getattr(enum, occ_name, None)
            if member is not None and value == member:
                return label
        return "other"

    def is_reversed(self, shape: Any) -> bool:
        """True if the shape carries ``TopAbs_REVERSED`` orientation.

        A face's stored surface normal points outward only for FORWARD
        orientation; for REVERSED faces it must be negated. Skipping this is
        the single most common cause of a torch pointing into the workpiece.
        """
        orient = self.sym("TopAbs", "TopAbs_Orientation")
        return shape.Orientation() == orient.TopAbs_REVERSED

    # -- maps ----------------------------------------------------------------

    def ancestors_map(self, shape: Any, child: str, parent: str) -> dict[int, list[Any]]:
        """Map each *child* sub-shape to the list of *parent* shapes containing it.

        ``ancestors_map(shape, 'edge', 'face')`` is the face-adjacency relation
        that weld seam recognition is built on.
        """
        map_cls = self.first_sym(
            (
                ("TopTools", "TopTools_IndexedDataMapOfShapeListOfShape"),
                (
                    "collections",
                    "IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher",
                ),
                ("collections", "IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape"),
            )
        )
        occ_map = map_cls()
        self.static("TopExp", "TopExp", "MapShapesAndAncestors")(
            shape, self.shape_enum(child), self.shape_enum(parent), occ_map
        )
        out: dict[int, list[Any]] = {}
        parent_kind = parent.lower()
        for i in range(1, occ_map.Extent() + 1):
            key_shape = occ_map.FindKey(i)
            parents: list[Any] = []
            shape_list = occ_map.FindFromIndex(i)
            for item in self._iter_shape_list(shape_list):
                parents.append(self.downcast(parent_kind, item))
            out[self.shape_key(key_shape)] = parents
        return out

    def _iter_shape_list(self, shape_list: Any) -> Iterator[Any]:
        """Iterate a ``TopTools_ListOfShape`` across both bindings.

        OCP binds the list as a Python sequence; pythonocc requires the OCCT
        list iterator. Python iteration is tried first because it is both
        faster and the common case for the pip install.
        """
        try:
            yield from iter(shape_list)
            return
        except TypeError:
            pass
        iterator_cls = self.first_sym(
            (
                ("TopTools", "TopTools_ListIteratorOfListOfShape"),
                ("collections", "List_TopoDS_Shape"),
            )
        )
        it = iterator_cls(shape_list)
        while it.More():
            yield it.Value()
            it.Next()

    # -- tessellation --------------------------------------------------------

    def tessellate(
        self,
        shape: Any,
        linear_deflection: float = 0.5,
        angular_deflection: float = 0.35,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mesh the shape; returns ``(vertices_mm, triangles)``.

        *linear_deflection* is in millimetres: 0.5 mm is plenty for collision
        checking and display of a weldment, and keeps a 900-face part under a
        few hundred thousand triangles.

        Triangle winding is corrected for REVERSED faces so the resulting mesh
        has consistent outward normals -- needed by the collision checker and
        by any viewer that culls back faces.
        """
        mesher_cls = self.sym("BRepMesh", "BRepMesh_IncrementalMesh")
        brep_tool = self.sym("BRep", "BRep_Tool")
        loc_cls = self.sym("TopLoc", "TopLoc_Location")

        # Drop any triangulation the STEP already carried, so our deflection wins.
        with contextlib.suppress(CadBackendError):  # pragma: no cover
            self.static("BRepTools", "BRepTools", "Clean")(shape)

        mesher_cls(shape, linear_deflection, False, angular_deflection, True)

        verts: list[np.ndarray] = []
        tris: list[np.ndarray] = []
        offset = 0

        triangulation_fn = getattr(brep_tool, "Triangulation_s", None) or brep_tool.Triangulation

        for face in self.explore(shape, "face"):
            loc = loc_cls()
            try:
                tri = triangulation_fn(face, loc)
            except TypeError:  # pragma: no cover - signature variant
                tri = triangulation_fn(face, loc, 0)
            if tri is None:
                continue

            n_nodes = tri.NbNodes()
            n_tris = tri.NbTriangles()
            if n_nodes == 0 or n_tris == 0:
                continue

            trsf = loc.Transformation()
            face_verts = np.empty((n_nodes, 3), dtype=float)
            for i in range(1, n_nodes + 1):
                p = tri.Node(i).Transformed(trsf)
                face_verts[i - 1] = (p.X(), p.Y(), p.Z())

            face_tris = np.empty((n_tris, 3), dtype=np.int64)
            for i in range(1, n_tris + 1):
                a, b, c = tri.Triangle(i).Get()
                face_tris[i - 1] = (a - 1, b - 1, c - 1)

            if self.is_reversed(face):
                face_tris = face_tris[:, ::-1]

            verts.append(face_verts)
            tris.append(face_tris + offset)
            offset += n_nodes

        if not verts:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        return np.vstack(verts), np.vstack(tris)

    def tessellate_by_face(
        self,
        shape: Any,
        linear_deflection: float = 0.5,
        angular_deflection: float = 0.35,
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Mesh the shape, keeping each face separate.

        Returns ``(face_key, vertices, triangles)`` per face. The combined
        mesh from :meth:`tessellate` loses which CAD face each triangle came
        from, and that mapping is what lets an operator click a face in a 3-D
        view and have the application know *which* face was picked -- the
        basis of tracing a weld line by hand.
        """
        mesher_cls = self.sym("BRepMesh", "BRepMesh_IncrementalMesh")
        brep_tool = self.sym("BRep", "BRep_Tool")
        loc_cls = self.sym("TopLoc", "TopLoc_Location")

        with contextlib.suppress(CadBackendError):
            self.static("BRepTools", "BRepTools", "Clean")(shape)
        mesher_cls(shape, linear_deflection, False, angular_deflection, True)

        triangulation_fn = getattr(brep_tool, "Triangulation_s", None) or brep_tool.Triangulation
        out: list[tuple[int, np.ndarray, np.ndarray]] = []

        for face in self.explore_unique(shape, "face"):
            loc = loc_cls()
            try:
                tri = triangulation_fn(face, loc)
            except TypeError:  # pragma: no cover - signature variant
                tri = triangulation_fn(face, loc, 0)
            if tri is None or tri.NbNodes() == 0 or tri.NbTriangles() == 0:
                continue

            trsf = loc.Transformation()
            n_nodes = tri.NbNodes()
            vertices = np.empty((n_nodes, 3), dtype=float)
            for i in range(1, n_nodes + 1):
                point = tri.Node(i).Transformed(trsf)
                vertices[i - 1] = (point.X(), point.Y(), point.Z())

            n_tris = tri.NbTriangles()
            triangles = np.empty((n_tris, 3), dtype=np.int64)
            for i in range(1, n_tris + 1):
                a, b, c = tri.Triangle(i).Get()
                triangles[i - 1] = (a - 1, b - 1, c - 1)
            if self.is_reversed(face):
                triangles = triangles[:, ::-1]

            out.append((self.shape_key(face), vertices, triangles))
        return out


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

_cached: CadBackend | None = None


def get_backend(prefer: str | None = None) -> CadBackend:
    """Return the active backend, importing it on first use.

    Parameters
    ----------
    prefer
        ``'OCP'`` or ``'pythonocc-core'`` to force a binding; otherwise the
        first importable one wins (OCP before pythonocc, because a pip
        environment is the more common case for the CAD-side tooling).
    """
    global _cached
    if _cached is not None and (prefer is None or _cached.name == prefer):
        return _cached

    candidates = _BINDINGS
    if prefer is not None:
        candidates = tuple(b for b in _BINDINGS if b[0] == prefer)
        if not candidates:
            raise CadBackendError(f"unknown backend preference {prefer!r}")

    for name, prefix in candidates:
        if _try_import(prefix):
            logger.info("CAD backend: %s (%s)", name, prefix)
            _cached = CadBackend(name=name, prefix=prefix)
            # OCCT's console printers must go before any reader runs, or STEP
            # transfer statistics end up interleaved with our own output.
            _cached.silence()
            return _cached

    raise CadBackendError(
        "No OpenCASCADE binding found. Install one of:\n"
        "  pip install cadquery-ocp          # OCP, works everywhere\n"
        "  conda install -c conda-forge pythonocc-core"
    )


def backend_available(prefer: str | None = None) -> bool:
    """True if a backend can be loaded -- for skipping tests and degrading the UI."""
    try:
        get_backend(prefer)
        return True
    except CadBackendError:
        return False


def backend_name() -> str | None:
    try:
        return get_backend().name
    except CadBackendError:
        return None
