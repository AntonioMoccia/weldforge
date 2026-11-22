"""Loading and interpolation of welding procedure data.

The numbers themselves live in ``data/procedures/*.yaml`` so a welding
engineer can adjust them -- after a procedure qualification, or for a different
consumable -- without touching code. This module turns them into typed objects
and handles the one piece of real maths: interpolating a synergic line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..model import Process, TransferMode, WeldPosition

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_DIR",
    "WireData",
    "ProcedureData",
    "MaterialData",
    "load_procedure",
    "load_materials",
    "available_procedures",
    "clear_cache",
]


def _default_data_dir() -> Path:
    """Locate ``data/`` whether running from a checkout or an installed wheel."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "procedures"
        if candidate.is_dir():
            return parent / "data"
    return here.parents[3] / "data"


DATA_DIR = _default_data_dir()


# ---------------------------------------------------------------------------
# wire / synergic line
# ---------------------------------------------------------------------------


@dataclass
class WireData:
    """One wire diameter and its synergic line."""

    diameter_mm: float
    min_current_a: float
    max_current_a: float
    modes: list[TransferMode] = field(default_factory=list)
    # Sorted by current: (current A, voltage V, wire feed m/min)
    curve: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def cross_section_mm2(self) -> float:
        return float(np.pi * (self.diameter_mm / 2.0) ** 2)

    def supports(self, mode: TransferMode) -> bool:
        return not self.modes or mode in self.modes

    def voltage_for_current(self, current_a: float) -> float:
        """Arc voltage along the synergic line, V."""
        return self._interp(current_a, column=1)

    def wire_speed_for_current(self, current_a: float) -> float:
        """Wire feed along the synergic line, m/min."""
        return self._interp(current_a, column=2)

    def current_for_wire_speed(self, wire_speed_m_min: float) -> float:
        """Invert the synergic line: what current gives this wire feed?

        This is the direction the procedure engine actually needs. Travel speed
        follows from the deposit cross-section, cross-section fixes the wire
        feed, and only then is the current known -- not the other way round.
        """
        if len(self.curve) == 0:
            return 0.0
        speeds = self.curve[:, 2]
        currents = self.curve[:, 0]
        clamped = float(np.clip(wire_speed_m_min, speeds[0], speeds[-1]))
        return float(np.interp(clamped, speeds, currents))

    def clamp_current(self, current_a: float) -> tuple[float, bool]:
        """``(current, was_clamped)`` within this wire's working range."""
        clamped = float(np.clip(current_a, self.min_current_a, self.max_current_a))
        return clamped, not np.isclose(clamped, current_a, rtol=1e-6)

    def _interp(self, current_a: float, column: int) -> float:
        if len(self.curve) == 0:
            return 0.0
        return float(np.interp(current_a, self.curve[:, 0], self.curve[:, column]))


# ---------------------------------------------------------------------------
# procedure
# ---------------------------------------------------------------------------


def _position_map(raw: dict | None, default: float) -> dict[WeldPosition, float]:
    out = {p: default for p in WeldPosition}
    for key, value in (raw or {}).items():
        try:
            out[WeldPosition(key)] = float(value)
        except ValueError:
            logger.warning("unknown welding position %r in procedure data", key)
    return out


@dataclass
class ProcedureData:
    """A complete procedure data set: one process, one material group."""

    name: str
    process: Process
    base_material_group: str
    filler: str
    gas: str
    gas_flow_l_min: float
    deposition_efficiency: float
    thermal_efficiency: float
    wires: list[WireData]
    short_arc_max_current_a: float
    out_of_position_modes: list[TransferMode]
    target_travel_speed: dict[WeldPosition, float]
    travel_speed_limits: tuple[float, float]
    max_leg_per_pass: dict[WeldPosition, float]
    torch_angles: dict[str, tuple[float, float]]
    weave: dict[WeldPosition, dict[str, Any]]
    stickout: dict[TransferMode, float]
    gap_travel_factor_per_mm: float
    max_gap_mm: float

    def wire(self, diameter_mm: float) -> WireData:
        """The wire of this diameter, or the nearest available."""
        if not self.wires:
            raise ValueError(f"procedure {self.name!r} declares no wires")
        exact = next(
            (w for w in self.wires if np.isclose(w.diameter_mm, diameter_mm, atol=1e-6)),
            None,
        )
        if exact is not None:
            return exact
        nearest = min(self.wires, key=lambda w: abs(w.diameter_mm - diameter_mm))
        logger.info(
            "procedure %s has no %.1f mm wire, using %.1f mm",
            self.name,
            diameter_mm,
            nearest.diameter_mm,
        )
        return nearest

    def default_wire(self, thickness_mm: float) -> WireData:
        """Pick a wire size for a member thickness.

        Shop practice: 0.8 mm for sheet, 1.0 mm up to about 6 mm, 1.2 mm for
        general structural work, 1.6 mm only on heavy section where its deposit
        rate pays for the extra heat.
        """
        if thickness_mm <= 0.0:
            return self.wire(1.2)
        if thickness_mm < 2.0:
            return self.wire(0.8)
        if thickness_mm < 6.0:
            return self.wire(1.0)
        if thickness_mm <= 20.0:
            return self.wire(1.2)
        return self.wire(1.6)

    def torch_angle(self, joint_type: str) -> tuple[float, float]:
        """``(work_angle_deg, travel_angle_deg)`` for a joint type."""
        return self.torch_angles.get(joint_type, self.torch_angles.get("unknown", (45.0, -10.0)))

    def weave_for(self, position: WeldPosition) -> dict[str, Any]:
        return self.weave.get(position, {})


def _parse_modes(raw: Any) -> list[TransferMode]:
    out = []
    for item in raw or []:
        try:
            out.append(TransferMode(item))
        except ValueError:
            logger.warning("unknown transfer mode %r in procedure data", item)
    return out


_procedure_cache: dict[str, ProcedureData] = {}


def load_procedure(name: str, data_dir: Path | None = None) -> ProcedureData:
    """Load ``data/procedures/<name>.yaml``."""
    if name in _procedure_cache:
        return _procedure_cache[name]

    base = Path(data_dir) if data_dir else DATA_DIR
    path = base / "procedures" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"procedure data {name!r} not found at {path}. "
            f"Available: {sorted(available_procedures(base))}"
        )
    raw = yaml.safe_load(path.read_text()) or {}

    wires = []
    for item in raw.get("wires", []):
        curve = np.array(item.get("curve", []), dtype=float).reshape(-1, 3)
        curve = curve[np.argsort(curve[:, 0])] if len(curve) else curve
        wires.append(
            WireData(
                diameter_mm=float(item["diameter_mm"]),
                min_current_a=float(item.get("min_current_a", 0.0)),
                max_current_a=float(item.get("max_current_a", 1e4)),
                modes=_parse_modes(item.get("modes")),
                curve=curve,
            )
        )
    wires.sort(key=lambda w: w.diameter_mm)

    selection = raw.get("mode_selection", {}) or {}
    limits = raw.get("travel_speed_limits_mm_s", [2.0, 20.0])
    gap = raw.get("gap_compensation", {}) or {}

    weave: dict[WeldPosition, dict[str, Any]] = {}
    for key, value in (raw.get("weave") or {}).items():
        try:
            weave[WeldPosition(key)] = dict(value)
        except ValueError:
            logger.warning("unknown welding position %r in weave data", key)

    stickout: dict[TransferMode, float] = {}
    for key, value in (raw.get("stickout_mm") or {}).items():
        try:
            stickout[TransferMode(key)] = float(value)
        except ValueError:
            logger.warning("unknown transfer mode %r in stickout data", key)

    data = ProcedureData(
        name=name,
        process=Process(raw.get("process", "GMAW")),
        base_material_group=str(raw.get("base_material_group", "carbon_steel")),
        filler=str(raw.get("filler", "")),
        gas=str(raw.get("gas", "")),
        gas_flow_l_min=float(raw.get("gas_flow_l_min", 15.0)),
        deposition_efficiency=float(raw.get("deposition_efficiency", 0.95)),
        thermal_efficiency=float(raw.get("thermal_efficiency", 0.8)),
        wires=wires,
        short_arc_max_current_a=float(selection.get("short_arc_max_current_a", 180.0)),
        out_of_position_modes=_parse_modes(selection.get("out_of_position_modes")),
        target_travel_speed=_position_map(raw.get("target_travel_speed_mm_s"), 6.0),
        travel_speed_limits=(float(limits[0]), float(limits[1])),
        max_leg_per_pass=_position_map(raw.get("max_leg_per_pass_mm"), 6.0),
        torch_angles={
            key: (float(value.get("work", 45.0)), float(value.get("travel", -10.0)))
            for key, value in (raw.get("torch_angles") or {}).items()
        },
        weave=weave,
        stickout=stickout,
        gap_travel_factor_per_mm=float(gap.get("per_mm_travel_factor", -0.1)),
        max_gap_mm=float(gap.get("max_gap_mm", 4.0)),
    )
    _procedure_cache[name] = data
    logger.info("loaded procedure %s: %d wire sizes", name, len(wires))
    return data


def available_procedures(data_dir: Path | None = None) -> list[str]:
    base = Path(data_dir) if data_dir else DATA_DIR
    folder = base / "procedures"
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml") if p.stem != "materials")


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------


@dataclass
class MaterialData:
    """Base material properties that constrain the procedure."""

    name: str
    group: str
    procedure: str
    yield_mpa: float
    cev: float
    density_kg_mm3: float
    heat_input_max_kj_mm: float
    interpass_max_c: float
    # Ascending (thickness_mm, preheat_C) bands.
    preheat_bands: list[tuple[float, float]] = field(default_factory=list)

    def preheat_for(self, thickness_mm: float) -> float:
        """Preheat temperature for a combined thickness, degrees C.

        Banded form of EN 1011-2: thicker sections cool faster and need
        preheat to keep the hydrogen-cracking risk down. The bands already
        include the grade's carbon equivalent, so no extra CEV term is applied
        on top.
        """
        for limit, temperature in self.preheat_bands:
            if thickness_mm <= limit:
                return float(temperature)
        return float(self.preheat_bands[-1][1]) if self.preheat_bands else 20.0


_material_cache: dict[str, MaterialData] | None = None


def load_materials(data_dir: Path | None = None) -> dict[str, MaterialData]:
    """Load ``data/procedures/materials.yaml``."""
    global _material_cache
    if _material_cache is not None:
        return _material_cache

    base = Path(data_dir) if data_dir else DATA_DIR
    path = base / "procedures" / "materials.yaml"
    if not path.exists():
        raise FileNotFoundError(f"materials data not found at {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    out: dict[str, MaterialData] = {}
    for name, item in (raw.get("materials") or {}).items():
        out[str(name)] = MaterialData(
            name=str(name),
            group=str(item.get("group", "carbon_steel")),
            procedure=str(item.get("procedure", "gmaw_carbon_steel")),
            yield_mpa=float(item.get("yield_mpa", 235.0)),
            cev=float(item.get("cev", 0.4)),
            density_kg_mm3=float(item.get("density_kg_mm3", 7.85e-6)),
            heat_input_max_kj_mm=float(item.get("heat_input_max_kj_mm", 3.0)),
            interpass_max_c=float(item.get("interpass_max_c", 250.0)),
            preheat_bands=[(float(a), float(b)) for a, b in item.get("preheat_bands_mm_c", [])],
        )
    _material_cache = out
    logger.info("loaded %d base materials", len(out))
    return out


def clear_cache() -> None:
    """Forget loaded data -- used by tests and after editing the YAML."""
    global _material_cache
    _procedure_cache.clear()
    _material_cache = None
