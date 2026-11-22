"""Translate a welding pass into setpoints a specific power source understands.

A procedure says "240 A, 27 V, 8.4 m/min, spray". No machine takes that
literally:

* a **generic analog** source wants two scaled references on two analog
  outputs;
* a **Fronius TPS/i** is synergic -- the robot picks a stored job and commands
  wire feed, and current and voltage *follow* from the characteristic, so
  sending them would be meaningless;
* a **Lincoln Power Wave** wants a weld-mode number and expresses arc length
  as a trim *multiplier* about 1.00 rather than a signed correction.

Each of those is a YAML profile in ``data/machines/``; this module is the
engine that reads one and does the scaling, the job lookup and the limit
checks. Adding a brand is a YAML file, not a code change -- which matters when
a cell is retrofitted and the machine on the floor is not the machine in the
drawing.

Values that fall outside a machine's declared limits are **clamped and
reported**, never silently truncated: a setpoint quietly rounded down is a
weld quietly made wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..model import MachineSetpoints, TransferMode, WeldPass, WeldProcedure
from .tables import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "SignalScale",
    "JobRule",
    "MachineProfile",
    "load_machine",
    "available_machines",
    "setpoints_for_pass",
    "setpoints_for_procedure",
]


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------


@dataclass
class SignalScale:
    """Linear map from an engineering value to a fieldbus raw word."""

    signal: str
    value_min: float
    value_max: float
    raw_min: int
    raw_max: int
    unit: str = ""
    read_only: bool = False
    # Some vendors express a signed correction as an offset about a midpoint
    # (Lincoln's trim is 1000 = nominal). When set, the raw range is centred
    # on this instead of spanning raw_min..raw_max linearly from value_min.
    offset_raw: int | None = None

    def to_raw(self, value: float) -> tuple[int, bool]:
        """``(raw_word, was_clamped)``."""
        span = self.value_max - self.value_min
        if abs(span) < 1e-12:
            return int(self.raw_min), False
        clamped = float(np.clip(value, self.value_min, self.value_max))
        was_clamped = not np.isclose(clamped, value, rtol=1e-9, atol=1e-9)
        fraction = (clamped - self.value_min) / span
        raw = self.raw_min + fraction * (self.raw_max - self.raw_min)
        return int(round(raw)), was_clamped

    def from_raw(self, raw: int) -> float:
        """Inverse map -- used when reading a machine's monitor words back."""
        raw_span = self.raw_max - self.raw_min
        if raw_span == 0:
            return self.value_min
        fraction = (raw - self.raw_min) / raw_span
        return self.value_min + fraction * (self.value_max - self.value_min)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalScale:
        return cls(
            signal=str(data.get("signal", "")),
            value_min=float(data.get("value_min", 0.0)),
            value_max=float(data.get("value_max", 1.0)),
            raw_min=int(data.get("raw_min", 0)),
            raw_max=int(data.get("raw_max", 32767)),
            unit=str(data.get("unit", "")),
            read_only=bool(data.get("read_only", False)),
            offset_raw=(int(data["offset_raw"]) if data.get("offset_raw") is not None else None),
        )


# ---------------------------------------------------------------------------
# job lookup
# ---------------------------------------------------------------------------


@dataclass
class JobRule:
    """One row of a machine's synergic job table."""

    job: int
    wire_diameter_mm: float | None = None
    gas: str | None = None
    group: str | None = None
    mode: TransferMode | None = None

    def matches(
        self,
        wire_diameter_mm: float,
        gas: str,
        group: str,
        mode: TransferMode,
    ) -> bool:
        if self.wire_diameter_mm is not None and not np.isclose(
            self.wire_diameter_mm, wire_diameter_mm, atol=0.05
        ):
            return False
        if self.gas is not None and self.gas != gas:
            return False
        if self.group is not None and self.group != group:
            return False
        return not (self.mode is not None and self.mode is not mode)


@dataclass
class MachineProfile:
    """One power source: its interface, scaling, job table and limits."""

    name: str
    vendor: str
    interface: str
    key: str = ""
    description: str = ""
    job_mode: str = "none"
    jobs: list[JobRule] = field(default_factory=list)
    fallback_job: int | None = None
    job_scale: SignalScale | None = None
    analog: dict[str, SignalScale] = field(default_factory=dict)
    corrections: dict[str, SignalScale] = field(default_factory=dict)
    digital: dict[str, str] = field(default_factory=dict)
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def is_synergic(self) -> bool:
        """True when the machine derives current and voltage from a stored job.

        On such a machine the procedure's current and voltage are *predictions*
        of what the characteristic will produce, not commands -- so they are
        reported for the record and the wire feed is what actually goes out.
        """
        return self.job_mode in ("synergic_job", "weld_mode")

    def select_job(
        self, wire_diameter_mm: float, gas: str, group: str, mode: TransferMode
    ) -> tuple[int | None, str]:
        """``(job_number, note)``; first matching rule wins."""
        if not self.is_synergic:
            return None, ""
        for rule in self.jobs:
            if rule.matches(wire_diameter_mm, gas, group, mode):
                return rule.job, ""
        if self.fallback_job is not None:
            return (
                self.fallback_job,
                f"no job for {wire_diameter_mm:.1f} mm / {gas} / {group} / "
                f"{mode.value}: fell back to job {self.fallback_job}, verify on the machine",
            )
        return None, (
            f"no job matches {wire_diameter_mm:.1f} mm / {gas} / {group} / {mode.value} "
            "and the profile declares no fallback"
        )

    def check_limits(self, weld_pass: WeldPass) -> list[str]:
        """Warnings for any pass value outside the machine's capability."""
        notes: list[str] = []
        checks = (
            ("current_a", weld_pass.current_a, "A"),
            ("voltage_v", weld_pass.voltage_v, "V"),
            ("wire_speed_m_min", weld_pass.wire_speed_m_min, "m/min"),
        )
        for key, value, unit in checks:
            bounds = self.limits.get(key)
            if bounds is None or value <= 0.0:
                continue
            lo, hi = bounds
            if value < lo:
                notes.append(
                    f"{key.replace('_', ' ')} {value:.1f} {unit} is below the "
                    f"{self.name} minimum of {lo:.1f} {unit}"
                )
            elif value > hi:
                notes.append(
                    f"{key.replace('_', ' ')} {value:.1f} {unit} exceeds the "
                    f"{self.name} maximum of {hi:.1f} {unit}"
                )
        return notes


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

_machine_cache: dict[str, MachineProfile] = {}


def load_machine(name: str, data_dir: Path | None = None) -> MachineProfile:
    """Load ``data/machines/<name>.yaml``."""
    if name in _machine_cache:
        return _machine_cache[name]

    base = Path(data_dir) if data_dir else DATA_DIR
    path = base / "machines" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"machine profile {name!r} not found at {path}. "
            f"Available: {sorted(available_machines(base))}"
        )
    raw = yaml.safe_load(path.read_text()) or {}
    selection = raw.get("job_selection", {}) or {}

    jobs: list[JobRule] = []
    for item in selection.get("jobs", []) or []:
        mode = item.get("mode")
        try:
            parsed_mode = TransferMode(mode) if mode else None
        except ValueError:
            logger.warning("machine %s: unknown transfer mode %r", name, mode)
            parsed_mode = None
        jobs.append(
            JobRule(
                job=int(item["job"]),
                wire_diameter_mm=(
                    float(item["wire_diameter_mm"])
                    if item.get("wire_diameter_mm") is not None
                    else None
                ),
                gas=item.get("gas"),
                group=item.get("group"),
                mode=parsed_mode,
            )
        )

    job_scale = None
    if selection.get("job_signal"):
        job_scale = SignalScale(
            signal=str(selection["job_signal"]),
            value_min=float(selection.get("job_raw_min", 0)),
            value_max=float(selection.get("job_raw_max", 99)),
            raw_min=int(selection.get("job_raw_min", 0)),
            raw_max=int(selection.get("job_raw_max", 99)),
        )

    profile = MachineProfile(
        name=str(raw.get("name", name)),
        vendor=str(raw.get("vendor", "")),
        interface=str(raw.get("interface", "")),
        key=name,
        description=str(raw.get("description", "")).strip(),
        job_mode=str(selection.get("mode", "none")),
        jobs=jobs,
        fallback_job=(
            int(selection["fallback_job"]) if selection.get("fallback_job") is not None else None
        ),
        job_scale=job_scale,
        analog={
            key: SignalScale.from_dict(value) for key, value in (raw.get("analog") or {}).items()
        },
        corrections={
            key: SignalScale.from_dict(value)
            for key, value in (raw.get("corrections") or {}).items()
        },
        digital={str(k): str(v) for k, v in (raw.get("digital") or {}).items()},
        limits={str(k): (float(v[0]), float(v[1])) for k, v in (raw.get("limits") or {}).items()},
    )
    _machine_cache[name] = profile
    logger.info(
        "loaded machine profile %s (%s, %s)", profile.name, profile.vendor, profile.interface
    )
    return profile


def available_machines(data_dir: Path | None = None) -> list[str]:
    base = Path(data_dir) if data_dir else DATA_DIR
    folder = base / "machines"
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml"))


# ---------------------------------------------------------------------------
# translation
# ---------------------------------------------------------------------------


def setpoints_for_pass(
    weld_pass: WeldPass,
    profile: MachineProfile,
    material_group: str = "carbon_steel",
    arc_length_correction: float = 0.0,
    dynamic_correction: float = 0.0,
) -> MachineSetpoints:
    """Build the machine setpoints for one pass."""
    notes: list[str] = []
    notes.extend(profile.check_limits(weld_pass))

    job, job_note = profile.select_job(
        weld_pass.wire_diameter_mm, weld_pass.gas, material_group, weld_pass.transfer_mode
    )
    if job_note:
        notes.append(job_note)

    setpoints = MachineSetpoints(
        pass_index=weld_pass.index,
        job_number=job,
        current_a=weld_pass.current_a,
        voltage_v=weld_pass.voltage_v,
        wire_speed_m_min=weld_pass.wire_speed_m_min,
        arc_length_correction=arc_length_correction,
        dynamic_correction=dynamic_correction,
        gas_flow_l_min=weld_pass.gas_flow_l_min,
        pulse_frequency_hz=0.0,
        machine_profile=profile.key or profile.name,
        notes=notes,
    )

    if profile.is_synergic:
        notes.append(
            f"{profile.name} is synergic: job {job} and wire feed command the arc; "
            "the current and voltage shown are what the characteristic should produce"
        )

    # --- analog words -------------------------------------------------------
    sources = {
        "current": weld_pass.current_a,
        "voltage": weld_pass.voltage_v,
        "wire_speed": weld_pass.wire_speed_m_min,
        "gas_flow": weld_pass.gas_flow_l_min,
    }
    for key, scale in profile.analog.items():
        if scale.read_only:
            continue
        value = sources.get(key)
        if value is None:
            continue
        raw, clamped = scale.to_raw(value)
        setpoints.analog_words[scale.signal] = raw
        if clamped:
            notes.append(
                f"{key} {value:.1f} clamped to the {scale.value_min:.0f}-"
                f"{scale.value_max:.0f} range of {scale.signal}"
            )

    # --- corrections --------------------------------------------------------
    correction_values = {
        "arc_length": arc_length_correction,
        "dynamic": dynamic_correction,
    }
    for key, scale in profile.corrections.items():
        value = correction_values.get(key, 0.0)
        if scale.offset_raw is not None:
            # Vendor expresses the correction about a midpoint (Lincoln trim).
            span = scale.value_max - scale.value_min
            half_raw = (scale.raw_max - scale.raw_min) / 2.0
            raw = scale.offset_raw + (value / (span / 2.0)) * half_raw if span else scale.offset_raw
            setpoints.analog_words[scale.signal] = int(round(raw))
        else:
            raw, clamped = scale.to_raw(value)
            setpoints.analog_words[scale.signal] = raw
            if clamped:
                notes.append(f"{key} correction {value:+.1f} clamped for {scale.signal}")

    # --- job word -----------------------------------------------------------
    if profile.job_scale is not None and job is not None:
        setpoints.analog_words[profile.job_scale.signal] = int(job)

    # --- discrete outputs ---------------------------------------------------
    # The arc-on bit is deliberately left False here: it is asserted by the
    # execution layer at the seam's start point, not by the planner.
    for role, signal in profile.digital.items():
        setpoints.digital_bits[signal] = role == "robot_ready"

    return setpoints


def setpoints_for_procedure(
    procedure: WeldProcedure,
    profile: MachineProfile,
    material_group: str = "carbon_steel",
) -> list[MachineSetpoints]:
    """Setpoints for every pass of a procedure, in pass order."""
    return [
        setpoints_for_pass(weld_pass, profile, material_group) for weld_pass in procedure.passes
    ]


def clear_cache() -> None:
    _machine_cache.clear()
