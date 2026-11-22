# Architecture

## The pipeline

```
                    weldforge.cad
CAD file  ─────────────────────────────▶  Part
  STEP / IGES / STL                        solids, faces, edges, mesh,
  OCCT or trimesh                          measured member thicknesses

                    weldforge.seams
Part  ─────────────────────────────────▶  Seam[]
  two detectors, chaining,                 polyline, tangents, flank normals,
  classification                           wire direction, groove angle,
                                           joint type, confidence

                    weldforge.wps
Seam  ─────────────────────────────────▶  WeldProcedure
  tables, pass planning,                   passes with current, voltage, wire
  volume conservation                      feed, travel speed, angles, weave
                                       ─▶  MachineSetpoints (per power source)

                    weldforge.fixture
Part + Seam[] + WeldCell  ─────────────▶  PlacementCandidate
  stable poses, constraint scoring          pose, score, position mix,
                                            violations

                    weldforge.path + weldforge.kin + weldforge.collide
Seam[] + WeldProcedure + Placement  ───▶  TorchPath[]
  torch poses, seam order, IK,              poses with joint solutions,
  collision checks                          approach and retract, flags

                    weldforge.sim
TorchPath[]  ──────────────────────────▶  JobReport
                                           reachable, collision-free, cycle
                                           time, wire, gas, per-seam detail

                    weldforge.post
WeldJob  ──────────────────────────────▶  robot program
                                           RAPID / KRL / URScript / Fanuc / ROS 2
```

`weldforge.plan.pipeline` only wires these together and records timings. Every
stage is independently usable, which is what makes the API's stage switches and
the CLI's separate subcommands possible.

## Units

**Millimetres, radians, seconds** everywhere inside. Conversions happen at
exactly two boundaries and nowhere else:

| Boundary | File | Converts |
|---|---|---|
| ROS 2 | `ros2_ws/src/weldforge_cell/conversions.py` | mm→m, m/min→m/s, kJ/mm→J/m, deg→rad |
| URScript | `src/weldforge/post/urscript.py` | mm→m |

`src/weldforge/units.py` holds the converters and the rationale. The previous
generations of this software mixed metres and millimetres across module
boundaries; that bug class is why this is enforced rather than conventional.

## Seam recognition

### Why two detectors

`detect_topological_seams` finds seams along **shared manifold edges** — the
case where the joint is visible in the B-rep. Correct for a single bent or
machined body, and for a weldment booleaned into one solid.

`detect_contact_seams` finds seams by **contact between bodies**. This is the
case that matters: on the 58-body weldment this was built against, all 2266
edges are internal to a single body, so the topological detector finds nothing
at all.

### How a contact becomes a seam

The weld does **not** run between the two faces that touch:

```
         web                        the contact is between
          ║                         base-top and web-bottom
          ║                         (two coincident planes,
    ══════╬══════                    opposed normals)
      base plate
                                    the WELDS run along the
     ↘    ║    ↙                    boundary of that footprint,
      seam   seam                   each between the base's top
                                    face and a SIDE face of the web
```

So: find the contacting pair, walk the landing face's boundary edges, keep the
stretches that really lie on the receiving face, and form a seam between the
receiving face and the third face adjacent to each boundary edge.

A T-joint footprint yields four seams — two sides and two ends — which is
exactly what gets welded, unless the ends are flush.

### Why trim probing cannot be used for contact joints

The base plate's top face continues **underneath** the web, so stepping either
way off the seam lands on material and "which side is the face on" has no
answer. The in-face directions instead come from projecting each flank's
outward normal into the other's tangent plane, which is exact for the concave
joints contact always produces.

That rule assumes both faces continue past the seam — true for a landing, false
where two members finish flush. So the assumption is **validated**: step along
each computed direction and check the point is still on the face. Without that
check, the detector reports a confident 90° fillet down the outside of every L
and across every flush web end.

### The sign work

| `dot(n_a, d_b)` | material angle | groove (weld-side) angle |
|---|---|---|
| `< 0` convex | `phi` | `360° − phi` |
| `> 0` concave | `360° − phi` | `phi` |
| `≈ 0` tangent | 180° | 180° |

Derived from a wedge of material filling `[0, theta]` about the seam axis;
the derivation is in the `seams/dihedral.py` docstring and tested against every
wedge angle from 5° to 355°.

Wire direction is `−normalise(n_a + n_b)`: the outward normals sum to the
outward bisector of the weld-side wedge, so its negation points from the air
into the joint root. Right for fillets, butts, laps and outside corners alike.

### Local thickness

Member thickness drives the fillet leg, the pass count and the heat-input
limit, so it is measured **at the seam**, not taken from the body:

```
gauge = min( ray cast along the inward normal,
             the flank face's own narrow width )
```

Neither alone is right. The ray is correct when the flank is a plate's *skin*;
cast from a plate's *edge* face it measures the plate's width instead — 100 mm
for an 8 mm plate on the lap specimen. The face's narrow width is correct in
exactly that case and wrong for a skin. Each over-estimates where the other
works, so the minimum is right in both, and it fails safe: an under-estimated
thickness yields a smaller fillet and less heat.

Measured at three stations and taking the median, so one station landing badly
cannot move the answer.

## Welding procedure

Travel speed and wire feed come from one equation, not two tables:

```
wire_area × wire_feed × deposition_efficiency  =  deposit_area × travel_speed
```

Pick the travel speed the welding position allows → the wire feed follows → the
current follows by inverting the synergic line → the voltage follows from the
current. If the current falls outside the wire's range it is clamped and the
travel speed **re-solved**, because the deposit must stay right and it is the
cycle time that gives. The clamp is reported.

Pass count divides **area**, not leg. A 12 mm fillet split in two leaves each
pass carrying 8.5 mm of leg, not 6 — area grows as the square. And since the
root is deliberately run small, passes are added until the *largest* one fits
the bead the position can hold.

Weaving is left to the controller. Every arc-welding controller executes a
weave natively; interpolating it into the path needs a station every 0.2 mm —
over a thousand points for one pass — fixes the amplitude at plan time so the
welder cannot trim it, and loses the dwell the weave exists to provide.

## Placement

Six weighted criteria, reported separately so the score can be explained:
reachability, welding position, torch access, clamp clearance, robot envelope,
table fit. Hard constraints (off the table, overweight, inside the robot base)
reject a candidate and are listed as violations, because a violated placement
is not merely worse — it is not an option.

Orientation is chosen first, from the part's own **stable resting poses**
(convex-hull facets whose support polygon contains the projected centre of
mass, so they are the ways it will actually sit on a bench), then position is
refined by a grid and a pattern search. The grid is forced **odd** so the table
centre is always a candidate: an even grid samples only the extremes, which on
a bench with corner clamps means every candidate is planted against a clamp.

Reachability is scored with the **closed-form IK only**. The numeric fallback
costs a hundred times more per call and is spent entirely on targets that are
out of reach — which is most of them during a search, and "out of reach" is all
the score needs. That change took a placement search from 80 s to 2 s.

## Positioner

Solved as **indexed** positioning: the positioner moves to a setting, stops,
the robot welds, it moves on. That is how most two-axis cells run discrete
seams and it is what the kinematics here can honestly promise.

Axes compose left to right in declaration order:

```
T_cell_part  =  A1(v1) @ A2(v2) @ ... @ T_mount_part
```

Each `Ai` is defined by its **nominal** axis and origin in the cell frame at
the zero position; composition then carries the later axes correctly, so at
`v1 = 90 deg` the second axis's effective direction in the cell is
`A1 . axis2` -- which is what physically happens when a tilt table carries a
rotary faceplate.

Search is a coarse grid per axis, always including zero, then a pattern
refinement. Scoring weights **reachability above welding position** (3 to 1),
because a beautifully presented seam the arm cannot touch is worth nothing,
and charges a continuity cost per radian of index travel — including from
home, so the first seam of a job does not swing the positioner for nothing.

Fully **coordinated** motion is deliberately not generated. It needs the
controller's own option (RAPID `MultiMove`, KUKA `#EX_AX`) and a
velocity-level solution across nine axes; an approximation drags the torch off
the joint. Seams whose welding position changes along their own length are
**reported** as needing it.

The index travel is charged to the cycle time. A quoted cycle that ignores a
two-axis positioner swinging between every weld is optimistic by exactly the
amount a plant manager notices.

## Jig location

A jig's placement is **determined, not searched**. The locators encode a 3-2-1
scheme: locators facing the same way form one locating plane, and for each
plane in order of how many locators it has, the part is translated along that
plane's normal until its extreme surface point touches. Three planes fix all
six degrees of freedom, and the translations are independent because the
planes are mutually perpendicular in any sane jig.

Seating is then checked per locator against the part's **surface**, not its
vertices — the same class of error as measuring a clamp to the nearest mesh
vertex. A locator the part does not reach means the part is not located, which
is a **violation**, not a lower score: a jig exists to locate the part.

## Preview

`weldforge preview` writes one self-contained HTML file: the part, the cell,
the jig locators, the seams coloured by joint type with their wire direction,
the torch path split into arc-on and air moves, and the robot animated along
the cycle.

No CDN, no install, no network — a welding cell often has none of them, and
looking at the plan is how most mistakes get caught. The renderer is about two
hundred lines of plain JavaScript on a 2-D canvas with painter's-algorithm
depth sorting. Not a game engine, but enough to see whether the torch points
at the joint.

## Kinematics

Closed-form IK for the standard 6R spherical-wrist arm returns **all eight**
solutions at once, which is what makes configuration-consistent paths possible:
seed each station with the previous one's answer and the arm never reconfigures
mid-bead. Numeric damped-least-squares IK handles anything that does not fit
the structure — and the structure is **checked**, not assumed, because a
silently mismatched closed form produces joint angles that satisfy its own
equations and put the torch somewhere else.

The `±2π` aliases of each joint are tried before declaring a target
unreachable: an industrial wrist travels well past ±180°, so the principal
arctangent value is often outside the limits while an equivalent angle a full
turn away is comfortably inside.

URDF and SRDF are **generated** from the same `data/robots/*.yaml` the planner
uses, and `urdf_to_robot` reads them back so the round trip is tested — exact to
1e-6 mm. Hand-maintaining both guarantees they drift, and a drifted URDF means
MoveIt plans for a different robot.

## Collision checking

Capsules against meshes: conservative, needs no FCL build, and fast enough to
check every station.

The torch needed more care. A single capsule caps the gas cup's flat front with
a **hemisphere** of the same radius, hanging 22 mm below the contact tip — and
the TCP is at the wire tip, a stick-out further forward. Approaching a fillet
at 45° on a plate lying on the bench, that phantom hemisphere dips through the
table and every seam is declared inaccessible. So the torch is modelled as a
thin wire capsule plus a frustum of widening segments, and the **wire alone**
is allowed to touch the part — it ends on the joint by construction.

The table is a finite box, not a half-space. An infinite plane at bench height
declares everything below it solid, including the robot's column standing on
the floor.

## Safety

| Guard | Where |
|---|---|
| `arc_enabled` ships `false`; nothing can strike an arc until a human arms it | `weldforge_cell/config/welder.yaml` |
| Execution refuses a plan the planner flagged, unless asked for a dry run | `execution_node._accept_goal` |
| Arc struck stationary at the start station; motion only after current is confirmed | `execution_node._strike` |
| Arc dropout mid-bead stops motion immediately | `execution_node._follow` |
| Arc and gas driven low on shutdown, however it happens | `welder_io_node.destroy_node` |
| Modbus backend left unimplemented rather than guessed | `welder_io_node._write_to_machine` |
| Every program header demands a dry run and axis-sign verification | `post/base.py::header_lines` |
