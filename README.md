# WeldForge

From a 3D model to a robotic weld.

WeldForge takes a weldment as CAD, works out **where the welds are**, **how to
weld them**, **where to put the part**, and **how the robot moves** — then
writes the program for the controller in your cell.

```
CAD file ─▶ seams ─▶ procedures ─▶ placement ─▶ torch paths ─▶ report ─▶ robot program
```

It runs four ways: a **desktop application** for the person setting the job
up, a **CLI**, a **REST API** with plugins for FreeCAD and SOLIDWORKS, and a
**ROS 2 Jazzy** workspace for the cell itself.

---

## Why this exists, and what is actually hard

Most of the chain is bookkeeping. Three parts are not, and they are where the
engineering went.

### 1. A weldment exported from CAD has no seams in it

Open a real assembly — the one this was built against has 58 bodies and 885
faces — and **every edge belongs to exactly one body**. The parts touch, but
they share no topology. A detector that looks for shared edges with a 90°
dihedral angle, which is the obvious approach and what the previous attempts at
this did, finds **zero** of the welds that have to be made.

So WeldForge detects seams two ways:

* **shared topology** — for a single bent or machined body, or a weldment that
  was booleaned before export;
* **contact** — for the assembly case, which is the normal one. It finds where
  two bodies touch, walks the boundary of the contact footprint, and forms a
  seam between the receiving face and the flank rising from each boundary edge.

### 2. Measuring the joint correctly is all sign work

Three joints that all measure 90° on the weld side are welded differently:

| | Looks like | Actually is |
|---|---|---|
| Inside corner of an L-bracket | 90° | a fillet |
| Outside edge of a cube | 90° | nothing — the outside of the part |
| Plate lying on a plate | 90° | a lap |

Only the **concavity sign** separates the first two, and only the **plate axis**
separates the first from the third. Measure the angle alone and the program
fills with welds that do not exist. The derivation is in
[`seams/dihedral.py`](src/weldforge/seams/dihedral.py) and is tested against
every wedge angle from 5° to 355°.

### 3. Presenting the seam to the torch is worth more than any clever path

A two-axis positioner turns each seam, in turn, into the best position it can
be. The same fillet goes from

* overhead at 4 mm/s in short arc, weaving on every fill pass, seven passes, to
* horizontal-vertical at 7 mm/s in spray, no weaving, four passes.

And a seam facing the floor is not slow, it is **unreachable**. So the planner
solves an **indexed** positioner setting per seam -- and it moves the
positioner only when that buys something, because indexing costs real seconds
which are charged to the cycle time. Fully coordinated nine-axis motion is
deliberately **not** generated: seams that would need it are reported instead
of approximated, because an approximation of coordinated motion drags the
torch off the joint.

A **jig** is handled differently again: its placement is *solved*, not
searched. The locators encode a 3-2-1 scheme, and the part goes where they put
it -- which is the whole point of a jig, and what makes an offline-planned
program repeatable. A part that does not reach a locator is reported as not
located rather than quietly welded in the wrong place.

### 4. The numbers have to agree with each other

A welding procedure that lists travel speed and wire feed from separate tables
lists two numbers that contradict each other. WeldForge derives them from one
equation:

```
wire cross-section × wire feed × efficiency  =  deposit area × travel speed
```

Pick the travel speed the position allows, and the wire feed follows; the wire
feed fixes the current through the synergic line, and the current fixes the
voltage. Nothing is chosen twice. The test suite checks the identity holds on
every pass of every procedure it generates.

**New here? Read [QUICKSTART.md](QUICKSTART.md)** — three ways to start it,
with the commands.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[cad,api,dev]"
weldforge info
```

For the desktop application, add `studio`:

```bash
pip install -e ".[cad,api,studio,dev]"
weldforge-studio
```

`cad` pulls in `cadquery-ocp`, the OpenCASCADE wheels — needed to read STEP and
IGES. Without it WeldForge still works on meshes (STL/OBJ/PLY) with lower
confidence, because a tessellation cannot tell 90.0° from 89.3°.

Inside the ROS container, `pythonocc-core` from conda-forge works equally well:
the [backend adapter](src/weldforge/cad/backend.py) normalises the differences
between the two bindings.

---

## Use it

### The desktop application

```bash
weldforge-studio            # or: weldforge-studio part.step
```

This is the one an operator or a welding engineer uses. One window, left to
right along the job:

| | |
|---|---|
| **Import CAD** | STEP, IGES, BREP or a mesh. The part appears on the cell's bench. |
| **Place the part** | Drop it on the table, snap it into the jig's 3-2-1 datums, nudge it by hand, or let the placement search find the pose that presents the most seams downhand. Lock it when you have measured it on the real bench. |
| **Detect the seams** | Contact detection and topological detection together. Each seam shows its joint type, length, groove angle, welding position, plate gauges and the detector's confidence. Switch off the ones you do not want. |
| **Trace a weld line** | Click along the model and the seam is yours. Two ways: along edges, where the groove comes from the edge's own two faces; or an edge plus the face it beds onto, which is the only information an assembly gives you, since the two bodies share no topology. Hand-traced seams survive re-detection — detection never deletes your work. |
| **Set the procedure** | Current, voltage, wire feed, travel speed, gas, pass sequence, preheat — generated from the joint and the material, then editable. A procedure you edit is marked, and a bulk regenerate leaves it alone. |
| **Simulate** | Plan the job and scrub through it frame by frame: torch orientation, external axis values, arc on and arc off, with the report naming any seam the arm cannot reach and which axis ran out. |
| **Send it** | Write the program to a folder, push it over FTP to a controller, or POST it to the on-machine PC's REST service. Every dispatch is recorded in the project: what went where, when. |

Everything is one file. **File ▸ Save** writes a `.weldforge` project holding
the part, the cell, the placement, every seam including the hand-traced ones,
the procedures, the positioner values, the generated programs and the dispatch
history. The CAD is **embedded, not referenced**: a project copied to the
machine PC opens with its geometry, because an engineer's folder tree does not
travel with the file, and a project that opens with no geometry looks like it
worked.

Keyboard: `1` select, `2` trace a weld line, `3` place the part; `F5` detect,
`F6` procedures, `F7` place automatically, `F8` plan; `Enter` accepts the thing
you are tracing, `Backspace` steps back, `Esc` cancels, `Space` switches a seam
on or off.

Planning runs on a background thread, so a 90-second plan on a 200-seam
weldment leaves the window alive and the progress visible.

### Command line

```bash
# What can this installation do?
weldforge info

# Where are the welds?
weldforge seams part.step

# How do I weld them?
weldforge wps part.step --material S355

# Plan the whole job and write the programs.
weldforge plan part.step --cell bench_cell -t rapid -t ros2 -o out/

# Look at it: one self-contained HTML file, no ROS and no install.
weldforge preview part.step --cell positioner_cell -o preview.html

# Serve the API for the CAD plugins.
weldforge serve

# Regenerate the URDF and SRDF from the robot and cell data.
weldforge urdf --cell positioner_cell -o ros2_ws/src/weldforge_description/urdf
```

`weldforge plan` exits **0** if the job is safe to run and **2** if it planned
but the report found problems — so a shop script can gate on it.

Real output on a 58-body structural node:

```
  #   joint   len mm  groove  pos  thk mm     gap  conf
  1   fillet  100.0   90      PB   75.0/18.0  0.0  0.63
  5   lap      92.3   90      PB   88.0/6.0   0.0  0.83  curved
  9   fillet   80.4   90      PD   6.0/6.0    0.0  0.63
 14   tee      76.6   90      PF   6.0/6.0    0.0  0.83
 ...
  193 seams, 4787 mm of weld in 7.82s
```

And the procedure it generates for one of them — note the flat seam takes 4
passes and the overhead one takes 7, uses no spray transfer, and weaves on the
fills but never on the root:

```
01 - fillet 100 mm (PB), 18.0 mm S355, leg 12.0 mm, 4 passes, preheat 175 C
  pass    leg  A    V     m/min  mm/s  mode       work/trav  weave  kJ/mm
  root    4.7  171  20.9  4.86   7.00  short_arc  45/-10     -      0.41
  fill 1  6.4  252  27.8  9.17   7.00  spray      33/-10     -      0.80
  fill 2  6.4  252  27.8  9.17   7.00  spray      57/-10     -      0.80
  cap     6.4  252  27.8  9.17   7.00  spray      45/-10     -      0.80
```

### FreeCAD and SOLIDWORKS

```bash
weldforge serve
```

* **FreeCAD** — symlink [`plugins/freecad/WeldForge`](plugins/freecad) into
  FreeCAD's `Mod` folder. The workbench draws the detected seams on the model,
  coloured by joint type, with the **wire direction** at each station — which is
  how a torch pointing the wrong way becomes obvious. Needs no pip install:
  the client is pure standard library.
* **SOLIDWORKS** — a C# add-in with a task pane, or a
  [Python COM script](plugins/solidworks/python) for when deploying a
  COM-registered add-in is not an option.

Both set **AP214** and turn off body flattening on export, because a flattened
STEP has no contacts and therefore no seams.

### Look at the plan without ROS

```bash
weldforge preview part.step -o preview.html && open preview.html
```

One HTML file: the part, the cell, the jig locators, every seam coloured by
joint type with its **wire direction** drawn at each station, the torch path
split into arc-on and air moves, and the robot animated along the whole cycle.
No CDN, no install, no network -- a welding cell often has none of them.

The wire-direction display is the one that earns its keep: a torch pointing
the wrong way is obvious there and invisible in a table of numbers.

### ROS 2

```bash
./docker/build.sh                                      # build and verify
ros2 launch weldforge_bringup weldforge.launch.py      # simulated, arc inhibited
```

Six packages: [`weldforge_msgs`](ros2_ws/src/weldforge_msgs) (interfaces),
[`weldforge_cell`](ros2_ws/src/weldforge_cell) (planner services, welder I/O,
weld-job execution), [`weldforge_description`](ros2_ws/src/weldforge_description)
(URDF, **generated** from the robot data), `weldforge_moveit_config`,
`weldforge_sim` and `weldforge_bringup`.

```bash
ros2 service call /weldforge_planner/load_part weldforge_msgs/srv/LoadPart "{path: /work/part.step}"
ros2 service call /weldforge_planner/plan_job  weldforge_msgs/srv/PlanJob  "{part_id: ''}"
ros2 action send_goal /weldforge_execution/execute weldforge_msgs/action/ExecuteWeldJob "{dry_run: true}"
```

**The arc interlock ships off.** `arc_enabled` defaults to `false` and no
amount of planning can strike an arc until a human sets it true. The execution
action also refuses any plan the planner itself flagged — unless you ask for a
dry run, which is how you investigate one.

---

## What comes out

| Target | Format | Units | Arc control |
|---|---|---|---|
| `rapid` | ABB `.mod` | mm, quaternion | `ArcLStart`/`ArcL`/`ArcLEnd` with `welddata` |
| `krl` | KUKA `.src` | mm, A-B-C Euler | digital outputs + arc-on feedback |
| `urscript` | UR `.script` | **m**, rotation vector | digital + analog I/O |
| `fanuc` | TP `.ls` | mm, W-P-R | `Arc Start`/`Arc End` schedules |
| `ros2` | `JointTrajectory` + weld plan YAML | **m**, rad | per-point `arc_on` flags |

Positioner axes travel with the program in each target's own convention:
RAPID's `extjoint`, KUKA's `E1..E6`, a second FANUC motion group, and extra
joints appended to the ROS trajectory. A program with them missing welds the
part in whatever orientation the previous seam left it in.

Plus machine setpoints for the power source itself — synergic job numbers and
scaled fieldbus words for Fronius, Lincoln or a generic analog source. Adding a
brand is [a YAML file](data/machines), not a code change.

---

## How it is put together

```
src/weldforge/
  units.py      the ONLY place lengths cross a unit boundary
  model.py      the domain model every layer speaks
  geom/         SE(3) transforms, polyline maths
  cad/          OCCT backend adapter, topology, loaders
  seams/        dihedral geometry, two detectors, chaining, classification
  wps/          procedure tables, pass planning, machine profiles
  fixture/      cell model, placement search, positioner, jig location
  kin/          robot model, FK, analytic + numeric IK, URDF export
  path/          torch poses, weaving, seam sequencing, path assembly
  collide/      capsule-against-mesh checking
  sim/          reachability, collisions, cycle time, consumables
  viz/          self-contained HTML 3D preview
  post/         RAPID, KRL, URScript, Fanuc, ROS 2
  plan/         the pipeline
  api/          REST service
  app/          the desktop application
```

The desktop application is layered so that nearly all of it runs without a
screen, and is therefore tested:

```
app/
  project.py      the document and the .weldforge file
  scene3d.py      the VTK scene and picking      -- runs offscreen
  tools.py        the interactive tools, as state machines, no Qt at all
  panels.py       the Qt dock panels             -- run headless
  send.py         getting the program to the machine
  main_window.py  the shell, the only part that needs a screen
```

That split is not tidiness. `QVTKRenderWindowInteractor` — the widget that puts
a VTK render window inside a Qt one — segfaults under an offscreen platform
plug-in, so if the seam-tracing rules or the project format lived in the window
they could not be tested at all. Instead the window is a router: it owns the
project, connects panels to tools, and contains no welding logic.

Every stage is usable and testable on its own. [`plan/pipeline.py`](src/weldforge/plan/pipeline.py)
only wires them together.

Units are **millimetres, radians, seconds** throughout. ROS and URScript want
metres; those conversions happen at exactly two boundaries and nowhere else.
The previous generations of this software mixed metres and millimetres across
module boundaries, which is why [`units.py`](src/weldforge/units.py) exists and
why it is the only sanctioned converter.

See [docs/architecture.md](docs/architecture.md) for the detail, and
[docs/commercial.md](docs/commercial.md) for the productisation notes.

---

## Tests

```bash
pytest -q                      # the whole suite
pytest -m "not slow" -q        # skip the real-weldment run
./docker/build.sh              # plus colcon build and the ROS node imports
```

The tests that matter are not coverage padding — each one pins a mistake that
was actually made and found while building this:

| Test | What it caught |
|---|---|
| `test_cad.py::test_face_normals_point_out_of_the_solid` | 362 of 885 faces are `REVERSED`; uncorrected, their normals point into the metal |
| `test_dihedral.py::test_the_two_90_degree_cases_are_distinguished_only_by_concavity` | an L-bracket's inside corner and a cube's outside edge both measure 90° |
| `test_seams.py::test_flush_ends_are_not_reported_as_seams` | phantom seams where two members finish level and there is no groove |
| `test_seams.py::test_curved_seam_arc_length_is_exact` | chord length under-quotes every curved weld by 10% |
| `test_wps.py::test_wire_feed_and_travel_speed_conserve_volume` | the identity the procedure engine is built on |
| `test_wps.py::test_no_pass_exceeds_the_bead_its_position_can_hold` | pass count must divide **area**, not leg — area grows as the square |
| `test_kinematics.py::test_analytic_ik_round_trip` | two IK bugs that were self-consistent and still put the torch elsewhere |
| `test_kinematics.py::test_solve_keeps_the_configuration_across_a_path` | an arm reconfiguring mid-bead |
| `test_pipeline.py::test_the_torch_is_allowed_to_touch_the_part_it_welds` | the torch "colliding" with the weld at every station |
| `test_pipeline.py::test_the_table_does_not_swallow_the_robot` | a half-space table containing the robot's own column |
| `test_polyline.py::test_distance_to_a_straight_seam_is_measured_to_the_line` | a clamp planted mid-weld reported as 100 mm clear |
| `test_ros_conversions.py` | the mm↔m boundary, verified without ROS installed |
| `test_positioner_jig_viz.py::test_the_positioner_stays_put_when_there_is_nothing_to_gain` | indexing the positioner for no gain, and charging the time anyway |
| `test_positioner_jig_viz.py::test_each_seam_is_planned_at_its_own_positioner_setting` | planning every seam at the setup pose, wrong for all but one |
| `test_positioner_jig_viz.py::test_a_part_that_misses_its_locators_is_rejected` | welding a part the jig never actually located |
| `test_positioner_jig_viz.py::test_preview_html_is_self_contained` | a viewer that needs a network, in a cell that has none |
| `test_app.py::test_tracing_an_edge_and_a_bedding_face_finds_the_same_joint` | a hand-traced seam must agree with the detector on a joint both can see |
| `test_app.py::test_a_detected_seam_cannot_be_deleted_only_switched_off` | "deleting" a detected seam, which the next detection undoes |
| `test_app.py::test_the_cad_travels_inside_the_project` | a project that opens with no geometry on the machine PC |
| `test_app.py::test_an_unknown_field_survives_a_round_trip` | opening a newer project in an older copy, silently dropping data |
| `test_app.py::test_a_job_the_planner_rejected_is_not_sent_by_accident` | sending a program the planner said was unreachable |
| `test_app.py::test_a_program_file_cannot_escape_the_chosen_folder` | a program named `../../startup.mod` writing outside the chosen folder |
| `test_app.py::test_dragging_the_part_is_a_matrix_update` | re-tessellating 900 faces on every mouse move |
| `test_app_qt.py::test_the_panel_signals_the_window_connects_all_exist` | a panel control wired to nothing, found months later at the machine |

---

## Before you weld anything

Every generated program carries this header, and it is not boilerplate:

> MACHINE GENERATED. Dry-run at reduced speed with the arc disabled before
> welding a part. Verify the tool frame, the work object and every axis sign
> against this cell.

The robot geometry shipped in [`data/robots`](data/robots) is **nominal**
data-sheet geometry. Link lengths are reliable. The zero-position mapping —
which direction each axis counts as positive and where its zero sits — differs
between vendors and between controller software versions, and **must** be
confirmed against the machine in your cell. A sign error there is only
discovered by a torch hitting a fixture.

The welding parameters in [`data/procedures`](data/procedures) are published
consumable-maker working ranges: a sound starting point for a WPS, not a
qualified procedure. Qualification to EN ISO 15614-1 is still a qualification.

## Licence

Proprietary. © Antonio Moccia.
