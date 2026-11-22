# Productising this

Notes for the conversation with your company about turning this into
machines you sell with software maintenance and licences. Written as an
engineering assessment, not a sales case: what is ready, what is not, and
what the gaps cost.

---

## What is ready today

| | State |
|---|---|
| Seam recognition on real assemblies | Works. 193 seams / 4.8 m found on a 58-body node in 8 s, with physically sensible member thicknesses. |
| Welding procedure generation | Works, and internally consistent. Covers GMAW on carbon steel and austenitic stainless, 0.8–1.6 mm wire, all seven EN ISO 6947 positions. |
| Part placement with constraints | Works. Chooses orientation and position, reports why it scored what it did. |
| Kinematics | Closed-form IK exact to machine precision on four arm families, with configuration-consistent paths. |
| Robot program output | Five targets, syntactically correct, with provenance headers. |
| Cell integration | ROS 2 Jazzy workspace with MoveIt, Gazebo and an execution action server that owns the arc. |
| CAD integration | REST API with a FreeCAD workbench and a SOLIDWORKS add-in. |
| Test coverage | ~640 tests; each significant one pins a mistake that was actually made. |

## What is not ready, and what each gap costs

Ordered by what would stop a sale.

### 1. Robot calibration is nominal (blocking)

The shipped geometry is data-sheet geometry. Link lengths are right; **per-axis
sign and zero offset are not verified** against any real controller, and they
differ by vendor and sometimes by controller software version.

A sign error puts the torch somewhere else with no warning. Right now the
mitigation is a header telling the operator to dry-run and check — adequate for
your own cell, not for a product.

*To close:* a calibration procedure — jog to known postures, read back joint
values, solve for the offsets — plus a per-cell calibration file. A week or two
of work, then a day per robot model on real hardware.

### 2. No procedure qualification path (blocking for pressure/structural work)

The welding parameters are published consumable working ranges. That is a sound
starting point for a WPS and **not** a qualified procedure. Anything to EN 1090
(structural steel) or EN 13445 (pressure) needs qualification to EN ISO 15614-1,
which means test coupons, destructive testing and a notified body.

*To close:* nothing software can do alone. What software *can* do is manage it:
store qualified ranges per customer, refuse to generate a procedure outside a
qualified envelope, and print a WPQR reference on every sheet. That turns a
liability into a feature — and it is the kind of feature that justifies a
subscription, because a qualified-range library is worth more every year.

### 3. Touch sensing and seam tracking (blocking for real production)

Everything here is **offline**: it assumes the part is where the fixture says.
Real parts have fit-up variation of a millimetre or more, and a 6 mm fillet
offset by 1.5 mm is a reject.

Production cells solve this with touch sensing (wire or nozzle contact to find
the joint before welding) and sometimes through-arc or laser seam tracking.
The architecture has the hooks — the plan knows exactly where each seam *should*
be, which is what a search needs as a starting point — but none of it is
implemented.

*To close:* touch sensing is a few weeks (the robot and power source already
support it; it is a search routine plus a frame correction applied to the
planned path). Through-arc tracking is more, and laser tracking means buying a
sensor.

### 4. Multi-pass thermal management (important)

Preheat and interpass limits are computed and reported, but nothing **measures**
or **waits**. A real multi-pass procedure on thick section needs interpass
temperature measured and the robot to pause.

*To close:* a pyrometer, an I/O read, and a wait state in the execution node.
Small, but it must exist before selling thick-section capability.

### 5. Curved-to-curved contact (coverage gap)

Contact detection handles planar faces and falls back to manual selection for
curved-to-curved contact — a tube saddled onto another tube, for instance. That
is a real and common joint in some sectors and is simply not covered.

*To close:* a surface-surface intersection via OCCT. Doable; a couple of weeks
with careful testing.

### 6. No operator UI of its own

The REST API, the CLI and the two CAD plugins are the interfaces. There is no
shop-floor screen: no touch-friendly seam list, no jog panel, no "run this
part" button for someone who does not use CAD.

*To close:* the API already exposes everything such a UI needs. It is front-end
work, not engineering work, but it is the difference between a tool your
engineers use and a machine you sell.

---

## Licensing: what actually fits

The software is Python plus a ROS 2 workspace. Three honest options:

### Node-locked licence file (recommended to start)

A signed file naming the cell, the robot and an expiry; checked at planner
start-up. Cheap to build (a day), familiar to industrial customers, works with
no network — which matters because a lot of welding cells are not on a network
at all.

Weak against a determined customer. That is usually acceptable: the realistic
threat is not cracking, it is a customer running a second cell on one licence,
and a node lock plus an audit clause handles that commercially.

### Floating licence server

Worth it only once a customer has several cells and wants to move entitlements
between them. Adds a network dependency to a machine that may not have one, so
it needs a grace period for offline operation.

### Subscription, with the value in the data

The most defensible, and it matches where the value really is: the **procedure
library**. A customer's qualified parameters, their own cells, their machine
profiles, their consumables. That grows every year and is what they would lose
by stopping paying — far stickier than code they could in principle replace.

This is the model to aim for, but it needs gap 2 (qualified-range management)
built first, because that is what makes the library worth paying for.

### On obfuscation

Python can be compiled to bytecode or packed with Nuitka. It raises the bar a
little and costs real debugging convenience on a shop floor, where "run this
one line to see what the planner thinks" is how a support call gets closed in
ten minutes instead of a site visit. Not worth it early.

---

## Maintenance: what you are committing to

Being specific about this matters, because a maintenance contract priced
without it loses money.

| Area | Cadence | Why |
|---|---|---|
| ROS 2 distribution | every 2 years | Jazzy's support ends May 2029; the next LTS will need a port |
| OpenCASCADE / OCP | yearly | a major OCCT release changes API spellings; the backend adapter localises this, which is why it exists |
| CAD format support | per CAD release | SOLIDWORKS and FreeCAD change export behaviour; the plugins need checking each year |
| Robot controller firmware | per customer upgrade | an axis-sign or zero change in a controller release invalidates calibration |
| Procedure data | per consumable change | new wire, new gas, new grade |
| Machine profiles | per power source | a new brand in the cell is a new YAML file, which is the easy case; a new *interface* is not |

Realistic steady-state effort for a handful of installed cells: **one engineer
at 20–30%**, rising sharply if customers run different robot brands, because
each brand's postprocessor and calibration is separately maintained.

The thing to resist is per-customer forks. The architecture is built against
this — cells, robots, machines, materials and procedures are all **data**, not
code — and that discipline is what keeps the maintenance number from
multiplying by the number of customers.

---

## What I would do next, in order

1. **Calibrate against your own robot.** Until the axis signs are verified on
   real hardware, nothing downstream can be trusted, and this is the one gap
   that nothing else can be built on top of.
2. **Weld a test coupon from a generated program.** Dry run first, then a
   simple T-joint in 6 mm. The parameters will need adjusting; the point is to
   find out *how much*, because that number tells you whether the procedure
   engine is a tool or a starting point.
3. **Implement touch sensing.** Without it the system cannot cope with real
   fit-up, and no production customer will accept that.
4. **Then** build the operator UI and the licence file. Both are
   straightforward once there is something proven to wrap.

Steps 1 and 2 are a week and will tell you more about whether this is a
product than any amount of further development.

---

## One thing to be honest about internally

This replaces an operator's judgement about *where* and *how* to weld with a
program's. On the cases it covers it does so defensibly, and it shows its
reasoning — every seam carries a confidence, every procedure its warnings,
every placement its violations. But a welding engineer still has to review the
output, and the system is built on that assumption rather than around it.

Selling it as "automatic programming" invites the first bad weld to become a
warranty problem. Selling it as "the programming done in minutes instead of
days, for an engineer to check" is both true and, in my view, the more valuable
claim.
