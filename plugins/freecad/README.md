# WeldForge workbench for FreeCAD

Recognise the weld seams in a weldment, see them drawn on the model, plan the
robot job and export a controller program — without leaving FreeCAD.

## Install

Copy or symlink the `WeldForge` directory into FreeCAD's `Mod` folder:

| | |
|---|---|
| Linux | `~/.local/share/FreeCAD/Mod/WeldForge` |
| macOS | `~/Library/Application Support/FreeCAD/Mod/WeldForge` |
| Windows | `%APPDATA%\FreeCAD\Mod\WeldForge` |

```bash
# Linux example, symlinked so edits take effect on the next FreeCAD restart
ln -s "$PWD/WeldForge" ~/.local/share/FreeCAD/Mod/WeldForge
```

Restart FreeCAD and pick **WeldForge** from the workbench selector.

No pip install is needed: the client uses only Python's standard library,
because what is available inside FreeCAD's bundled interpreter varies by
platform and by how FreeCAD was installed.

## Start the server

```bash
weldforge serve            # http://127.0.0.1:8000
```

Set `WELDFORGE_URL` to point the workbench somewhere else — a server on the
cell PC, say.

## Use

| Button | What it does | How long |
|---|---|---|
| **Check server** | confirms the server is up and whether it can read STEP | instant |
| **Detect seams** | exports the selection, recognises the seams, draws them | seconds |
| **Plan weld job** | procedures, part placement, torch paths, simulation | seconds to minutes |
| **Export program** | writes a program for RAPID, KRL, URScript, Fanuc or ROS 2 | instant |
| **Clear overlay** | removes the drawn seams | instant |

Select the bodies you want, or select nothing and the whole document is used.

### What the overlay shows

* **Seam colour** is the joint type — blue fillet/tee, yellow lap, green butt,
  magenta corner.
* **Dashed** means confidence below 0.50. Look at those: they are where the
  geometry is ambiguous, usually a near-tangent face pair or a very short seam.
* **Yellow spokes** are the wire direction at each station. This is the display
  worth checking, because a torch pointing the wrong way is obvious here and
  invisible in a table of numbers.

Everything is drawn into a `WeldForge seams` group so it can be removed in one
action, and it is never saved into your document.

## Exporting correctly

The workbench uses `Import.export`, which writes a STEP **assembly** with each
body separate. That matters: WeldForge finds seams by detecting **contact**
between bodies, so a flattened or fused export has no contacts and yields no
seams between parts.

If you export STEP by hand instead, keep the bodies separate.

## Settings

Edit them in the task pane, or set the defaults in `commands.py`:

| Setting | Default | Notes |
|---|---|---|
| Cell | `bench_cell` | from `GET /cells` |
| Material | `S355` | from `GET /materials` |
| Fillet leg ratio | `0.7` | fraction of the thinner member; `1.0` for a full-strength fillet |

## Before you weld anything

Every generated program says so in its header:

> Dry-run at reduced speed with the arc disabled before welding a part.
> Verify the tool frame, the work object and every axis sign against this cell.

The robot geometry WeldForge ships is **nominal** data-sheet geometry. Link
lengths are reliable; the per-axis sign and zero offset differ between vendors
and must be confirmed against the machine in your cell.

## Troubleshooting

| Symptom | Cause |
|---|---|
| "cannot reach the WeldForge server" | the server is not running, or `WELDFORGE_URL` points elsewhere |
| "this server has no OpenCASCADE backend" | the server cannot read STEP; `pip install cadquery-ocp` where it runs |
| 0 seams found on an assembly | the STEP was flattened, or the parts do not actually touch in the model |
| Seams found but all dashed | the geometry is ambiguous — usually near-tangent faces; check the model |
| No seams between parts, only within one | the bodies were fused on export |
