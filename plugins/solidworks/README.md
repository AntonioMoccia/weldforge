# WeldForge for SOLIDWORKS

Two ways in, because deploying a compiled COM add-in is not always possible.

| | C# add-in | Python script |
|---|---|---|
| Experience | toolbar + task pane inside SOLIDWORKS | command prompt |
| Needs | compiling, admin rights to COM-register, matching interop | `pip install pywin32` |
| Use it when | you can get IT to deploy it | you cannot, or you want to batch many parts |

Both do the same work and talk to the same REST API, so the planning results
are identical.

---

## The thing that matters in both: how the STEP is exported

A weldment's seams are the places two parts **touch**. WeldForge finds them by
contact detection, which needs the bodies kept separate in the exported file.

So both paths set, and restore afterwards:

* **AP214** rather than AP203, which preserves the assembly structure;
* **"Save bodies as a single solid" OFF** — flattening the assembly fuses the
  bodies and destroys every contact, and the detector then finds no joints
  between parts at all.

If you export STEP by hand instead, set those two options, or expect zero
seams.

---

## C# add-in

### Build

```powershell
cd plugins\solidworks\WeldForgeAddin
# Adjust the interop HintPaths in WeldForgeAddin.csproj for your SOLIDWORKS year.
dotnet build -c Release
```

### Register (administrator PowerShell)

```powershell
cd bin\Release\net48
# Use the 64-bit regasm; SOLIDWORKS is 64-bit.
& "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\regasm.exe" /codebase WeldForge.SwAddin.dll
```

Then restart SOLIDWORKS. If the add-in does not appear, enable it under
**Tools → Add-Ins**.

### Unregister

```powershell
& "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\regasm.exe" /unregister WeldForge.SwAddin.dll
```

### Use

1. Start the server: `weldforge serve`
2. Open a weldment assembly in SOLIDWORKS.
3. **WeldForge → Detect seams.** The task pane lists every seam with its joint
   type, length, welding position and confidence. Anything below 0.50 is shown
   in orange: check it before planning.
4. Pick the cell, material and fillet leg ratio, then **Plan weld job**.
5. **Export program** for your controller.

---

## Python script

```powershell
pip install pywin32
weldforge serve                                   # in another window

cd plugins\solidworks\python
python weldforge_solidworks.py                    # the open document
python weldforge_solidworks.py --seams-only       # just recognise, do not plan
python weldforge_solidworks.py --file C:\parts\frame.SLDASM `
    --cell bench_cell --material S355 `
    --target rapid --target ros2 --out C:\programs
```

Exit codes: `0` planned and safe to run, `2` planned but the report found
problems, `1` something failed. Usable in a batch script.

---

## Settings

| Setting | Where | Default |
|---|---|---|
| Server URL | `WELDFORGE_URL` environment variable, or `--url` | `http://127.0.0.1:8000` |
| Cell | task pane / `--cell` | `bench_cell` |
| Material | task pane / `--material` | `S355` |
| Fillet leg ratio | task pane / `--leg-ratio` | `0.7` of the thinner member |

The cell, material and controller lists are read from the server, so they stay
in step with whatever the server actually supports rather than being
hard-coded here.

---

## Before you weld anything

Every generated program carries this in its header, and it is not boilerplate:

> Dry-run at reduced speed with the arc disabled before welding a part.
> Verify the tool frame, the work object and every axis sign against this cell.

The robot geometry shipped with WeldForge is **nominal** data-sheet geometry.
The zero-position mapping — which direction each axis counts as positive, and
where its zero is — differs between vendors and controller versions, and must
be confirmed against the machine in your cell.
