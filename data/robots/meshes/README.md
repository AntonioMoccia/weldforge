# Robot link meshes

One folder per robot, named as the robot's YAML file refers to it:

```
data/robots/meshes/<folder>/base.stl, link1.stl, ... link6.stl
```

and in `data/robots/<robot>.yaml`:

```yaml
visual:
  meshes_dir: <folder>
  links:
    - { mesh: base.stl }                             # the base casting
    - { mesh: link1.stl, xyz: [0, 0, 0], rpy_deg: [0, 0, 0] }   # moves with A1
    - { mesh: link2.stl }                            # moves with A2
    # ... one entry per joint
```

Any format the CAD importer reads will do: STEP, STL, OBJ, PLY.

**Where the meshes come from.** The vendor's STEP of the robot, or the visual
meshes in the ROS support package for that arm (`kuka_kr16_support`,
`abb_irb_support`, `fanuc_arcmate_support`, …).

**The offsets matter.** Those meshes are modelled in the vendor's own link
frames. The kinematics here are modified Denavit-Hartenberg, and the two do
not coincide, so each link normally needs `xyz` and `rpy_deg` set once by
comparing a mesh against the frame it is drawn at. Without them you get
correctly-shaped links in the wrong places. Nothing here changes reach, joint
limits or collision checking -- those come from the joint parameters.
