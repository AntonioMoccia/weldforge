"""WeldForge workbench for FreeCAD.

Install by copying or symlinking this directory into FreeCAD's ``Mod`` folder:

===========  =================================================================
Linux        ``~/.local/share/FreeCAD/Mod/WeldForge``
macOS        ``~/Library/Application Support/FreeCAD/Mod/WeldForge``
Windows      ``%APPDATA%\\FreeCAD\\Mod\\WeldForge``
===========  =================================================================

Then start the server (``weldforge serve``) and pick "WeldForge" from the
workbench selector. The server address comes from ``WELDFORGE_URL`` and
defaults to ``http://127.0.0.1:8000``.
"""

import FreeCAD
import FreeCADGui


class WeldForgeWorkbench(FreeCADGui.Workbench):
    """From a weldment model to a robot program, without leaving FreeCAD."""

    MenuText = "WeldForge"
    ToolTip = (
        "Recognise weld seams, generate welding procedures, plan the robot "
        "paths and export a controller program"
    )

    def __init__(self):
        import os

        icon = os.path.join(os.path.dirname(__file__), "icons", "weldforge.svg")
        if os.path.exists(icon):
            self.Icon = icon

    def Initialize(self):  # noqa: N802 - FreeCAD's API
        """Register the commands and build the toolbar and menu."""
        import os
        import sys

        # The workbench directory must be importable as a package, which it is
        # not when FreeCAD loads InitGui.py directly.
        here = os.path.dirname(__file__)
        parent = os.path.dirname(here)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        try:
            from WeldForge.commands import register_commands

            self.commands = register_commands()
        except Exception as exc:
            FreeCAD.Console.PrintError(
                f"[WeldForge] could not load the commands: {exc}\n"
            )
            self.commands = []
            return

        self.appendToolbar("WeldForge", self.commands)
        self.appendMenu("WeldForge", self.commands)
        FreeCAD.Console.PrintMessage("[WeldForge] workbench ready\n")

    def Activated(self):  # noqa: N802
        from WeldForge.commands import state

        if not state.client.reachable():
            FreeCAD.Console.PrintWarning(
                f"[WeldForge] no server at {state.client.base_url}. "
                "Start one with: weldforge serve\n"
            )

    def Deactivated(self):  # noqa: N802
        pass

    def GetClassName(self):  # noqa: N802
        return "Gui::PythonWorkbench"


FreeCADGui.addWorkbench(WeldForgeWorkbench())
