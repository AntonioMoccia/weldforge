// WeldForge add-in for SOLIDWORKS.
//
// Registers a toolbar and a task pane, exports the active assembly or
// weldment as STEP, and drives the WeldForge API.
//
// The export is the part that needs care. A weldment's seams are found by
// contact between bodies, so the STEP must keep the bodies **separate**:
//
//   swExportSTEPAC214  (AP214) keeps an assembly's components apart, which
//                      is what we need;
//   "Save bodies as a single solid" / any flattening option destroys every
//                      contact and with it every seam between parts.
//
// So the add-in sets AP214 and clears the flattening preferences before
// exporting, and restores whatever the user had afterwards.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using SolidWorks.Interop.swpublished;

namespace WeldForge
{
    [ComVisible(true)]
    [Guid("8E1F6C4A-5B2D-4C7E-9A31-6F0D2B8E4A17")]
    [ClassInterface(ClassInterfaceType.None)]
    [ProgId("WeldForge.SwAddin")]
    public class WeldForgeAddin : ISwAddin
    {
        private const string AddinTitle = "WeldForge";
        private const string AddinDescription =
            "Recognise weld seams, generate welding procedures and export robot programs";

        private ISldWorks _app;
        private ICommandManager _commandManager;
        private int _addinCookie;
        private TaskPaneHost _taskPane;
        private ITaskpaneView _taskPaneView;

        private const int CommandGroupId = 7312;

        // -- COM registration ------------------------------------------------

        [ComRegisterFunction]
        public static void RegisterFunction(Type type)
        {
            // SOLIDWORKS discovers add-ins through these registry keys. Written
            // here rather than by an installer so the add-in can be registered
            // with regasm during development.
            string keyPath = @"SOFTWARE\SolidWorks\Addins\{" + type.GUID.ToString() + "}";
            using (var key = Registry.LocalMachine.CreateSubKey(keyPath))
            {
                key.SetValue(null, 0);                       // 0 = not loaded at startup
                key.SetValue("Description", AddinDescription);
                key.SetValue("Title", AddinTitle);
            }

            string userPath = @"Software\SolidWorks\AddInsStartup\{" + type.GUID.ToString() + "}";
            using (var key = Registry.CurrentUser.CreateSubKey(userPath))
            {
                key.SetValue(null, 1);                       // load for this user
            }
        }

        [ComUnregisterFunction]
        public static void UnregisterFunction(Type type)
        {
            Registry.LocalMachine.DeleteSubKey(
                @"SOFTWARE\SolidWorks\Addins\{" + type.GUID.ToString() + "}", false);
            Registry.CurrentUser.DeleteSubKey(
                @"Software\SolidWorks\AddInsStartup\{" + type.GUID.ToString() + "}", false);
        }

        // -- ISwAddin --------------------------------------------------------

        public bool ConnectToSW(object thisSw, int cookie)
        {
            _app = (ISldWorks)thisSw;
            _addinCookie = cookie;
            _app.SetAddinCallbackInfo2(0, this, _addinCookie);
            _commandManager = _app.GetCommandManager(_addinCookie);

            AddCommandGroup();
            AddTaskPane();
            return true;
        }

        public bool DisconnectFromSW()
        {
            _commandManager.RemoveCommandGroup2(CommandGroupId, true);
            if (_taskPaneView != null)
            {
                _taskPaneView.DeleteView();
                _taskPaneView = null;
            }
            _taskPane = null;
            _commandManager = null;
            _app = null;
            GC.Collect();
            GC.WaitForPendingFinalizers();
            return true;
        }

        // -- UI --------------------------------------------------------------

        private void AddCommandGroup()
        {
            int errors = 0;
            var group = _commandManager.CreateCommandGroup2(
                CommandGroupId, AddinTitle, AddinDescription, "", -1, true, ref errors);

            group.AddCommandItem2("Detect seams", -1,
                "Export the model and recognise its weld seams",
                "Detect seams", 0, nameof(OnDetectSeams), nameof(OnEnableWithDocument), 0,
                (int)swCommandItemType_e.swMenuAndToolbarItem);

            group.AddCommandItem2("Plan weld job", -1,
                "Generate procedures, place the part, plan the paths and simulate",
                "Plan weld job", 1, nameof(OnPlanJob), nameof(OnEnableWithDocument), 1,
                (int)swCommandItemType_e.swMenuAndToolbarItem);

            group.AddCommandItem2("Export program", -1,
                "Write the planned job as a robot controller program",
                "Export program", 2, nameof(OnExportProgram), nameof(OnEnableWithJob), 2,
                (int)swCommandItemType_e.swMenuAndToolbarItem);

            group.AddCommandItem2("WeldForge panel", -1,
                "Show the WeldForge task pane",
                "WeldForge panel", 3, nameof(OnShowPanel), nameof(OnEnableAlways), 3,
                (int)swCommandItemType_e.swMenuAndToolbarItem);

            group.HasToolbar = true;
            group.HasMenu = true;
            group.Activate();

            // Available in part, assembly and drawing documents.
            foreach (int documentType in new[]
                     {
                         (int)swDocumentTypes_e.swDocPART,
                         (int)swDocumentTypes_e.swDocASSEMBLY
                     })
            {
                var tab = _commandManager.AddCommandTab(documentType, AddinTitle);
                var box = tab.AddCommandTabBox();
                var ids = new[] { group.CommandID[0], group.CommandID[1], group.CommandID[2] };
                var styles = new[]
                {
                    (int)swCommandTabButtonTextDisplay_e.swCommandTabButton_TextBelow,
                    (int)swCommandTabButtonTextDisplay_e.swCommandTabButton_TextBelow,
                    (int)swCommandTabButtonTextDisplay_e.swCommandTabButton_TextBelow
                };
                box.AddCommands(ids, styles);
            }
        }

        private void AddTaskPane()
        {
            _taskPaneView = _app.CreateTaskpaneView2("", AddinTitle);
            _taskPane = (TaskPaneHost)_taskPaneView.AddControl(
                "WeldForge.TaskPaneHost", "");
            if (_taskPane != null)
            {
                _taskPane.Initialise(this);
            }
        }

        // -- enablement ------------------------------------------------------

        public int OnEnableAlways() => 1;

        public int OnEnableWithDocument()
        {
            return _app.ActiveDoc != null ? 1 : 0;
        }

        public int OnEnableWithJob()
        {
            return _taskPane != null && _taskPane.HasJob ? 1 : 0;
        }

        // -- commands --------------------------------------------------------

        public void OnShowPanel()
        {
            _taskPaneView?.ShowView();
        }

        public void OnDetectSeams()
        {
            OnShowPanel();
            _taskPane?.DetectSeams();
        }

        public void OnPlanJob()
        {
            OnShowPanel();
            _taskPane?.PlanJob();
        }

        public void OnExportProgram()
        {
            OnShowPanel();
            _taskPane?.ExportProgram();
        }

        // -- export ----------------------------------------------------------

        /// <summary>
        /// Save the active document as a STEP AP214 assembly, keeping the
        /// bodies separate, and return the path.
        /// </summary>
        public string ExportStep()
        {
            var document = (IModelDoc2)_app.ActiveDoc;
            if (document == null)
            {
                throw new WeldForgeException("no document is open in SOLIDWORKS");
            }

            string name = Path.GetFileNameWithoutExtension(document.GetTitle());
            if (string.IsNullOrWhiteSpace(name))
            {
                name = "weldment";
            }
            string path = Path.Combine(Path.GetTempPath(),
                name.Replace(" ", "_") + "_weldforge.step");

            // Remember the user's preferences and restore them afterwards.
            int previousFormat = _app.GetUserPreferenceIntegerValue(
                (int)swUserPreferenceIntegerValue_e.swStepAP);
            bool previousFlatten = _app.GetUserPreferenceToggle(
                (int)swUserPreferenceToggle_e.swStepExportPartAsSingleBody);

            try
            {
                // AP214 keeps the assembly structure; AP203 can flatten it.
                _app.SetUserPreferenceIntegerValue(
                    (int)swUserPreferenceIntegerValue_e.swStepAP,
                    (int)swStepAPValue_e.swSTEPAP214);
                // Never fuse the bodies: contact between them IS the weld.
                _app.SetUserPreferenceToggle(
                    (int)swUserPreferenceToggle_e.swStepExportPartAsSingleBody, false);

                int errors = 0;
                int warnings = 0;
                bool ok = document.Extension.SaveAs3(
                    path,
                    (int)swSaveAsVersion_e.swSaveAsCurrentVersion,
                    (int)swSaveAsOptions_e.swSaveAsOptions_Silent,
                    null, null, ref errors, ref warnings);

                if (!ok || !File.Exists(path))
                {
                    throw new WeldForgeException(
                        $"STEP export failed (errors {errors}, warnings {warnings})");
                }
                return path;
            }
            finally
            {
                _app.SetUserPreferenceIntegerValue(
                    (int)swUserPreferenceIntegerValue_e.swStepAP, previousFormat);
                _app.SetUserPreferenceToggle(
                    (int)swUserPreferenceToggle_e.swStepExportPartAsSingleBody, previousFlatten);
            }
        }

        public string ActiveDocumentName()
        {
            var document = (IModelDoc2)_app?.ActiveDoc;
            return document == null
                ? string.Empty
                : Path.GetFileNameWithoutExtension(document.GetTitle());
        }

        public void StatusMessage(string message)
        {
            _app?.SendMsgToUser2(
                message,
                (int)swMessageBoxIcon_e.swMbInformation,
                (int)swMessageBoxBtn_e.swMbOk);
        }
    }
}
