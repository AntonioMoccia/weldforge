// The WeldForge task pane.
//
// A WinForms control hosted in SOLIDWORKS's task pane. WinForms rather than
// WPF because the task pane hosts an ActiveX control and WinForms interops
// with it without a host wrapper -- fewer moving parts in an add-in that has
// to survive SOLIDWORKS upgrades.
//
// Every API call runs off the UI thread and marshals its result back: a
// SOLIDWORKS add-in that blocks the UI thread makes the whole application
// look hung, and a plan can take minutes.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WeldForge
{
    [ComVisible(true)]
    [Guid("2C9B4D18-7A63-4E5F-8B02-1D4A7E6C9F35")]
    [ClassInterface(ClassInterfaceType.AutoDual)]
    [ProgId("WeldForge.TaskPaneHost")]
    public class TaskPaneHost : UserControl
    {
        private WeldForgeAddin _addin;
        private readonly WeldForgeClient _client = new WeldForgeClient();

        private ComboBox _cellBox;
        private ComboBox _materialBox;
        private NumericUpDown _legRatio;
        private Button _detectButton;
        private Button _planButton;
        private Button _exportButton;
        private ComboBox _targetBox;
        private ListView _seamList;
        private TextBox _log;
        private Label _status;

        private string _partId;
        private string _jobId;

        public bool HasJob => !string.IsNullOrEmpty(_jobId);

        public TaskPaneHost()
        {
            BuildUi();
        }

        public void Initialise(WeldForgeAddin addin)
        {
            _addin = addin;
            _ = LoadReferenceDataAsync();
        }

        // -- UI --------------------------------------------------------------

        private void BuildUi()
        {
            Dock = DockStyle.Fill;
            BackColor = SystemColors.Control;
            Padding = new Padding(8);
            Font = new Font("Segoe UI", 8.5f);

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 9,
                AutoSize = true
            };
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 70));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            _status = new Label { Dock = DockStyle.Fill, Text = "connecting...", AutoSize = false, Height = 32 };
            layout.Controls.Add(_status, 0, 0);
            layout.SetColumnSpan(_status, 2);

            layout.Controls.Add(new Label { Text = "Cell", Dock = DockStyle.Fill }, 0, 1);
            _cellBox = new ComboBox { Dock = DockStyle.Fill, DropDownStyle = ComboBoxStyle.DropDownList };
            layout.Controls.Add(_cellBox, 1, 1);

            layout.Controls.Add(new Label { Text = "Material", Dock = DockStyle.Fill }, 0, 2);
            _materialBox = new ComboBox { Dock = DockStyle.Fill, DropDownStyle = ComboBoxStyle.DropDownList };
            layout.Controls.Add(_materialBox, 1, 2);

            layout.Controls.Add(new Label { Text = "Leg ratio", Dock = DockStyle.Fill }, 0, 3);
            _legRatio = new NumericUpDown
            {
                Dock = DockStyle.Fill,
                DecimalPlaces = 2,
                Increment = 0.05m,
                Minimum = 0.3m,
                Maximum = 1.5m,
                Value = 0.70m
            };
            layout.Controls.Add(_legRatio, 1, 3);

            _detectButton = new Button { Text = "Detect seams", Dock = DockStyle.Fill, Height = 28 };
            _detectButton.Click += (s, e) => DetectSeams();
            layout.Controls.Add(_detectButton, 0, 4);
            layout.SetColumnSpan(_detectButton, 2);

            _planButton = new Button { Text = "Plan weld job", Dock = DockStyle.Fill, Height = 28 };
            _planButton.Click += (s, e) => PlanJob();
            layout.Controls.Add(_planButton, 0, 5);
            layout.SetColumnSpan(_planButton, 2);

            _seamList = new ListView
            {
                Dock = DockStyle.Fill,
                View = View.Details,
                FullRowSelect = true,
                Height = 180
            };
            _seamList.Columns.Add("Joint", 70);
            _seamList.Columns.Add("mm", 50);
            _seamList.Columns.Add("Pos", 40);
            _seamList.Columns.Add("Conf", 45);
            layout.Controls.Add(_seamList, 0, 6);
            layout.SetColumnSpan(_seamList, 2);

            _targetBox = new ComboBox { Dock = DockStyle.Fill, DropDownStyle = ComboBoxStyle.DropDownList };
            layout.Controls.Add(_targetBox, 0, 7);
            _exportButton = new Button { Text = "Export program", Dock = DockStyle.Fill, Enabled = false };
            _exportButton.Click += (s, e) => ExportProgram();
            layout.Controls.Add(_exportButton, 1, 7);

            _log = new TextBox
            {
                Dock = DockStyle.Fill,
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                Height = 160,
                BackColor = Color.White
            };
            layout.Controls.Add(_log, 0, 8);
            layout.SetColumnSpan(_log, 2);

            Controls.Add(layout);
        }

        private void Log(string message)
        {
            if (InvokeRequired)
            {
                BeginInvoke(new Action<string>(Log), message);
                return;
            }
            _log.AppendText(DateTime.Now.ToString("HH:mm:ss") + "  " + message + Environment.NewLine);
        }

        private void SetStatus(string message)
        {
            if (InvokeRequired)
            {
                BeginInvoke(new Action<string>(SetStatus), message);
                return;
            }
            _status.Text = message;
        }

        private void SetBusy(bool busy)
        {
            if (InvokeRequired)
            {
                BeginInvoke(new Action<bool>(SetBusy), busy);
                return;
            }
            _detectButton.Enabled = !busy;
            _planButton.Enabled = !busy;
            _exportButton.Enabled = !busy && HasJob;
            Cursor = busy ? Cursors.WaitCursor : Cursors.Default;
        }

        // -- reference data --------------------------------------------------

        private async Task LoadReferenceDataAsync()
        {
            try
            {
                string health = await _client.HealthSummaryAsync();
                var cells = await _client.CellsAsync();
                var materials = await _client.MaterialsAsync();
                var targets = await _client.TargetsAsync();

                BeginInvoke(new Action(() =>
                {
                    _cellBox.Items.AddRange(cells.Cast<object>().ToArray());
                    if (_cellBox.Items.Count > 0) _cellBox.SelectedIndex = 0;
                    _materialBox.Items.AddRange(materials.Cast<object>().ToArray());
                    int s355 = _materialBox.Items.IndexOf("S355");
                    _materialBox.SelectedIndex = s355 >= 0 ? s355 : 0;
                    _targetBox.Items.AddRange(targets.Cast<object>().ToArray());
                    int rapid = _targetBox.Items.IndexOf("rapid");
                    _targetBox.SelectedIndex = rapid >= 0 ? rapid : 0;
                    _status.Text = health.Replace(Environment.NewLine, "  |  ");
                }));
                Log("connected to " + _client.BaseUrl);
            }
            catch (WeldForgeException exc)
            {
                SetStatus("no server: start it with  weldforge serve");
                Log(exc.Message);
            }
        }

        // -- actions ---------------------------------------------------------

        public void DetectSeams()
        {
            _ = DetectSeamsAsync();
        }

        private async Task DetectSeamsAsync()
        {
            SetBusy(true);
            try
            {
                Log("exporting STEP...");
                string path = await Task.Run(() => _addin.ExportStep());
                Log("uploading " + Path.GetFileName(path));

                var part = await _client.UploadPartAsync(path, _addin.ActiveDocumentName());
                _partId = part.Id;
                Log($"{part.Name}: {part.SolidCount} solids, {part.FaceCount} faces, " +
                    $"{part.MassKg:F1} kg");

                var summary = await _client.DetectSeamsAsync(_partId);
                BeginInvoke(new Action(() =>
                {
                    _seamList.Items.Clear();
                    foreach (var seam in summary.Seams)
                    {
                        var item = new ListViewItem(seam.JointType);
                        item.SubItems.Add(seam.LengthMm.ToString("F0"));
                        item.SubItems.Add(seam.Position);
                        item.SubItems.Add(seam.Confidence.ToString("F2"));
                        // Doubtful detections are flagged, not hidden.
                        if (seam.Confidence < 0.5) item.ForeColor = Color.DarkOrange;
                        _seamList.Items.Add(item);
                    }
                }));
                Log($"{summary.Count} seams, {summary.TotalLengthMm:F0} mm of weld");

                try { File.Delete(path); } catch (IOException) { }
            }
            catch (Exception exc)
            {
                Log("ERROR " + exc.Message);
                _addin?.StatusMessage("WeldForge: " + exc.Message);
            }
            finally
            {
                SetBusy(false);
            }
        }

        public void PlanJob()
        {
            _ = PlanJobAsync();
        }

        private async Task PlanJobAsync()
        {
            SetBusy(true);
            try
            {
                if (string.IsNullOrEmpty(_partId))
                {
                    await DetectSeamsAsync();
                    if (string.IsNullOrEmpty(_partId)) return;
                }

                string cell = _cellBox.SelectedItem?.ToString() ?? "bench_cell";
                string material = _materialBox.SelectedItem?.ToString() ?? "S355";
                double legRatio = (double)_legRatio.Value;

                Log($"planning on {cell} in {material}, leg ratio {legRatio:F2}");
                _jobId = await _client.StartPlanAsync(_partId, cell, material, legRatio);

                var progress = new Progress<string>(message => Log("  " + message));
                var result = await _client.WaitForJobAsync(_jobId, progress);

                Log(result.Ok ? "VERDICT: OK" : "VERDICT: PROBLEMS");
                Log($"  {result.SeamCount} seams, {result.PassCount} passes, " +
                    $"{result.WeldLengthMm:F0} mm");
                Log($"  cycle {result.CycleTimeSeconds:F0} s, arc {result.ArcTimeSeconds:F0} s, " +
                    $"duty {result.DutyCycle:P0}");
                Log($"  {result.WireKg * 1000:F0} g wire, {result.GasLitres:F0} l gas");
                foreach (var error in result.Errors.Take(5)) Log("  ERROR " + error);
                foreach (var warning in result.Warnings.Take(5)) Log("  warn  " + warning);

                SetStatus(result.Ok
                    ? $"plan OK: {result.CycleTimeSeconds:F0} s cycle"
                    : "plan has problems: see the log");
            }
            catch (Exception exc)
            {
                Log("ERROR " + exc.Message);
            }
            finally
            {
                SetBusy(false);
            }
        }

        public void ExportProgram()
        {
            _ = ExportProgramAsync();
        }

        private async Task ExportProgramAsync()
        {
            if (!HasJob)
            {
                Log("plan a job first");
                return;
            }
            string target = _targetBox.SelectedItem?.ToString() ?? "rapid";

            string directory;
            using (var dialog = new FolderBrowserDialog
                   { Description = "Where to write the robot program" })
            {
                if (dialog.ShowDialog() != DialogResult.OK) return;
                directory = dialog.SelectedPath;
            }

            SetBusy(true);
            try
            {
                var files = await _client.ExportAsync(_jobId, target);
                foreach (var file in files)
                {
                    string path = Path.Combine(directory, file.Key);
                    File.WriteAllText(path, file.Value);
                    Log("wrote " + path);
                }
                _addin?.StatusMessage(
                    $"WeldForge wrote {files.Count} file(s) for {target}.\n\n" +
                    "Dry-run the program at reduced speed with the arc disabled " +
                    "before welding a part.");
            }
            catch (Exception exc)
            {
                Log("ERROR " + exc.Message);
            }
            finally
            {
                SetBusy(false);
            }
        }
    }
}
