using System;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace missing_data
{
    public partial class ClusterAnalysisVisualizationForm : Form
    {
        private ClusterAnalysisOutput analysisOutput;
        private string outputPath;
        private string inputPath;
        private string pythonExe;
        private string runnerPath;
        private TextBox clusterBox;

        // herdando configuração da outra pasta

        public ClusterAnalysisVisualizationForm(ClusterAnalysisOutput analysisOutput, string inputPath, string outputPath, string pythonExe, string runnerPath)
        {
            InitializeComponent();

            this.analysisOutput = analysisOutput;
            this.outputPath = Path.Combine(outputPath, "results");
            this.inputPath = inputPath;
            this.pythonExe = pythonExe;
            this.runnerPath = runnerPath;

            BuildLayout();
        }

        private void BuildLayout()
        {
            Text = "Cluster analysis results";
            Size = new Size(1200, 800);
            MinimumSize = new Size(900, 600);

            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 3,
                Padding = new Padding(10)
            };

            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));

            var summary = BuildClusterSummaryPanel();
            var tabs = BuildVisualizationTabs();
            var buttons = BuildButtonPanel();

            root.Controls.Add(summary, 0, 0);
            root.Controls.Add(tabs, 0, 1);
            root.Controls.Add(buttons, 0, 2);

            Controls.Add(root);
        }

        private Control BuildClusterSummaryPanel()
        {
            var group = new GroupBox
            {
                Text = "Recommended clusters",
                Dock = DockStyle.Top,
                AutoSize = true,
                Padding = new Padding(10)
            };

            clusterBox = new TextBox
            {
                Dock = DockStyle.Top,
                Height = 90,
                Multiline = true,
                ReadOnly = false,
                ScrollBars = ScrollBars.Vertical,
                Font = new Font("Consolas", 9)
            };

            clusterBox.Text =
                "# One cluster per line\r\n" +
                "# Format: ClusterName: well1,well2,...\r\n\r\n";

            foreach (var cluster in analysisOutput.Clusters)
            {
                clusterBox.AppendText(
                    cluster.Key +
                    ": " + string.Join(",", cluster.Value) +
                    Environment.NewLine
                );
            }

            group.Controls.Add(clusterBox);

            return group;
        }

        private Control BuildVisualizationTabs()
        {
            var tabs = new TabControl
            {
                Dock = DockStyle.Fill
            };

            foreach (var item in analysisOutput.Visualizations)
            {
                string visualizationName = item.Key;
                string visualizationPath = ResolveVisualizationPath(item.Value);

                var tab = new TabPage(visualizationName);

                if (!File.Exists(visualizationPath))
                {
                    var missingLabel = new Label
                    {
                        Dock = DockStyle.Fill,
                        TextAlign = ContentAlignment.MiddleCenter,
                        Text =
                            "Visualization file not found:" +
                            Environment.NewLine +
                            visualizationPath
                    };

                    tab.Controls.Add(missingLabel);
                    tabs.TabPages.Add(tab);
                    continue;
                }

                var pictureBox = new PictureBox
                {
                    Dock = DockStyle.Fill,
                    SizeMode = PictureBoxSizeMode.Zoom,
                    Image = LoadImageWithoutLocking(visualizationPath)
                };

                tab.Controls.Add(pictureBox);
                tabs.TabPages.Add(tab);
            }

            return tabs;
        }

        private string ResolveVisualizationPath(string path)
        {
            if (Path.IsPathRooted(path))
            {
                return path;
            }

            return Path.Combine(outputPath, path);
        }

        private Image LoadImageWithoutLocking(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);

            using (var stream = new MemoryStream(bytes))
            {
                return Image.FromStream(stream);
            }
        }

        private Control BuildButtonPanel()
        {
            var panel = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.RightToLeft,
                Padding = new Padding(0, 8, 0, 0)
            };

            var closeButton = new Button
            {
                Text = "Close",
                Width = 100,
                DialogResult = DialogResult.OK
            };

            closeButton.Click += (sender, e) => Close();

            var runButton = new Button
            {
                Text = "Run",
                Width = 1000,
                DialogResult = DialogResult.OK
            };

            runButton.Click += (sender, e) => RunPrediction(sender, e, runButton);

            panel.Controls.Add(closeButton);
            panel.Controls.Add(runButton);

            return panel;
        }

        private Dictionary<string, List<int>> ParseClusters(string text)
        {
            var clusters = new Dictionary<string, List<int>>();

            var lines = text.Split(
                new[] { Environment.NewLine },
                StringSplitOptions.RemoveEmptyEntries);

            foreach (var line in lines)
            {
                var trimmed_line = line.Trim();

                if (string.IsNullOrWhiteSpace(trimmed_line))
                    continue;

                if (trimmed_line.StartsWith("#"))
                    continue;


                var parts = trimmed_line.Split(':');
                if (parts.Length != 2)
                {
                    throw new Exception($"Invalid line: {line}");
                }

                string clustername = parts[0].Trim();
                string wells_string = parts[1].Trim();

                List<int> wells = wells_string
                    .Split(',')
                    .Select(w => int.Parse(w.Trim()))
                    .ToList();

                clusters[clustername] = wells;
            }

            return clusters;
        }

        private async void RunPrediction(object sender, EventArgs e, Button btn)
        {
            try
            {
                btn.Enabled = false;

                var clusters = ParseClusters(this.clusterBox.Text);

                if (clusters.Count() < 1)
                {
                    MessageBox.Show(
                        "You must form at least one cluster",
                        "No cluster formed",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning
                   );
                    return;
                }

                var payload = new
                {
                    clusters = clusters
                };

                string workDir = Path.Combine(
                    Path.GetTempPath(), "vs_predictior_petrel"
                );

                Directory.CreateDirectory(workDir);

                string clustersPath = Path.Combine(workDir, "clusters.json");

                File.WriteAllText(
                    clustersPath,
                    JsonConvert.SerializeObject(payload, Formatting.Indented)
                );

                    var result = await Utils.RunPythonPredictionAsync(
                        pythonExe,
                        runnerPath,
                        inputPath,
                        outputPath,
                        clustersPath
                    );

                /*
                if (!string.IsNullOrWhiteSpace(result.Stdout))
                {
                    // AppendStatus(result.Stdout);
                }

                if (!string.IsNullOrWhiteSpace(result.Stderr))
                {
                    // AppendStatus(result.Stderr);
                }
                */

                if (result.ExitCode != 0)
                {
                    // AppendStatus(result.Stderr);

                    MessageBox.Show(
                        result.Stderr,
                        "Python analysis failed",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                }

                if (!File.Exists(outputPath))
                {
                    MessageBox.Show(
                        "Python finished but did not generate an output.",
                        "Missing output",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );

                    return;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    ex.ToString(),
                    "Unexpected error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
            finally
            {
                btn.Enabled = true;
            }
        }
    }
}
