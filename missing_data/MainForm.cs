using Newtonsoft.Json;
using Slb.Ocean.Petrel;
using Slb.Ocean.Petrel.DomainObject;
using Slb.Ocean.Petrel.DomainObject.Well;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace missing_data
{
    public class ClusterAnalysisOutput
    {
        [JsonProperty("status")]
        public string Status { get; set; }

        [JsonProperty("clusters")]
        public Dictionary<string, List<int>> Clusters { get; set; }

        [JsonProperty("visualizations")]
        public Dictionary<string, string> Visualizations { get; set; }
    }

    public class WellListItem
    {
        public Borehole Borehole { get; private set; }

        public WellListItem(Borehole borehole)
        {
            Borehole = borehole;
        }

        public override string ToString()
        {
            return Borehole.Name;
        }
    }

    public class PythonConfiguration
    {
        public int SequenceLength { get; set; }
        public double MaskValue { get; set; }
        public int NumEpochs { get; set; }
        public int Patience { get; set; }
        public string TargetFeature { get; set; }
    }

    public partial class MainForm : Form
    {
        // List boxes
        private CheckedListBox wellsListBoxPrediction;
        private CheckedListBox wellsListBoxTraining;

        // Prediction Configuration
        private TextBox trainedModelFolderTextBox;
        private Button browseTrainingFolderButton;

        // Configuration
        private TextBox statusTextBox;
        private Button runButton;
        private ComboBox vsComboBox;
        private ComboBox vpComboBox;
        private ComboBox rhoComboBox;
        private ComboBox grComboBox;
        private ComboBox porosityComboBox;
        private ComboBox saturationComboBox;
        private ComboBox clayComboBox;
        private ComboBox caliperComboBox;
        private TextBox outputCurveNameTextBox;

        private NumericUpDown sequenceLengthNumBox;
        private NumericUpDown maskValueNumBox;
        private NumericUpDown numEpochsNumBox;
        private NumericUpDown patienceNumBox;
        private TextBox targetFeatureTextBox;


        // Tabs
        private TabPage mainTab;
        private TabPage TrainingTab;
        private TabPage configurationTab;

        // Petrel Data
        private Project project;
        private WellRoot wellroot;

        public MainForm()
        {
            project = PetrelProject.PrimaryProject ?? throw new InvalidOperationException("No active Petrel Project.");
            wellroot = WellRoot.Get(project) ?? throw new InvalidOperationException("WellRoot not found");

            InitializeComponent();
            BuildLayout();
        }

        private void BuildLayout()
        {
            this.Text = "VS Predictor";
            this.Size = new Size(1200, 800);
            this.MinimumSize = new Size(900, 600);

            var tabs = new TabControl
            {
                Dock = DockStyle.Fill,
                Font = new Font("Consolas", 9)
            };

            mainTab = new TabPage("VS Predictor");
            TrainingTab = new TabPage("Neural Network Training");
            configurationTab = new TabPage("Configuration");

            tabs.TabPages.Add(mainTab);
            tabs.TabPages.Add(TrainingTab);
            tabs.TabPages.Add(configurationTab);

            statusTextBox = new TextBox
            {
                Dock = DockStyle.Bottom,
                Height = 150,
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                Font = new Font("Consolas", 9),
                Text = "Ready."
            };

            mainTab.Controls.Add(BuildMainSplit());
            mainTab.Controls.Add(statusTextBox);

            TrainingTab.Controls.Add(BuildTrainingSplit());

            configurationTab.Controls.Add(BuildConfigurationPanel());

            Controls.Add(tabs);
        }

        private SplitContainer BuildMainSplit()
        {
            var mainSplit = new SplitContainer
            {
                Dock = DockStyle.Fill,
                Orientation = Orientation.Vertical,
                SplitterDistance = 360,
                Padding = new Padding(12),
            };

            var leftPanel = BuildWellSelectionPanel();
            var rightPanel = BuildPredictionConfigurationPanel();

            mainSplit.Panel1.Controls.Add(leftPanel);
            mainSplit.Panel2.Controls.Add(rightPanel);

            return mainSplit;
        }

        private SplitContainer BuildTrainingSplit()
        {
            var trainingSplit = new SplitContainer
            {
                Dock = DockStyle.Fill,
                Orientation = Orientation.Vertical,
                SplitterDistance = 360,
                Padding = new Padding(12),
            };

            var leftPanel = BuildTrainingWellSelectionPanel();
            var rightPanel = BuildTrainingConfigurationPanel();

            trainingSplit.Panel1.Controls.Add(leftPanel);
            trainingSplit.Panel2.Controls.Add(rightPanel);

            return trainingSplit;
        }

        private TableLayoutPanel BuildConfigurationPanel()
        {
            var configurationPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                AutoSize = true,
            };

            configurationPanel.ColumnStyles.Add(
                new ColumnStyle(SizeType.Percent, 100));

            var pythonConfigGroup = new GroupBox
            {
                Text = "Python base_config",
                Dock = DockStyle.Top,
                AutoSize = true,
                Padding = new Padding(12)
            };

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                ColumnCount = 2
            };

            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 140));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            sequenceLengthNumBox = AddIntegerRow(layout, "Sequence length:", 0, 15, 1, 500);
            maskValueNumBox = AddDecimalRow(layout, "Mask value:", 1, -1.0m, -9999m, 9999m);
            numEpochsNumBox = AddIntegerRow(layout, "Num epochs:", 2, 500, 1, 10000);
            patienceNumBox = AddIntegerRow(layout, "Patience:", 3, 150, 1, 5000);

            targetFeatureTextBox = new TextBox
            {
                Dock = DockStyle.Fill,
                Text = "VS"
            };

            layout.Controls.Add(new Label
            {
                Text = "Target feature:",
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft
            }, 0, 4);

            layout.Controls.Add(targetFeatureTextBox, 1, 4);

            pythonConfigGroup.Controls.Add(layout);

            var curveLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                ColumnCount = 2
            };

            curveLayout.ColumnStyles.Add(
                new ColumnStyle(SizeType.Absolute, 140));

            curveLayout.ColumnStyles.Add(
                new ColumnStyle(SizeType.Percent, 100));

            var curveMappingGroup = new GroupBox
            {
                Text = "Curve mapping",
                Dock = DockStyle.Top,
                AutoSize = true,
                Padding = new Padding(12)
            };

            vpComboBox = AddCurveRow(curveLayout, "VP:", 0);
            vsComboBox = AddCurveRow(curveLayout, "VS:", 1);
            rhoComboBox = AddCurveRow(curveLayout, "RHO:", 2);
            grComboBox = AddCurveRow(curveLayout, "GR:", 3);
            porosityComboBox = AddCurveRow(curveLayout, "Porosity:", 4);
            saturationComboBox = AddCurveRow(curveLayout, "Saturation:", 5);
            clayComboBox = AddCurveRow(curveLayout, "Clay:", 6);
            caliperComboBox = AddCurveRow(curveLayout, "Caliper:", 7);

            outputCurveNameTextBox = new TextBox
            {
                Dock = DockStyle.Fill,
                Text = "VS_PREDICTED_ML"
            };

            curveLayout.Controls.Add(
                new Label
                {
                    Text = "Output curve:",
                    TextAlign = ContentAlignment.MiddleLeft
                },
                0,
                8);

            curveLayout.Controls.Add(outputCurveNameTextBox, 1, 8);

            curveMappingGroup.Controls.Add(curveLayout);

            configurationPanel.Controls.Add(curveMappingGroup);
            configurationPanel.Controls.Add(pythonConfigGroup);

            return configurationPanel;
        }

        private Control BuildWellSelectionPanel()
        {
            var group = new GroupBox
            {
                Text = "Wells",
                Dock = DockStyle.Fill,
                Padding = new Padding(10),
            };

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                RowCount = 3,
                ColumnCount = 1
            };

            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));

            var wellDropTarget = new Slb.Ocean.Petrel.UI.DropTarget
            {
                Dock = DockStyle.Fill,
                Text = "Drop wells here"
            };

            wellsListBoxPrediction = new CheckedListBox
            {
                Dock = DockStyle.Fill,
                CheckOnClick = true
            };

            LoadWellsIntoCheckBox(wellsListBoxPrediction);

            wellDropTarget.DragDrop += (sender, e) => WellDropTarget_DragDrop(sender, e, wellsListBoxPrediction);
            wellDropTarget.DragEnter += WellDropTarget_DragEnter;

            var buttons = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                Height = 40,
                FlowDirection = FlowDirection.LeftToRight
            };

            var selectAllButton = new Button { Text = "Select all", Width = 100 };
            var clearButton = new Button { Text = "Clear", Width = 100 };

            selectAllButton.Click += (sender, e) => SelectAllButton_Click(wellsListBoxPrediction);
            clearButton.Click += (sender, e) => ClearButton_Click(wellsListBoxPrediction);

            buttons.Controls.Add(selectAllButton);
            buttons.Controls.Add(clearButton);

            layout.Controls.Add(wellDropTarget, 0, 0);
            layout.Controls.Add(wellsListBoxPrediction, 0, 1);
            layout.Controls.Add(buttons, 0, 2);

            group.Controls.Add(layout);

            return group;
        }

        private Control BuildPredictionConfigurationPanel()
        {
            var group = new GroupBox
            {
                Text = "Prediction Configuration",
                Dock = DockStyle.Fill,
                Padding = new Padding(12)
            };

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                ColumnCount = 3,
                RowCount = 2
            };

            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 140));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 90));

            layout.Controls.Add(
                new Label
                {
                    Text = "Training folder:",
                    Dock = DockStyle.Fill,
                    TextAlign = ContentAlignment.MiddleLeft
                },
                0,
                0
            );

            trainedModelFolderTextBox = new TextBox
            {
                Dock = DockStyle.Fill,
                ReadOnly = true
            };

            browseTrainingFolderButton = new Button
            {
                Text = "Browse...",
                Dock = DockStyle.Fill
            };

            browseTrainingFolderButton.Click += BrowseTrainingFolderButton_Click;

            layout.Controls.Add(trainedModelFolderTextBox, 1, 0);
            layout.Controls.Add(browseTrainingFolderButton, 2, 0);

            runButton = new Button
            {
                Text = "Run prediction",
                Height = 36,
                Dock = DockStyle.Bottom
            };

            runButton.Click += async (sender, e) => await RunButton_ClickAsync(wellsListBoxPrediction);

            group.Controls.Add(runButton);
            group.Controls.Add(layout);

            return group;
        }

        private void BrowseTrainingFolderButton_Click(object sender, EventArgs e)
        {
            using (var dialog = new FolderBrowserDialog())
            {
                dialog.Description = "Select the trained experiment folder";

                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    trainedModelFolderTextBox.Text = dialog.SelectedPath;
                }
            }
        }

        private Control BuildTrainingWellSelectionPanel()
        {
            var group = new GroupBox
            {
                Text = "Wells",
                Dock = DockStyle.Fill,
                Padding = new Padding(10),
            };

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                RowCount = 3,
                ColumnCount = 1
            };

            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 15));

            var wellDropTarget = new Slb.Ocean.Petrel.UI.DropTarget
            {
                Dock = DockStyle.Fill,
                Text = "Drop wells here"
            };

            wellsListBoxTraining = new CheckedListBox
            {
                Dock = DockStyle.Fill,
                CheckOnClick = true
            };

            LoadWellsIntoCheckBox(wellsListBoxTraining);

            var buttons = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                Height = 40,
                FlowDirection = FlowDirection.LeftToRight
            };

            var selectAllButton = new Button { Text = "Select all", Width = 100 };
            var clearButton = new Button { Text = "Clear", Width = 100 };

            selectAllButton.Click += (sender, e) => SelectAllButton_Click(wellsListBoxTraining);
            clearButton.Click += (sender, e) => ClearButton_Click(wellsListBoxTraining);

            wellDropTarget.DragDrop += (sender, e) => WellDropTarget_DragDrop(sender, e, wellsListBoxTraining);
            wellDropTarget.DragEnter += WellDropTarget_DragEnter;

            buttons.Controls.Add(selectAllButton);
            buttons.Controls.Add(clearButton);

            layout.Controls.Add(wellDropTarget, 0, 0);
            layout.Controls.Add(wellsListBoxTraining, 0, 1);
            layout.Controls.Add(buttons, 0, 2);

            group.Controls.Add(layout);

            return group;
        }

        private Control BuildTrainingConfigurationPanel()
        {
            var group = new GroupBox
            {
                Text = "Training Configuration",
                Dock = DockStyle.Fill,
                Padding = new Padding(12)
            };

            var container = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 2
            };

            container.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            container.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            container.RowStyles.Add(new RowStyle(SizeType.Absolute, 12));

            runButton = new Button
            {
                Text = "Run Training",
                Height = 36,
                Dock = DockStyle.Bottom
            };

            runButton.Click += async (sender, e) => await RunButtonTraining_Click(wellsListBoxTraining);

            container.Controls.Add(runButton, 0, 0);

            group.Controls.Add(container);

            return group;
        }


        private ComboBox AddCurveRow(TableLayoutPanel layout, string label, int row)
        {
            var combo = new ComboBox
            {
                Dock = DockStyle.Fill,
                DropDownStyle = ComboBoxStyle.DropDownList
            };

            layout.Controls.Add(
                new Label
                {
                    Text = label,
                    TextAlign = ContentAlignment.MiddleLeft,
                    Dock = DockStyle.Fill
                },
                0,
                row
            );

            layout.Controls.Add(combo, 1, row);

            combo.Items.AddRange(LoadLogHeaders());

            return combo;
        }

        private void LoadWellsIntoCheckBox(CheckedListBox clBox)
        {
            clBox.Items.Clear();

            var boreholeCollections =
                wellroot.BoreholeCollection?.BoreholeCollections
                ?? throw new InvalidOperationException("No borehole collections found.");

            foreach (var collection in boreholeCollections)
            {
                foreach (var borehole in collection)
                {
                    clBox.Items.Add(new WellListItem(borehole), false);
                }
            }
        }

        private string[] LoadLogHeaders()
        {
            var boreholeCollections = wellroot.BoreholeCollection?.BoreholeCollections
                ?? throw new InvalidOperationException("No borehole collections found.");

            var logNames = new List<string>();

            foreach (var collection in boreholeCollections)
            {
                foreach (var borehole in collection)
                {
                    foreach (var log in borehole.Logs.WellLogs)
                    {
                        logNames.Add(log.Name);
                    }
                }
            }

            return logNames
                .Distinct()
                .OrderBy(name => name)
                .ToArray();
        }


        private async Task RunButtonTraining_Click(CheckedListBox clb)
        {
            try
            {
                var selectedWells = GetSelectedWells(clb);

                if (selectedWells.Count() == 0)
                {
                    MessageBox.Show("Select at least one well.", "No wells selected", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
                var selectedTrainingConfiguration = getCurveMapping();
                var pythonConfiguration = GetPythonConfiguration();
                
                string workDir = Path.Combine(
                    Path.GetTempPath(), "vs_predictior_petrel"
                );

                Directory.CreateDirectory(workDir);

                string inputPath = Path.Combine(workDir, "cluster_analysis_input.json");
                string outputPath = Path.Combine(workDir, "cluster_analysis_output.json");

                var payload = BuildClusterAnalysisPayload(selectedWells, selectedTrainingConfiguration, pythonConfiguration);

                string json = JsonConvert.SerializeObject(payload, Formatting.Indented);

                File.WriteAllText(inputPath, json);

                string appData =
                    Environment.GetFolderPath(
                    Environment.SpecialFolder.ApplicationData
                );

                string projectDir = Path.Combine(
                    appData,
                    "recriando_tcc_caio_pra"
                );

                string pythonExe = Path.Combine(
                    projectDir,
                    ".venv",
                    "Scripts",
                    "python.exe"
                );

                string runnerPath = Path.Combine(
                    projectDir,
                    "predictor.py"
                );

                AppendStatus(
                    "Python executable: " + pythonExe
                );

                AppendStatus(
                    "Runner script: " + runnerPath
                );

                if (!File.Exists(pythonExe))
                {
                    throw new FileNotFoundException(
                        "Python executable not found.",
                        pythonExe
                    );
                }

                if (!File.Exists(runnerPath))
                {
                    throw new FileNotFoundException(
                        "Predictor script not found.",
                        runnerPath
                    );
                }

                var result = await Utils.RunPythonAnalysisAsync(
                    pythonExe,
                    runnerPath,
                    "analyze",
                    inputPath,
                    outputPath
                );

                if (!string.IsNullOrWhiteSpace(result.Stdout))
                {
                    // AppendStatus(result.Stdout);
                }

                if (!string.IsNullOrWhiteSpace(result.Stderr))
                {
                    // AppendStatus(result.Stderr);
                }

                if (result.ExitCode != 0)
                {
                    // AppendStatus(result.Stderr);

                    MessageBox.Show(
                        result.Stderr,
                        "Python analysis failed",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                        );

                    return;
                }

                if (!File.Exists(outputPath))
                {
                    MessageBox.Show(
                        "Python finished but did not generate output JSON.",
                        "Missing output",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );

                    return;
                }

                // AppendStatus("Reading cluster analysis result...");

                string outputJson = File.ReadAllText(outputPath);

                var analysisOutput = JsonConvert.DeserializeObject<ClusterAnalysisOutput>(
                    outputJson
                );

                if (analysisOutput == null)
                {
                    throw new InvalidOperationException(
                        "Could not parse cluster analysis output JSON."
                    );
                }

                if (analysisOutput.Status != "success")
                {
                    throw new InvalidOperationException(
                        "Cluster analysis did not finish successfully."
                    );
                }

                AppendStatus("Clusters found: " + analysisOutput.Clusters.Count);

                var form = new ClusterAnalysisVisualizationForm(analysisOutput);

                if (form.ShowDialog() != DialogResult.OK)
                    return;

                var clusters = form.EditedClusters;

                var payload_clusters = new
                {
                    clusters
                };

                string workDir_clusters = Path.Combine(
                    Path.GetTempPath(), "vs_predictior_petrel"
                );

                Directory.CreateDirectory(workDir_clusters);

                string clustersPath = Path.Combine(workDir_clusters, "clusters.json");

                File.WriteAllText(
                    clustersPath,
                    JsonConvert.SerializeObject(payload_clusters, Formatting.Indented)
                );

                    var result_clusters = await Utils.RunPythonTrainingAsync(
                        pythonExe,
                        runnerPath,
                        inputPath,
                        outputPath,
                        clustersPath
                    );

                if (!string.IsNullOrWhiteSpace(result_clusters.Stdout))
                {
                    AppendStatus(result_clusters.Stdout);
                }

                if (!string.IsNullOrWhiteSpace(result_clusters.Stderr))
                {
                    AppendStatus(result_clusters.Stderr);
                }

                if (result_clusters.ExitCode != 0)
                {
                     AppendStatus(result_clusters.Stderr);

                    MessageBox.Show(
                        result_clusters.Stderr,
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
                // AppendStatus("ERROR: " + ex.Message);

                MessageBox.Show(
                    ex.ToString(),
                    "Unexpected error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
            finally
            {
                runButton.Enabled = true;
            }
        }

        private async Task RunButton_ClickAsync(CheckedListBox clb)
        {
            if (string.IsNullOrWhiteSpace(trainedModelFolderTextBox.Text))
            {
                MessageBox.Show(
                    "Please select a trained model folder.",
                    "Missing folder",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);

                return;
            }

            try
            {
                runButton.Enabled = false;
                AppendStatus("Starting cluster analysis...");

                var selectedWells = GetSelectedWells(clb);

                if (selectedWells.Count() == 0)
                {
                    MessageBox.Show(
                        "Select at leat one well.",
                        "No wells selected",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning
                    );

                    return;
                }

                var selectedPredictionConfiguration = getCurveMapping();
                var pythonConfiguration = GetPythonConfiguration();

                AppendStatus("Selected wells: " + selectedWells.Count().ToString());

                string workDir = Path.Combine(
                    Path.GetTempPath(), "vs_predictior_petrel"
                );

                Directory.CreateDirectory(workDir);

                string inputPath = Path.Combine(workDir, "cluster_analysis_input.json");
                string outputPath = Path.Combine(workDir, "cluster_analysis_output.json");

                AppendStatus("Exporting selected wells to JSON...");

                var payload = BuildClusterAnalysisPayload(selectedWells, selectedPredictionConfiguration, pythonConfiguration);

                string json = JsonConvert.SerializeObject(payload, Formatting.Indented);

                File.WriteAllText(inputPath, json);

                AppendStatus("Data saved on: " + inputPath);

                string appData =
                    Environment.GetFolderPath(
                    Environment.SpecialFolder.ApplicationData
                );

                string projectDir = Path.Combine(
                    appData,
                    "recriando_tcc_caio_pra"
                );

                string pythonExe = Path.Combine(
                    projectDir,
                    ".venv",
                    "Scripts",
                    "python.exe"
                );

                string runnerPath = Path.Combine(
                    projectDir,
                    "predictor.py"
                );

                AppendStatus(
                    "Python executable: " + pythonExe
                );

                AppendStatus(
                    "Runner script: " + runnerPath
                );

                if (!File.Exists(pythonExe))
                {
                    throw new FileNotFoundException(
                        "Python executable not found.",
                        pythonExe
                    );
                }

                if (!File.Exists(runnerPath))
                {
                    throw new FileNotFoundException(
                        "Predictor script not found.",
                        runnerPath
                    );
                }

                var result = await Utils.RunPythonAnalysisAsync(
                    pythonExe,
                    runnerPath,
                    "analyze",
                    inputPath,
                    outputPath
                );

                if (!string.IsNullOrWhiteSpace(result.Stdout))
                {
                    AppendStatus(result.Stdout);
                }

                if (!string.IsNullOrWhiteSpace(result.Stderr))
                {
                    AppendStatus(result.Stderr);
                }

                if (result.ExitCode != 0)
                {
                    AppendStatus(result.Stderr);

                    MessageBox.Show(
                        result.Stderr,
                        "Python analysis failed",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                        );

                    return;
                }

                if (!File.Exists(outputPath))
                {
                    MessageBox.Show(
                        "Python finished but did not generate output JSON.",
                        "Missing output",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );

                    return;
                }

                AppendStatus("Reading cluster analysis result...");

                string outputJson = File.ReadAllText(outputPath);

                var analysisOutput = JsonConvert.DeserializeObject<ClusterAnalysisOutput>(
                    outputJson
                );

                if (analysisOutput == null)
                {
                    throw new InvalidOperationException(
                        "Could not parse cluster analysis output JSON."
                    );
                }

                if (analysisOutput.Status != "success")
                {
                    throw new InvalidOperationException(
                        "Cluster analysis did not finish successfully."
                    );
                }

                AppendStatus("Clusters found: " + analysisOutput.Clusters.Count);

                var form = new ClusterAnalysisVisualizationForm(analysisOutput);

                if (form.ShowDialog() != DialogResult.OK)
                    return;

                var clusters = form.EditedClusters;

                var payload_clusters = new
                {
                    clusters
                };

                string workDir_clusters = Path.Combine(
                    Path.GetTempPath(), "vs_predictior_petrel"
                );

                Directory.CreateDirectory(workDir_clusters);

                string clustersPath = Path.Combine(workDir_clusters, "clusters.json");

                File.WriteAllText(
                    clustersPath,
                    JsonConvert.SerializeObject(payload_clusters, Formatting.Indented)
                );

                    var result_clusters = await Utils.RunPythonPredictionAsync(
                        pythonExe,
                        runnerPath,
                        inputPath,
                        outputPath,
                        clustersPath,
                        trainedModelFolderTextBox.Text
                    );

                if (!string.IsNullOrWhiteSpace(result_clusters.Stdout))
                {
                    // AppendStatus(result.Stdout);
                }

                if (!string.IsNullOrWhiteSpace(result_clusters.Stderr))
                {
                    // AppendStatus(result.Stderr);
                }

                if (result_clusters.ExitCode != 0)
                {
                    // AppendStatus(result.Stderr);

                    MessageBox.Show(
                        result_clusters.Stderr,
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
                AppendStatus("ERROR: " + ex.Message);

                MessageBox.Show(
                    ex.ToString(),
                    "Unexpected error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
            finally
            {
                runButton.Enabled = true;
            }
        }

        private void AppendStatus(string message)
        {
            if (statusTextBox == null)
            {
                return;
            }
            else
            {
                statusTextBox.AppendText(
                    Environment.NewLine + "[" + DateTime.Now.ToString("HH:mm:ss") + "] " + message);
            }
        }

        private void SelectAllButton_Click(CheckedListBox clb)
        {
            for (int i = 0; i < clb.Items.Count; i++)
            {
                clb.SetItemChecked(i, true);
            }
        }

        private void ClearButton_Click(CheckedListBox clb)
        {
            for (int i = 0; i < clb.Items.Count; i++)
            {
                clb.SetItemChecked(i, false);
            }
        }

        private void WellDropTarget_DragDrop(object sender, DragEventArgs e, CheckedListBox clb)
        {
            string[] formats = e.Data.GetFormats();

            if (formats == null || formats.Length == 0)
            {
                PetrelLogger.InfoOutputWindow("Drop ignored: no data formats.");
                return;
            }

            object data = null;

            foreach (string format in formats)
            {
                data = e.Data.GetData(format);

                if (data is System.Collections.ArrayList)
                {
                    break;
                }
            }

            var droppedItems = data as System.Collections.ArrayList;

            if (droppedItems == null)
            {
                PetrelLogger.InfoOutputWindow("Drop ignored: no ArrayList found.");
                return;
            }

            if (droppedItems == null)
            {
                return;
            }

            foreach (object item in droppedItems)
            {

                var borehole = item as Borehole;

                if (borehole == null)
                {
                    continue;
                }

                var well_name = borehole.Name;

                for (int i = 0; i < clb.Items.Count; i++)
                {
                    object list_item = clb.Items[i];

                    if (list_item == null)
                    {
                        continue;
                    }

                    var item_name = clb.GetItemText(list_item);


                    PetrelLogger.InfoOutputWindow(
                        item_name + " " + well_name
                    );

                    if (string.Equals(item_name, well_name, StringComparison.OrdinalIgnoreCase))
                    {
                        bool currentState = clb.GetItemChecked(i);
                        clb.SetItemChecked(i, !currentState);
                        break;
                    }
                }
            }
        }

        private void WellDropTarget_DragEnter(object sender, DragEventArgs e)
        {
            string[] formats = e.Data.GetFormats();

            if (formats != null && formats.Length > 0)
            {
                e.Effect = DragDropEffects.Copy;
            }
            else
            {
                e.Effect = DragDropEffects.None;
            }
        }

        private NumericUpDown AddIntegerRow(
            TableLayoutPanel layout,
            string label,
            int row,
            int defaultValue,
            int min,
            int max)
        {
            var input = new NumericUpDown
            {
                Dock = DockStyle.Fill,
                Minimum = min,
                Maximum = max,
                Value = defaultValue,
                DecimalPlaces = 0
            };

            layout.Controls.Add(
                new Label
                {
                    Text = label,
                    TextAlign = ContentAlignment.MiddleLeft,
                    Dock = DockStyle.Fill
                },
                0,
                row
            );

            layout.Controls.Add(input, 1, row);

            return input;
        }

        private string GetRequiredTextValue(
            TextBox textBox,
            string fieldName)
        {
            if (textBox == null)
            {
                throw new InvalidOperationException(
                    fieldName + " control was not initialized."
                );
            }

            string value = textBox.Text.Trim();

            if (string.IsNullOrWhiteSpace(value))
            {
                throw new InvalidOperationException(
                    fieldName + " cannot be empty."
                );
            }

            return value;
        }

        private string GetRequiredComboValue(
            ComboBox comboBox,
            string fieldName)
        {
            if (comboBox == null)
            {
                throw new InvalidOperationException(
                    fieldName + " control was not initialized."
                );
            }

            if (comboBox.SelectedItem == null)
            {
                throw new InvalidOperationException(
                    "Please select a value for " + fieldName + "."
                );
            }

            string value = comboBox.SelectedItem.ToString();

            if (string.IsNullOrWhiteSpace(value))
            {
                throw new InvalidOperationException(
                    fieldName + " cannot be empty."
                );
            }

            return value;
        }

        private List<Borehole> GetSelectedWells(CheckedListBox clb)
        {
            var selected = new List<Borehole>();

            foreach (object item in clb.CheckedItems)
            {
                if (item is WellListItem wellItem)
                {
                    selected.Add(wellItem.Borehole);
                }
            }

            return selected;
        }


        private Dictionary<String, String> getCurveMapping()
        {
            var selected = new Dictionary<string, string>
            {
                { "VS", GetRequiredComboValue(vsComboBox, "VS") },
                { "VP", GetRequiredComboValue(vpComboBox, "VP") },
                { "RHO", GetRequiredComboValue(rhoComboBox, "RHO") },
                { "GR", GetRequiredComboValue(grComboBox, "GR") },
                { "POROSIDADE", GetRequiredComboValue(porosityComboBox, "Porosity") },
                { "SATURACAO", GetRequiredComboValue(saturationComboBox, "Saturation") },
                { "ARGILOSIDADE", GetRequiredComboValue(clayComboBox, "Clay") },
                { "CALIPER", GetRequiredComboValue(caliperComboBox, "Caliper") },
                { "OUTPUT_CURVE", GetRequiredTextValue(outputCurveNameTextBox, "Output curve") }
            };

            return selected;
        }

        private object BuildClusterAnalysisPayload(List<Borehole> selectedWells, Dictionary<String, String> curveMapping, PythonConfiguration pythonConfiguration)
        {
            var wells = new List<object>();

            foreach (var borehole in selectedWells)
            {
                wells.Add(new
                {
                    name = borehole.Name,
                    logs = ExtractWellLogSamples(borehole)
                });
            }

            return new
            {
                curveMapping,
                wells,
                pythonConfiguration
            };
        }


        private List<Object> ExtractWellLogSamples(Borehole borehole)
        {
            var logs = new List<Object>();

            foreach (var log in borehole.Logs.WellLogs)
            {
                var samples = log.Samples.Select(s => new {
                    md = s.MD,
                    value = s.Value
                }).ToList();

                logs.Add(new
                {
                    name = log.Name,
                    samples
                });
            }

            return logs;
        }


        private NumericUpDown AddDecimalRow(
            TableLayoutPanel layout,
            string label,
            int row,
            decimal defaultValue,
            decimal min,
            decimal max)
        {
            var input = new NumericUpDown
            {
                Dock = DockStyle.Fill,
                Minimum = min,
                Maximum = max,
                Value = defaultValue,
                DecimalPlaces = 2,
                Increment = 0.1m
            };

            layout.Controls.Add(
                new Label
                {
                    Text = label,
                    TextAlign = ContentAlignment.MiddleLeft,
                    Dock = DockStyle.Fill
                },
                0,
                row
            );

            layout.Controls.Add(input, 1, row);

            return input;
        }

        private PythonConfiguration GetPythonConfiguration()
        {
            return new PythonConfiguration
            {
                SequenceLength = int.Parse(sequenceLengthNumBox.Text),
                MaskValue = double.Parse(maskValueNumBox.Text),
                NumEpochs = int.Parse(numEpochsNumBox.Text),
                Patience = int.Parse(patienceNumBox.Text),
                TargetFeature = targetFeatureTextBox.Text
            };
        }
    }
}