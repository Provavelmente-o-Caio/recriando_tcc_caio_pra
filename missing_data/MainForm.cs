using Newtonsoft.Json;
using Slb.Ocean.Petrel;
using Slb.Ocean.Petrel.DomainObject;
using Slb.Ocean.Petrel.DomainObject.Well;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Newtonsoft.Json.Linq;

namespace missing_data
{
    public class ClusterAnalysisOutput
    {
        [JsonProperty("schema_version")]
        public int SchemaVersion { get; set; }

        [JsonProperty("job_id")]
        public string JobId { get; set; }

        [JsonProperty("status")]
        public string Status { get; set; }

        [JsonProperty("clusters")]
        public Dictionary<string, List<int>> Clusters { get; set; }

        [JsonProperty("visualizations")]
        public Dictionary<string, string> Visualizations { get; set; }
    }

    public class PredictionResultOutput
    {
        [JsonProperty("schema_version")]
        public int SchemaVersion { get; set; }

        [JsonProperty("status")]
        public string Status { get; set; }

        [JsonProperty("job_id")]
        public string JobId { get; set; }

        [JsonProperty("predictions_file")]
        public string PredictionsFile { get; set; }

        [JsonProperty("summary_report_file")]
        public string SummaryReportFile { get; set; }

        [JsonProperty("plots_dir")]
        public string PlotsDirectory { get; set; }

        [JsonProperty("well_count")]
        public int WellCount { get; set; }

        [JsonProperty("wells_with_ground_truth")]
        public int WellsWithGroundTruth { get; set; }

        [JsonProperty("wells_without_ground_truth")]
        public int WellsWithoutGroundTruth { get; set; }
    }

    public class TrainingResultOutput
    {
        [JsonProperty("schema_version")]
        public int SchemaVersion { get; set; }

        [JsonProperty("job_id")]
        public string JobId { get; set; }

        [JsonProperty("status")]
        public string Status { get; set; }

        [JsonProperty("experiment_dir")]
        public string ExperimentDirectory { get; set; }

        [JsonProperty("final_model_dir")]
        public string FinalModelDirectory { get; set; }
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

    public class UnitConfiguration
    {
        public string VP { get; set; }
        public string VS { get; set; }
        public string PythonVelocityUnit { get; set; }
        public string PayloadVelocityUnit { get; set; }
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
        private TextBox trainingStatusTextBox;
        private Button predictionButton;
        private Button cancelPredictionButton;
        private ProgressBar predictionProgressBar;
        private Button trainingButton;
        private Button cancelTrainingButton;
        private ProgressBar trainingProgressBar;
        private CancellationTokenSource predictionCancellationSource;
        private CancellationTokenSource trainingCancellationSource;
        private readonly PythonRuntimeService pythonRuntimeService;
        private ComboBox vsComboBox;
        private ComboBox vpComboBox;
        private ComboBox rhoComboBox;
        private ComboBox grComboBox;
        private ComboBox porosityComboBox;
        private ComboBox saturationComboBox;
        private ComboBox clayComboBox;
        private ComboBox caliperComboBox;
        private TextBox outputCurveNameTextBox;
        private ComboBox vpUnitComboBox;
        private ComboBox vsUnitComboBox;

        private NumericUpDown sequenceLengthNumBox;
        private NumericUpDown maskValueNumBox;
        private NumericUpDown numEpochsNumBox;
        private NumericUpDown patienceNumBox;
        private NumericUpDown optunaJobsNumBox;
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

            pythonRuntimeService = new PythonRuntimeService();
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
            trainingStatusTextBox = new TextBox
            {
                Dock = DockStyle.Bottom,
                Height = 150,
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                Font = new Font("Consolas", 9),
                Text = "Ready."
            };
            TrainingTab.Controls.Add(trainingStatusTextBox);

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

            int processorCount = Math.Max(1, Environment.ProcessorCount);
            var optunaLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                ColumnCount = 2
            };
            optunaLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 140));
            optunaLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            optunaJobsNumBox = AddIntegerRow(
                optunaLayout,
                "Processes:",
                0,
                processorCount,
                1,
                processorCount);

            var optunaGroup = new GroupBox
            {
                Text = "Optuna",
                Dock = DockStyle.Top,
                AutoSize = true,
                Padding = new Padding(12)
            };
            optunaGroup.Controls.Add(optunaLayout);

            var unitsLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                ColumnCount = 2
            };
            unitsLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 140));
            unitsLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            vpUnitComboBox = AddUnitRow(unitsLayout, "VP unit:", 0);
            vsUnitComboBox = AddUnitRow(unitsLayout, "VS unit:", 1);

            var unitsGroup = new GroupBox
            {
                Text = "Velocity units",
                Dock = DockStyle.Top,
                AutoSize = true,
                Padding = new Padding(12)
            };
            unitsGroup.Controls.Add(unitsLayout);

            var configurationTabs = new TabControl
            {
                Dock = DockStyle.Fill
            };
            var generalTab = new TabPage("General");
            var generalPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                AutoSize = true
            };
            generalPanel.Controls.Add(curveMappingGroup);
            generalPanel.Controls.Add(pythonConfigGroup);
            generalPanel.Controls.Add(optunaGroup);
            generalTab.Controls.Add(generalPanel);

            var unitsTab = new TabPage("Units");
            unitsTab.Controls.Add(unitsGroup);

            configurationTabs.TabPages.Add(generalTab);
            configurationTabs.TabPages.Add(unitsTab);
            configurationPanel.Controls.Add(configurationTabs);

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
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 15));

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
                RowCount = 3
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

            predictionButton = new Button
            {
                Text = "Run prediction",
                Height = 36,
                Dock = DockStyle.Fill
            };

            cancelPredictionButton = new Button
            {
                Text = "Cancel",
                Height = 36,
                Dock = DockStyle.Fill,
                Enabled = false
            };

            predictionButton.Click += async (sender, e) => await RunButton_ClickAsync(wellsListBoxPrediction);
            cancelPredictionButton.Click += (sender, e) =>
            {
                predictionCancellationSource?.Cancel();
                AppendStatus("Prediction cancellation requested...");
                cancelPredictionButton.Enabled = false;
            };

            var actionPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Bottom,
                ColumnCount = 1,
                RowCount = 3,
                Padding = new Padding(0)
            };
            actionPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 70));
            actionPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 30));
            actionPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            actionPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            actionPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 10));
            actionPanel.Controls.Add(predictionButton, 0, 0);
            actionPanel.Controls.Add(cancelPredictionButton, 0, 1);

            predictionProgressBar = new ProgressBar
            {
                Style = ProgressBarStyle.Continuous,
                Minimum = 0,
                Maximum = 100,
                Dock = DockStyle.Fill,
                Visible = false
            };
            actionPanel.Controls.Add(predictionProgressBar, 0, 2);

            group.Controls.Add(actionPanel);
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
                    string validationError = ValidateTrainedModelFolder(dialog.SelectedPath);
                    if (!string.IsNullOrWhiteSpace(validationError))
                    {
                        MessageBox.Show(
                            validationError,
                            "Invalid trained model folder",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Warning);
                        return;
                    }

                    trainedModelFolderTextBox.Text = dialog.SelectedPath;
                    AppendStatus("Validated trained model folder: " + dialog.SelectedPath);
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
                Dock = DockStyle.Bottom,
                ColumnCount = 1,
                RowCount = 3,
                Padding = new Padding(0)
            };

            container.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            container.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
            container.RowStyles.Add(new RowStyle(SizeType.Absolute, 10));

            trainingButton = new Button
            {
                Text = "Run Training",
                Height = 36,
                Dock = DockStyle.Fill
            };

            cancelTrainingButton = new Button
            {
                Text = "Cancel",
                Height = 36,
                Dock = DockStyle.Fill,
                Enabled = false
            };

            trainingButton.Click += async (sender, e) => await RunButtonOptunaTraining_Click(wellsListBoxTraining);
            cancelTrainingButton.Click += (sender, e) =>
            {
                trainingCancellationSource?.Cancel();
                AppendTrainingStatus("Cancellation requested...");
                cancelTrainingButton.Enabled = false;
            };

            container.Controls.Add(trainingButton, 0, 0);
            container.Controls.Add(cancelTrainingButton, 0, 1);
            trainingProgressBar = new ProgressBar
            {
                Style = ProgressBarStyle.Continuous,
                Minimum = 0,
                Maximum = 100,
                Dock = DockStyle.Fill,
                Visible = false
            };
            container.Controls.Add(trainingProgressBar, 0, 2);

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

        private ComboBox AddUnitRow(TableLayoutPanel layout, string label, int row)
        {
            var combo = new ComboBox
            {
                Dock = DockStyle.Fill,
                DropDownStyle = ComboBoxStyle.DropDownList
            };
            combo.Items.Add("m/s");
            combo.Items.Add("km/s");
            combo.SelectedItem = "m/s";

            layout.Controls.Add(
                new Label
                {
                    Text = label,
                    TextAlign = ContentAlignment.MiddleLeft,
                    Dock = DockStyle.Fill
                },
                0,
                row);
            layout.Controls.Add(combo, 1, row);

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
                trainingCancellationSource = new CancellationTokenSource();
                CancellationToken cancellationToken = trainingCancellationSource.Token;
                trainingButton.Enabled = false;
                cancelTrainingButton.Enabled = true;
                trainingProgressBar.Visible = true;

                var selectedWells = GetSelectedWells(clb);

                if (selectedWells.Count() == 0)
                {
                    MessageBox.Show("Select at least one well.", "No wells selected", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
                var selectedTrainingConfiguration = getCurveMapping();
                var pythonConfiguration = GetPythonConfiguration();
                
                string jobId;
                string workDir = CreateJobDirectory("training", out jobId);
                string inputPath = Path.Combine(workDir, "training_input.json");
                string analysisOutputPath = Path.Combine(workDir, "analysis_output.json");
                string outputPath = Path.Combine(workDir, "training_result.json");

                var payload = BuildClusterAnalysisPayload(selectedWells, selectedTrainingConfiguration, pythonConfiguration);

                string json = JsonConvert.SerializeObject(payload, Formatting.Indented);

                File.WriteAllText(inputPath, json);

                AppendStatus("Python executable: " + pythonRuntimeService.PythonExecutable);
                AppendStatus("Runner script: " + pythonRuntimeService.PredictorScript);

                var result = await RunPythonWithProgressAsync(
                    () => pythonRuntimeService.AnalyzeAsync(
                        inputPath,
                        analysisOutputPath,
                        jobId,
                        cancellationToken),
                    analysisOutputPath,
                    jobId,
                    trainingProgressBar,
                    AppendStatus);

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

                if (!File.Exists(analysisOutputPath))
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

                string outputJson = File.ReadAllText(analysisOutputPath);

                var analysisOutput = JsonConvert.DeserializeObject<ClusterAnalysisOutput>(
                    outputJson
                );

                if (analysisOutput == null
                    || analysisOutput.SchemaVersion != 1
                    || analysisOutput.JobId != jobId)
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

                string clusterValidationError = ValidateEditedClusters(
                    clusters,
                    selectedWells,
                    selectedTrainingConfiguration);
                if (!string.IsNullOrWhiteSpace(clusterValidationError))
                {
                    MessageBox.Show(
                        clusterValidationError,
                        "Invalid clusters",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning);
                    return;
                }

                var payload_clusters = new
                {
                    clusters
                };

                string clustersPath = Path.Combine(workDir, "clusters.json");

                File.WriteAllText(
                    clustersPath,
                    JsonConvert.SerializeObject(payload_clusters, Formatting.Indented)
                );

                    var result_clusters = await RunPythonWithProgressAsync(
                        () => pythonRuntimeService.TrainAsync(
                            inputPath,
                            outputPath,
                            clustersPath,
                            jobId,
                            cancellationToken,
                            (line, isError) => AppendStatus(line)),
                        outputPath,
                        jobId,
                        trainingProgressBar,
                        AppendStatus);

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
                    return;
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

                var trainingOutput = JsonConvert.DeserializeObject<TrainingResultOutput>(
                    File.ReadAllText(outputPath));
                if (trainingOutput == null
                    || trainingOutput.SchemaVersion != 1
                    || trainingOutput.Status != "success"
                    || trainingOutput.JobId != jobId)
                {
                    throw new InvalidOperationException(
                        "Training produced an invalid or stale result contract.");
                }

                AppendStatus("Training completed for job " + jobId + ".");

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
                trainingButton.Enabled = true;
                cancelTrainingButton.Enabled = false;
                trainingProgressBar.Visible = false;
                trainingCancellationSource?.Dispose();
                trainingCancellationSource = null;
            }
        }

        private async Task RunButtonOptunaTraining_Click(CheckedListBox clb)
        {
            try
            {
                if (GetSelectedWells(clb).Count == 0)
                {
                    MessageBox.Show(
                        "Select at least one well to confirm that the Petrel project is ready. "
                        + "Optuna training uses the wells selected in this interface.",
                        "No wells selected",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning);
                    return;
                }

                var selectedWells = GetSelectedWells(clb);
                trainingButton.Enabled = false;
                cancelTrainingButton.Enabled = true;
                trainingProgressBar.Visible = true;
                trainingProgressBar.BringToFront();
                trainingProgressBar.Refresh();
                trainingCancellationSource = new CancellationTokenSource();
                CancellationToken cancellationToken = trainingCancellationSource.Token;
                var selectedConfiguration = getCurveMapping();
                var pythonConfiguration = GetPythonConfiguration();
                string jobId;
                string workDir = CreateJobDirectory("optuna_training", out jobId);
                string inputPath = Path.Combine(workDir, "optuna_training_input.json");
                string clustersPath = Path.Combine(workDir, "optuna_training_clusters.json");
                string analysisOutputPath = Path.Combine(workDir, "optuna_cluster_analysis_output.json");
                var payload = BuildClusterAnalysisPayload(
                    selectedWells,
                    selectedConfiguration,
                    pythonConfiguration);
                File.WriteAllText(
                    inputPath,
                    JsonConvert.SerializeObject(payload, Formatting.Indented));

                AppendTrainingStatus("Analyzing clusters before Optuna training...");
                var analysisResult = await RunPythonWithProgressAsync(
                    () => pythonRuntimeService.AnalyzeAsync(
                        inputPath,
                        analysisOutputPath,
                        jobId,
                        cancellationToken,
                        (line, isError) => AppendTrainingStatus(line)),
                    analysisOutputPath,
                    jobId,
                    trainingProgressBar,
                    AppendTrainingStatus);

                if (analysisResult.ExitCode != 0)
                {
                    MessageBox.Show(
                        analysisResult.Stderr,
                        "Cluster analysis failed",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error);
                    return;
                }

                if (!File.Exists(analysisOutputPath))
                    throw new InvalidOperationException(
                        "Cluster analysis finished without generating an output JSON.");

                var analysisOutput = JsonConvert.DeserializeObject<ClusterAnalysisOutput>(
                    File.ReadAllText(analysisOutputPath));

                if (analysisOutput == null
                    || analysisOutput.SchemaVersion != 1
                    || analysisOutput.JobId != jobId
                    || analysisOutput.Status != "success")
                    throw new InvalidOperationException(
                        "Cluster analysis did not finish successfully.");

                AppendTrainingStatus(
                    "Clusters found: " + analysisOutput.Clusters.Count);

                var clusterForm = new ClusterAnalysisVisualizationForm(analysisOutput);
                if (clusterForm.ShowDialog() != DialogResult.OK)
                    return;

                string clusterValidationError = ValidateEditedClusters(
                    clusterForm.EditedClusters,
                    selectedWells,
                    selectedConfiguration);
                if (!string.IsNullOrWhiteSpace(clusterValidationError))
                {
                    MessageBox.Show(
                        clusterValidationError,
                        "Invalid clusters",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning);
                    return;
                }

                File.WriteAllText(
                    clustersPath,
                    JsonConvert.SerializeObject(
                        new
                        {
                            clusters = clusterForm.EditedClusters
                        },
                        Formatting.Indented));
                string outputPath = Path.Combine(workDir, "optuna_results");

                AppendTrainingStatus("Starting Optuna training with the analyzed clusters...");
                AppendTrainingStatus("Selected wells: " + selectedWells.Count);
                AppendTrainingStatus("Clusters JSON: " + clustersPath);
                AppendTrainingStatus("Input JSON: " + inputPath);
                AppendTrainingStatus("Optuna script: " + pythonRuntimeService.OptunaScript);
                AppendTrainingStatus("Output directory: " + outputPath);
                int optunaJobs = (int)optunaJobsNumBox.Value;
                AppendTrainingStatus("Optuna jobs: " + optunaJobs);

                var result = await RunPythonWithProgressAsync(
                    () => pythonRuntimeService.RunOptunaTrainingAsync(
                        inputPath,
                        clustersPath,
                        outputPath,
                        trials: 30,
                        jobs: optunaJobs,
                        jobId: jobId,
                        cancellationToken: cancellationToken,
                        outputCallback: (line, isError) => AppendTrainingStatus(line)),
                    outputPath,
                    jobId,
                    trainingProgressBar,
                    AppendTrainingStatus);

                if (result.ExitCode != 0)
                {
                    MessageBox.Show(
                        result.Stderr,
                        "Optuna training failed",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error);
                    return;
                }

                string trainingContractPath = Path.Combine(outputPath, "training_result.json");
                if (!File.Exists(trainingContractPath))
                    throw new InvalidOperationException(
                        "Optuna finished without generating its training result contract.");

                var trainingContract = JsonConvert.DeserializeObject<TrainingResultOutput>(
                    File.ReadAllText(trainingContractPath));
                if (trainingContract == null
                    || trainingContract.SchemaVersion != 1
                    || trainingContract.Status != "success"
                    || trainingContract.JobId != jobId)
                    throw new InvalidOperationException(
                        "Optuna produced an invalid or stale result contract.");

                AppendTrainingStatus("Optuna training completed: " + outputPath);
                MessageBox.Show(
                    "Optuna training completed successfully.\n\nResults:\n" + outputPath,
                    "Training complete",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
            }
            catch (OperationCanceledException)
            {
                AppendTrainingStatus("Optuna training cancelled.");
            }
            catch (Exception ex)
            {
                AppendTrainingStatus("ERROR: " + ex.Message);
                MessageBox.Show(
                    ex.ToString(),
                    "Unexpected error",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
            finally
            {
                trainingButton.Enabled = true;
                cancelTrainingButton.Enabled = false;
                trainingProgressBar.Visible = false;
                trainingCancellationSource?.Dispose();
                trainingCancellationSource = null;
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

            string modelValidationError = ValidateTrainedModelFolder(
                trainedModelFolderTextBox.Text);
            if (!string.IsNullOrWhiteSpace(modelValidationError))
            {
                MessageBox.Show(
                    modelValidationError,
                    "Invalid trained model folder",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
                return;
            }

            try
            {
                predictionButton.Enabled = false;
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

                predictionCancellationSource = new CancellationTokenSource();
                CancellationToken cancellationToken = predictionCancellationSource.Token;
                cancelPredictionButton.Enabled = true;
                predictionProgressBar.Visible = true;
                predictionProgressBar.BringToFront();
                predictionProgressBar.Refresh();

                var selectedPredictionConfiguration = getCurveMapping();
                var pythonConfiguration = GetPythonConfiguration();

                AppendStatus("Selected wells: " + selectedWells.Count().ToString());

                string jobId;
                string workDir = CreateJobDirectory("prediction", out jobId);
                string inputPath = Path.Combine(workDir, "prediction_input.json");
                string analysisOutputPath = Path.Combine(workDir, "analysis_output.json");
                string predictionOutputPath = Path.Combine(workDir, "prediction_result.json");

                AppendStatus("Exporting selected wells to JSON...");

                var payload = BuildClusterAnalysisPayload(selectedWells, selectedPredictionConfiguration, pythonConfiguration);

                string json = JsonConvert.SerializeObject(payload, Formatting.Indented);

                File.WriteAllText(inputPath, json);

                AppendStatus("Data saved on: " + inputPath);

                AppendStatus("Python executable: " + pythonRuntimeService.PythonExecutable);
                AppendStatus("Runner script: " + pythonRuntimeService.PredictorScript);

                var result = await RunPythonWithProgressAsync(
                    () => pythonRuntimeService.AnalyzeAsync(
                        inputPath,
                        analysisOutputPath,
                        jobId,
                        cancellationToken),
                    analysisOutputPath,
                    jobId,
                    predictionProgressBar,
                    AppendStatus);

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

                if (!File.Exists(analysisOutputPath))
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

                string outputJson = File.ReadAllText(analysisOutputPath);

                var analysisOutput = JsonConvert.DeserializeObject<ClusterAnalysisOutput>(
                    outputJson
                );

                if (analysisOutput == null
                    || analysisOutput.SchemaVersion != 1
                    || analysisOutput.JobId != jobId)
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

                string clusterValidationError = ValidateEditedClusters(
                    clusters,
                    selectedWells,
                    selectedPredictionConfiguration);
                if (!string.IsNullOrWhiteSpace(clusterValidationError))
                {
                    MessageBox.Show(
                        clusterValidationError,
                        "Invalid clusters",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning);
                    return;
                }

                var payload_clusters = new
                {
                    clusters
                };

                string clustersPath = Path.Combine(workDir, "clusters.json");

                File.WriteAllText(
                    clustersPath,
                    JsonConvert.SerializeObject(payload_clusters, Formatting.Indented)
                );

                    var result_clusters = await RunPythonWithProgressAsync(
                        () => pythonRuntimeService.PredictAsync(
                            inputPath,
                            predictionOutputPath,
                            clustersPath,
                            trainedModelFolderTextBox.Text,
                            jobId,
                            cancellationToken,
                            (line, isError) => AppendStatus(line)),
                        predictionOutputPath,
                        jobId,
                        predictionProgressBar,
                        AppendStatus);

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
                    MessageBox.Show(
                        result_clusters.Stderr,
                        "Prediction failed",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    return;
                }

                if (!File.Exists(predictionOutputPath))
                {
                    MessageBox.Show(
                        "Python finished but did not generate the prediction result contract.",
                        "Missing output",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );

                    return;
                }

                var predictionOutput = JsonConvert.DeserializeObject<PredictionResultOutput>(
                    File.ReadAllText(predictionOutputPath));

                if (predictionOutput == null
                    || predictionOutput.SchemaVersion != 1
                    || predictionOutput.Status != "success"
                    || predictionOutput.JobId != jobId)
                {
                    throw new InvalidOperationException(
                        "Prediction did not finish successfully.");
                }

                AppendStatus(
                    "Prediction completed for "
                    + predictionOutput.WellCount
                    + " wells. Results: "
                    + predictionOutput.PredictionsFile);

                using (var resultsForm = new PredictionResultsForm(predictionOutputPath))
                {
                    resultsForm.ShowDialog(this);
                }
            }
            catch (OperationCanceledException)
            {
                AppendStatus("Prediction cancelled.");
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
                predictionButton.Enabled = true;
                cancelPredictionButton.Enabled = false;
                predictionProgressBar.Visible = false;
                predictionCancellationSource?.Dispose();
                predictionCancellationSource = null;
            }
        }

        private async Task<PythonProcessResult> RunPythonWithProgressAsync(
            Func<Task<PythonProcessResult>> operation,
            string outputPath,
            string jobId,
            ProgressBar progressBar,
            Action<string> appendStatus)
        {
            string progressPath = pythonRuntimeService.GetProgressPath(outputPath);
            if (progressBar != null)
            {
                progressBar.Value = 0;
            }
            Task<PythonProcessResult> processTask = operation();
            Task monitorTask = MonitorProgressAsync(
                progressPath,
                jobId,
                progressBar,
                appendStatus,
                processTask);

            PythonProcessResult result;
            try
            {
                result = await processTask;
            }
            finally
            {
                await monitorTask;
            }

            PythonExecutionContract contract = result.ExecutionContract;
            if (contract == null)
                throw new InvalidOperationException(
                    "Python finished without generating execution_contract.json.");

            if (contract.SchemaVersion != 1 || contract.JobId != jobId)
                throw new InvalidOperationException(
                    "Python returned an invalid or stale execution contract.");

            if (contract.Status == "cancelled")
                throw new OperationCanceledException(contract.Message);

            if (contract.Status != "success")
                throw new InvalidOperationException(
                    "Python execution failed: " + contract.Message);

            return result;
        }

        private async Task MonitorProgressAsync(
            string progressPath,
            string jobId,
            ProgressBar progressBar,
            Action<string> appendStatus,
            Task processTask)
        {
            string lastStage = null;
            string lastMessage = null;

            while (!processTask.IsCompleted)
            {
                UpdateProgressFromContract(
                    progressPath,
                    jobId,
                    progressBar,
                    appendStatus,
                    ref lastStage,
                    ref lastMessage);
                await Task.Delay(250);
            }

            UpdateProgressFromContract(
                progressPath,
                jobId,
                progressBar,
                appendStatus,
                ref lastStage,
                ref lastMessage);
        }

        private void UpdateProgressFromContract(
            string progressPath,
            string jobId,
            ProgressBar progressBar,
            Action<string> appendStatus,
            ref string lastStage,
            ref string lastMessage)
        {
            if (!File.Exists(progressPath))
                return;

            PythonProgressContract progress;
            try
            {
                progress = JsonConvert.DeserializeObject<PythonProgressContract>(
                    File.ReadAllText(progressPath));
            }
            catch (IOException)
            {
                return;
            }
            catch (JsonException)
            {
                return;
            }

            if (progress == null
                || progress.SchemaVersion != 1
                || progress.JobId != jobId)
                return;

            int percent = Math.Max(0, Math.Min(100, (int)Math.Round(progress.Percent)));
            if (progressBar != null)
            {
                if (progressBar.InvokeRequired)
                {
                    progressBar.BeginInvoke(new Action<int>(
                        value => progressBar.Value = value), percent);
                }
                else
                {
                    progressBar.Value = percent;
                }
            }

            if (appendStatus != null
                && (progress.Stage != lastStage || progress.Message != lastMessage))
            {
                appendStatus(
                    "Progress " + percent + "% - "
                    + progress.Stage + ": " + progress.Message);
                lastStage = progress.Stage;
                lastMessage = progress.Message;
            }
        }

        private void AppendStatus(string message)
        {
            if (statusTextBox == null)
            {
                return;
            }

            if (statusTextBox.InvokeRequired)
            {
                statusTextBox.BeginInvoke(
                    new Action<string>(AppendStatus),
                    message);
                return;
            }

            statusTextBox.AppendText(
                Environment.NewLine + "[" + DateTime.Now.ToString("HH:mm:ss") + "] " + message);
        }

        private string ValidateTrainedModelFolder(string folderPath)
        {
            if (string.IsNullOrWhiteSpace(folderPath))
                return "Select a trained experiment folder.";

            string experimentDirectory;
            try
            {
                experimentDirectory = Path.GetFullPath(folderPath);
            }
            catch (ArgumentException ex)
            {
                return "The selected model path is invalid: " + ex.Message;
            }

            if (!Directory.Exists(experimentDirectory))
                return "The selected model folder does not exist.";

            string bestConfigPath = Path.Combine(
                experimentDirectory,
                "best_config.json");
            string finalModelDirectory = Path.Combine(
                experimentDirectory,
                "final_model");

            if (!File.Exists(bestConfigPath))
            {
                if (string.Equals(
                    Path.GetFileName(experimentDirectory),
                    "final_model",
                    StringComparison.OrdinalIgnoreCase))
                {
                    return "Select the experiment folder that contains this final_model folder, "
                        + "not the final_model folder itself.";
                }

                return "The selected folder does not contain best_config.json.";
            }

            if (!Directory.Exists(finalModelDirectory))
                return "The selected folder does not contain a final_model subfolder.";

            JObject bestConfig;
            try
            {
                bestConfig = JObject.Parse(File.ReadAllText(bestConfigPath));
            }
            catch (JsonException ex)
            {
                return "best_config.json could not be read: " + ex.Message;
            }
            catch (IOException ex)
            {
                return "best_config.json could not be read: " + ex.Message;
            }

            var clusterProperties = bestConfig.Properties().ToList();
            if (clusterProperties.Count == 0)
                return "best_config.json does not contain any cluster configuration.";

            var invalidClusters = clusterProperties
                .Where(property =>
                    property.Value.Type != JTokenType.Object
                    || property.Value["features"] == null
                    || property.Value["hyperparams"] == null)
                .Select(property => property.Name)
                .ToList();

            if (invalidClusters.Count > 0)
            {
                return "best_config.json has invalid cluster configuration for: "
                    + string.Join(", ", invalidClusters)
                    + ". Each cluster must contain features and hyperparams.";
            }

            var missingArtifacts = new List<string>();
            foreach (var property in clusterProperties)
            {
                string modelPath = Path.Combine(
                    finalModelDirectory,
                    "final_model_cluster_" + property.Name + ".pth");
                string scalerPath = Path.Combine(
                    finalModelDirectory,
                    "final_scaler_cluster_" + property.Name + ".pkl");

                if (!File.Exists(modelPath))
                    missingArtifacts.Add(Path.GetFileName(modelPath));
                if (!File.Exists(scalerPath))
                    missingArtifacts.Add(Path.GetFileName(scalerPath));
            }

            if (missingArtifacts.Count > 0)
            {
                return "The selected model is incomplete. Missing artifacts: "
                    + string.Join(", ", missingArtifacts);
            }

            return null;
        }

        private void AppendTrainingStatus(string message)
        {
            if (trainingStatusTextBox == null)
            {
                return;
            }

            if (trainingStatusTextBox.InvokeRequired)
            {
                trainingStatusTextBox.BeginInvoke(
                    new Action<string>(AppendTrainingStatus),
                    message);
                return;
            }

            trainingStatusTextBox.AppendText(
                Environment.NewLine
                + "[" + DateTime.Now.ToString("HH:mm:ss") + "] "
                + message);
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
            var units = GetUnitConfiguration();
            var wells = new List<object>();

            foreach (var borehole in selectedWells)
            {
                wells.Add(new
                {
                    name = borehole.Name,
                    logs = ExtractWellLogSamples(borehole, curveMapping, units)
                });
            }

            return new
            {
                curveMapping,
                units,
                wells,
                pythonConfiguration
            };
        }

        private string ValidateEditedClusters(
            Dictionary<string, List<int>> clusters,
            List<Borehole> selectedWells,
            Dictionary<string, string> curveMapping)
        {
            if (clusters == null || clusters.Count == 0)
                return "At least one cluster must be provided.";

            var invalidNames = clusters.Keys
                .Where(name => string.IsNullOrWhiteSpace(name))
                .ToList();
            if (invalidNames.Count > 0)
                return "Every cluster must have a non-empty name.";

            var duplicateNames = clusters.Keys
                .GroupBy(name => name.Trim(), StringComparer.OrdinalIgnoreCase)
                .Where(group => group.Count() > 1)
                .Select(group => group.Key)
                .ToList();
            if (duplicateNames.Count > 0)
            {
                return "Cluster names must be unique. Duplicated names: "
                    + string.Join(", ", duplicateNames) + ".";
            }

            string vsLogName;
            if (curveMapping == null
                || !curveMapping.TryGetValue("VS", out vsLogName)
                || string.IsNullOrWhiteSpace(vsLogName))
            {
                return "The VS log must be configured before validating clusters.";
            }

            int labeledWellCount = selectedWells.Count(well =>
                well != null
                && well.Logs != null
                && well.Logs.WellLogs.Any(log =>
                    log != null
                    && string.Equals(
                        log.Name,
                        vsLogName,
                        StringComparison.OrdinalIgnoreCase)));

            if (labeledWellCount == 0)
                return "At least one selected well must contain the configured VS log.";

            var assignedWells = new HashSet<int>();
            foreach (var cluster in clusters)
            {
                if (cluster.Value == null || cluster.Value.Count == 0)
                {
                    return "Cluster '" + cluster.Key + "' must contain at least one well.";
                }

                foreach (int wellIndex in cluster.Value)
                {
                    if (wellIndex < 0 || wellIndex >= labeledWellCount)
                    {
                        return "Cluster '" + cluster.Key + "' contains well index "
                            + wellIndex + ", but valid indices range from 0 to "
                            + (labeledWellCount - 1) + " for wells containing VS.";
                    }

                    if (!assignedWells.Add(wellIndex))
                    {
                        return "Well index " + wellIndex
                            + " is assigned to more than one cluster.";
                    }
                }
            }

            return null;
        }


        private List<Object> ExtractWellLogSamples(
            Borehole borehole,
            Dictionary<String, String> curveMapping,
            UnitConfiguration units)
        {
            var logs = new List<Object>();

            foreach (var log in borehole.Logs.WellLogs)
            {
                string canonicalName = curveMapping
                    .Where(pair => string.Equals(pair.Value, log.Name, StringComparison.OrdinalIgnoreCase))
                    .Select(pair => pair.Key)
                    .FirstOrDefault();
                double conversionFactor = GetConversionFactor(canonicalName, units);
                var samples = log.Samples.Select(s => new {
                    md = s.MD,
                    value = s.Value * conversionFactor
                }).ToList();

                logs.Add(new
                {
                    name = log.Name,
                    samples
                });
            }

            return logs;
        }

        private UnitConfiguration GetUnitConfiguration()
        {
            return new UnitConfiguration
            {
                VP = GetSelectedUnit(vpUnitComboBox, "VP"),
                VS = GetSelectedUnit(vsUnitComboBox, "VS"),
                PythonVelocityUnit = "km/s",
                PayloadVelocityUnit = "km/s"
            };
        }

        private string GetSelectedUnit(ComboBox comboBox, string label)
        {
            if (comboBox == null || comboBox.SelectedItem == null)
                throw new InvalidOperationException(label + " unit must be selected.");

            return comboBox.SelectedItem.ToString();
        }

        private double GetConversionFactor(string canonicalName, UnitConfiguration units)
        {
            if (!string.Equals(canonicalName, "VP", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(canonicalName, "VS", StringComparison.OrdinalIgnoreCase))
                return 1.0;

            string sourceUnit = string.Equals(canonicalName, "VP", StringComparison.OrdinalIgnoreCase)
                ? units.VP
                : units.VS;

            if (string.Equals(sourceUnit, units.PythonVelocityUnit, StringComparison.OrdinalIgnoreCase))
                return 1.0;
            if (string.Equals(sourceUnit, "m/s", StringComparison.OrdinalIgnoreCase)
                && string.Equals(units.PythonVelocityUnit, "km/s", StringComparison.OrdinalIgnoreCase))
                return 0.001;
            if (string.Equals(sourceUnit, "km/s", StringComparison.OrdinalIgnoreCase)
                && string.Equals(units.PythonVelocityUnit, "m/s", StringComparison.OrdinalIgnoreCase))
                return 1000.0;

            throw new InvalidOperationException(
                "Unsupported velocity conversion from " + sourceUnit
                + " to " + units.PythonVelocityUnit + ".");
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

        private string CreateJobDirectory(string operation, out string jobId)
        {
            jobId = operation + "_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss")
                + "_" + Guid.NewGuid().ToString("N");
            string directory = Path.Combine(Path.GetTempPath(), "vs_predictior_petrel", jobId);
            Directory.CreateDirectory(directory);
            return directory;
        }

    }
}