using Slb.Ocean.Petrel;
using Slb.Ocean.Petrel.DomainObject;
using Slb.Ocean.Petrel.DomainObject.Well;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace missing_data
{
    public partial class MainForm : Form
    {
        private CheckedListBox wellsListBox;
        private ComboBox vpComboBox;
        private ComboBox rhoComboBox;
        private ComboBox grComboBox;
        private ComboBox porosityComboBox;
        private ComboBox saturationComboBox;
        private ComboBox clayComboBox;
        private ComboBox caliperComboBox;
        private TextBox outputCurveNameTextBox;
        private TextBox statusTextBox;
        private Button runButton;
        private TabPage mainTab;
        private TabPage TrainingTab;
        private TabPage configurationTab;
        private Slb.Ocean.Petrel.UI.DropTarget wellDropTarget;

        public MainForm()
        {
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
            var rightPanel = BuildConfigurationPanel();

            mainSplit.Panel1.Controls.Add(leftPanel);
            mainSplit.Panel2.Controls.Add(rightPanel);

            return mainSplit;
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

            wellDropTarget= new Slb.Ocean.Petrel.UI.DropTarget
            {
                Dock = DockStyle.Fill,
                Text = "Drop wells here"
            };

            wellDropTarget.DragDrop += WellDropTarget_DragDrop;
            wellDropTarget.DragEnter += WellDropTarget_DragEnter;

            wellsListBox = new CheckedListBox
            {
                Dock = DockStyle.Fill,
                CheckOnClick = true
            };

            LoadWellsIntoCheckBox(wellsListBox);

            var buttons = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                Height = 40,
                FlowDirection = FlowDirection.LeftToRight
            };

            var selectAllButton = new Button { Text = "Select all", Width = 100 };
            var clearButton = new Button { Text = "Clear", Width = 100 };

            selectAllButton.Click += SelectAllButton_Click;
            clearButton.Click += ClearButton_Click;

            buttons.Controls.Add(selectAllButton);
            buttons.Controls.Add(clearButton);

            layout.Controls.Add(wellDropTarget, 0, 0);
            layout.Controls.Add(wellsListBox, 0, 1);
            layout.Controls.Add(buttons, 0, 2);

            group.Controls.Add(layout);

            return group;
        }

        /*
        private Control BuildWellSelectionPanel()
        {
            var group = new GroupBox
            {
                Text = "Wells",
                Dock = DockStyle.Fill,
                Padding = new Padding(10),
            };

            wellsListBox = new CheckedListBox
            {
                Dock = DockStyle.Fill,
                CheckOnClick = true
            };

            LoadWellsIntoCheckBox(wellsListBox);

            var buttons = new FlowLayoutPanel
            {
                Dock = DockStyle.Bottom,
                Height = 40,
                FlowDirection = FlowDirection.LeftToRight
            };

            var selectAllButton = new Button { Text = "Select all", Width = 100 };
            var clearButton = new Button { Text = "Clear", Width = 100 };

            selectAllButton.Click += SelectAllButton_Click;
            clearButton.Click += ClearButton_Click;

            buttons.Controls.Add(selectAllButton);
            buttons.Controls.Add(clearButton);

            group.Controls.Add(wellsListBox);
            group.Controls.Add(buttons);

            return group;
        }
        */

         private Control BuildConfigurationPanel()
         {
            var group = new GroupBox
            {
                Text = "Prediction configuration",
                Dock = DockStyle.Fill,
                Padding = new Padding(12)
            };

            var layout = new TableLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                ColumnCount = 2,
                RowCount = 9
            };

            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 140));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            vpComboBox = AddCurveRow(layout, "VP:", 0);
            rhoComboBox = AddCurveRow(layout, "RHO:", 1);
            grComboBox = AddCurveRow(layout, "GR:", 2);
            porosityComboBox = AddCurveRow(layout, "Porosity:", 3);
            saturationComboBox = AddCurveRow(layout, "Saturation:", 4);
            clayComboBox = AddCurveRow(layout, "Clay:", 5);
            caliperComboBox = AddCurveRow(layout, "Caliper:", 6);

            outputCurveNameTextBox = new TextBox
            {
                Dock = DockStyle.Fill,
                Text = "VS_PREDICTED_ML"
            };

            layout.Controls.Add(new Label { Text = "Output curve:", TextAlign = ContentAlignment.MiddleLeft }, 0, 7);
            layout.Controls.Add(outputCurveNameTextBox, 1, 7);

            runButton = new Button
            {
                Text = "Run prediction",
                Height = 36,
                Dock = DockStyle.Top
            };

            runButton.Click += RunButton_Click;

            group.Controls.Add(runButton);
            group.Controls.Add(layout);

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

            return combo;
        }


        private void LoadWellsIntoCheckBox(CheckedListBox clBox)
        {
            clBox.Items.Clear();

            Project project = PetrelProject.PrimaryProject ?? throw new InvalidOperationException("No active Petrel Project.");

            WellRoot wellroot = WellRoot.Get(project) ?? throw new InvalidOperationException("WellRoot not found");

            var boreholeCollections = wellroot.BoreholeCollection?.BoreholeCollections ?? throw new InvalidOperationException("No borehole collections found.");

            if (!boreholeCollections.Any()) {
                throw new InvalidOperationException("No boreholes availible in the project");
            }

            var nameOccurrences = new Dictionary<string, int>();
            var wellNames = boreholeCollections
                .SelectMany(c => c)
                .Select(w => w.Name)
                .Distinct()
                .OrderBy(n => n)
                .ToArray();

            clBox.Items.AddRange(wellNames);
        }

        private void RunButton_Click(object sender, EventArgs e)
        {
        }

        private void SelectAllButton_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < wellsListBox.Items.Count; i++)
            {
                wellsListBox.SetItemChecked(i, true);
            }
        }

        private void ClearButton_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < wellsListBox.Items.Count; i++)
            {
                wellsListBox.SetItemChecked(i, false);
            }
        }

        private void WellDropTarget_DragDrop(object sender, DragEventArgs e)
        {
            object droppedObject = e.Data.GetData(e.Data.GetFormats()[0]);

            if (droppedObject == null)
            {
                return;
            }

            
            PetrelLogger.InfoOutputWindow(string.Format(droppedObject.ToString()));
        }

        private void WellDropTarget_DragEnter(object sender, DragEventArgs e)
        {
            e.Effect = DragDropEffects.Copy;
        }
    }
}
