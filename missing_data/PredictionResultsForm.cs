using Newtonsoft.Json.Linq;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace missing_data
{
    public class PredictionResultsForm : Form
    {
        private readonly JObject result;
        private DataGridView wellsGrid;

        public PredictionResultsForm(string resultPath)
        {
            if (!File.Exists(resultPath))
                throw new FileNotFoundException("Prediction result contract not found.", resultPath);

            result = JObject.Parse(File.ReadAllText(resultPath));
            BuildLayout();
        }

        private void BuildLayout()
        {
            Text = "Prediction results";
            Size = new Size(1100, 650);
            MinimumSize = new Size(850, 500);
            StartPosition = FormStartPosition.CenterParent;

            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 2,
                Padding = new Padding(10)
            };
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));

            root.Controls.Add(BuildResultsTabs(), 0, 0);
            root.Controls.Add(BuildActionPanel(), 0, 1);
            Controls.Add(root);
        }

        private Control BuildResultsTabs()
        {
            var tabs = new TabControl
            {
                Dock = DockStyle.Fill
            };

            var resultsPage = new TabPage("Results");
            var resultsLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 2,
                Padding = new Padding(4)
            };
            resultsLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 90));
            resultsLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            resultsLayout.Controls.Add(BuildSummaryPanel(), 0, 0);
            resultsLayout.Controls.Add(BuildWellsGrid(), 0, 1);
            resultsPage.Controls.Add(resultsLayout);
            tabs.TabPages.Add(resultsPage);

            AddPlotTabs(tabs);
            return tabs;
        }

        private void AddPlotTabs(TabControl tabs)
        {
            string plotsDirectory = (string)result["plots_dir"];
            if (string.IsNullOrWhiteSpace(plotsDirectory)
                || !Directory.Exists(plotsDirectory))
            {
                return;
            }

            foreach (string plotPath in Directory
                .GetFiles(plotsDirectory, "*.png")
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            {
                var page = new TabPage(Path.GetFileNameWithoutExtension(plotPath));
                var pictureBox = new PictureBox
                {
                    Dock = DockStyle.Fill,
                    SizeMode = PictureBoxSizeMode.Zoom,
                    Image = LoadImageWithoutLocking(plotPath)
                };
                page.Controls.Add(pictureBox);
                tabs.TabPages.Add(page);
            }
        }

        private Image LoadImageWithoutLocking(string path)
        {
            using (var stream = new MemoryStream(File.ReadAllBytes(path)))
            using (var image = Image.FromStream(stream))
            {
                return new Bitmap(image);
            }
        }

        private Control BuildSummaryPanel()
        {
            var summary = new TextBox
            {
                Dock = DockStyle.Fill,
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                Font = new Font("Consolas", 9),
                Text = BuildSummaryText()
            };
            return summary;
        }

        private string BuildSummaryText()
        {
            var metrics = result["metrics"] as JObject;
            string text =
                "Status: " + (string)result["status"] + Environment.NewLine
                + "Wells: " + (int?)result["well_count"] + " | "
                + "With VS: " + (int?)result["wells_with_ground_truth"] + " | "
                + "Without VS: " + (int?)result["wells_without_ground_truth"];

            if (metrics != null)
            {
                foreach (var property in metrics.Properties())
                {
                    text += Environment.NewLine
                        + property.Name + ": mean="
                        + property.Value["mean"]
                        + ", std="
                        + property.Value["std"];
                }
            }

            return text;
        }

        private Control BuildWellsGrid()
        {
            wellsGrid = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AutoGenerateColumns = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect
            };

            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "Well",
                DataPropertyName = "id",
                FillWeight = 180
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "Cluster",
                DataPropertyName = "cluster",
                FillWeight = 120
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "Ground truth",
                DataPropertyName = "ground_truth",
                FillWeight = 100
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "Predictions",
                DataPropertyName = "prediction_count",
                FillWeight = 100
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "R²",
                DataPropertyName = "r2",
                FillWeight = 80
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "RMSE",
                DataPropertyName = "rmse",
                FillWeight = 80
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "MAE",
                DataPropertyName = "mae",
                FillWeight = 80
            });

            var wells = result["wells"] as JArray;
            if (wells != null)
            {
                foreach (JObject well in wells)
                {
                    var metrics = well["metrics"] as JObject;
                    wellsGrid.Rows.Add(
                        (string)well["id"],
                        (string)well["cluster"],
                        (bool?)well["has_ground_truth"] == true ? "Yes" : "No",
                        (int?)well["prediction_count"],
                        FormatMetric(metrics, "R2"),
                        FormatMetric(metrics, "RMSE"),
                        FormatMetric(metrics, "MAE"));
                }
            }

            return wellsGrid;
        }

        private string FormatMetric(JObject metrics, string name)
        {
            if (metrics == null || metrics[name] == null || metrics[name].Type == JTokenType.Null)
                return "N/A";

            double value;
            if (double.TryParse(metrics[name].ToString(), out value))
                return value.ToString("0.####");

            return "N/A";
        }

        private Control BuildActionPanel()
        {
            var panel = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.RightToLeft
            };

            var closeButton = new Button
            {
                Text = "Close",
                Width = 100,
                DialogResult = DialogResult.OK
            };
            closeButton.Click += (sender, args) => Close();

            var plotsButton = new Button
            {
                Text = "Open plots",
                Width = 110
            };
            plotsButton.Click += (sender, args) => OpenPath((string)result["plots_dir"]);

            var reportButton = new Button
            {
                Text = "Open report",
                Width = 110
            };
            reportButton.Click += (sender, args) => OpenPath((string)result["summary_report_file"]);

            panel.Controls.Add(closeButton);
            panel.Controls.Add(plotsButton);
            panel.Controls.Add(reportButton);
            return panel;
        }

        private void OpenPath(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path) && !Directory.Exists(path))
            {
                MessageBox.Show(
                    "The selected result path does not exist:\n" + path,
                    "Missing result",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
                return;
            }

            Process.Start(new ProcessStartInfo
            {
                FileName = path,
                UseShellExecute = true
            });
        }
    }
}
