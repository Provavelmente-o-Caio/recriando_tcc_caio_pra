using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace missing_data
{
    public class PredictionResultsForm : Form
    {
        private readonly JObject result;
        private DataGridView wellsGrid;
        private readonly Dictionary<string, JObject> predictionData =
            new Dictionary<string, JObject>(StringComparer.OrdinalIgnoreCase);

        public PredictionResultsForm(string resultPath)
        {
            if (!File.Exists(resultPath))
                throw new FileNotFoundException("Prediction result contract not found.", resultPath);

            result = JObject.Parse(File.ReadAllText(resultPath));
            LoadPredictionData();
            BuildLayout();
        }

        private void LoadPredictionData()
        {
            string predictionsPath = (string)result["predictions_file"];
            if (string.IsNullOrWhiteSpace(predictionsPath) || !File.Exists(predictionsPath))
                return;

            var predictions = JObject.Parse(File.ReadAllText(predictionsPath));
            foreach (var property in predictions.Properties())
            {
                var well = property.Value as JObject;
                if (well != null)
                    predictionData[property.Name] = well;
            }
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
                ReadOnly = false,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AutoGenerateColumns = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect
            };

            wellsGrid.Columns.Add(new DataGridViewCheckBoxColumn
            {
                HeaderText = "Export",
                Name = "export",
                FillWeight = 60,
                ReadOnly = false
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "Well",
                Name = "id",
                ReadOnly = true,
                DataPropertyName = "id",
                FillWeight = 180
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "Cluster",
                Name = "cluster",
                ReadOnly = true,
                DataPropertyName = "cluster",
                FillWeight = 120
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "Ground truth",
                Name = "ground_truth",
                ReadOnly = true,
                DataPropertyName = "ground_truth",
                FillWeight = 100
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "Predictions",
                Name = "prediction_count",
                ReadOnly = true,
                DataPropertyName = "prediction_count",
                FillWeight = 100
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "R²",
                Name = "r2",
                ReadOnly = true,
                DataPropertyName = "r2",
                FillWeight = 80
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "RMSE",
                Name = "rmse",
                ReadOnly = true,
                DataPropertyName = "rmse",
                FillWeight = 80
            });
            wellsGrid.Columns.Add(new DataGridViewTextBoxColumn
            {
                HeaderText = "MAE",
                Name = "mae",
                ReadOnly = true,
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
                        false,
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

            var exportButton = new Button
            {
                Text = "Export selected wells",
                Width = 150
            };
            exportButton.Click += (sender, args) => ExportSelectedWells();
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
            panel.Controls.Add(exportButton);
            panel.Controls.Add(plotsButton);
            panel.Controls.Add(reportButton);
            return panel;
        }

        private void ExportSelectedWells()
        {
            var selectedWellIds = wellsGrid.Rows
                .Cast<DataGridViewRow>()
                .Where(row => row.Cells["export"].Value is bool
                    && (bool)row.Cells["export"].Value)
                .Select(row => Convert.ToString(row.Cells["id"].Value))
                .Where(id => !string.IsNullOrWhiteSpace(id))
                .ToList();

            if (selectedWellIds.Count == 0)
            {
                MessageBox.Show(
                    "Select at least one well in the Export column.",
                    "No wells selected",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
                return;
            }

            string predictionsPath = (string)result["predictions_file"];
            if (predictionData.Count == 0)
            {
                MessageBox.Show(
                    "The prediction data file could not be found:\n" + predictionsPath,
                    "Missing prediction data",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
                return;
            }

            using (var dialog = new FolderBrowserDialog
            {
                Description = "Choose a folder for the exported LAS wells."
            })
            {
                if (dialog.ShowDialog(this) != DialogResult.OK)
                    return;

                var exported = new List<string>();
                var missing = new List<string>();
                foreach (string wellId in selectedWellIds)
                {
                    JObject data;
                    if (!predictionData.TryGetValue(wellId, out data))
                    {
                        missing.Add(wellId);
                        continue;
                    }

                    string fileName = SanitizeFileName(wellId) + ".las";
                    string outputPath = Path.Combine(dialog.SelectedPath, fileName);
                    File.WriteAllText(outputPath, BuildLasContent(wellId, data));
                    exported.Add(fileName);
                }

                string message = "Exported " + exported.Count + " LAS file(s) to:\n"
                    + dialog.SelectedPath;
                if (missing.Count > 0)
                    message += "\n\nMissing prediction data for: "
                        + string.Join(", ", missing);

                MessageBox.Show(
                    message,
                    exported.Count > 0 ? "Export complete" : "Export failed",
                    MessageBoxButtons.OK,
                    exported.Count > 0
                        ? MessageBoxIcon.Information
                        : MessageBoxIcon.Warning);
            }
        }

        private string BuildLasContent(string wellId, JObject data)
        {
            var originalData = data["original_data"] as JArray;
            var depths = data["depths"] as JArray;
            var predictions = data["predictions"] as JArray;
            if (originalData == null || originalData.Count == 0)
                throw new InvalidDataException(
                    "Original well data is not available for " + wellId);

            string outputCurve = "VS_PREDICTED_ML";
            string outputUnit = "m/s";
            var units = result["units"] as JObject;
            if (units != null && units["VS"] != null)
                outputUnit = units["VS"].ToString();

            double outputFactor = string.Equals(outputUnit, "m/s",
                StringComparison.OrdinalIgnoreCase) ? 1000.0 : 1.0;

            var predictionByDepth = new Dictionary<double, double>();
            if (depths != null && predictions != null)
            {
                int predictionCount = Math.Min(depths.Count, predictions.Count);
                for (int index = 0; index < predictionCount; index++)
                {
                    predictionByDepth[depths[index].Value<double>()] =
                        predictions[index].Value<double>() * outputFactor;
                }
            }

            var originalColumns = originalData
                .OfType<JObject>()
                .SelectMany(row => row.Properties().Select(property => property.Name))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Where(column => !string.Equals(column, "DEPT",
                    StringComparison.OrdinalIgnoreCase))
                .ToList();
            var rows = new List<string>();
            foreach (JObject originalRow in originalData.OfType<JObject>())
            {
                JToken depthToken = originalRow["DEPT"];
                if (depthToken == null || depthToken.Type == JTokenType.Null)
                    continue;

                double depth = depthToken.Value<double>();
                var values = new List<string>
                {
                    FormatLasValue(depth)
                };
                foreach (string column in originalColumns)
                {
                    JToken value = originalRow[column];
                    values.Add(FormatOriginalValue(column, value));
                }

                double prediction;
                values.Add(predictionByDepth.TryGetValue(depth, out prediction)
                    ? FormatLasValue(prediction)
                    : "-999.25");
                rows.Add(string.Join(" ", values));
            }

            if (rows.Count == 0)
                throw new InvalidDataException("No depth samples for " + wellId);

            double start = originalData[0]["DEPT"].Value<double>();
            double stop = originalData[originalData.Count - 1]["DEPT"].Value<double>();
            var curveHeaders = new List<string>
            {
                " DEPT.M                 : DEPTH"
            };
            curveHeaders.AddRange(originalColumns.Select(column =>
                " " + column + ".               : ORIGINAL CURVE"));
            curveHeaders.Add(" " + outputCurve + "." + outputUnit
                + " : PREDICTED VS");

            return "~Version" + Environment.NewLine
                + " VERS.                 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0"
                + Environment.NewLine
                + " WRAP.                  NO  : ONE LINE PER DEPTH STEP"
                + Environment.NewLine
                + "~Well" + Environment.NewLine
                + " WELL.                  " + wellId + " : WELL NAME"
                + Environment.NewLine
                + "~Curve Information" + Environment.NewLine
                + string.Join(Environment.NewLine, curveHeaders)
                + Environment.NewLine
                + "~Parameter Information" + Environment.NewLine
                + " STRT.M                 " + start.ToString("0.######", CultureInfo.InvariantCulture)
                + " : START DEPTH"
                + Environment.NewLine
                + " STOP.M                 " + stop.ToString("0.######", CultureInfo.InvariantCulture)
                + " : STOP DEPTH"
                + Environment.NewLine
                + "~ASCII" + Environment.NewLine
                + string.Join(Environment.NewLine, rows)
                + Environment.NewLine;
        }

        private string FormatLasValue(JToken value)
        {
            if (value == null || value.Type == JTokenType.Null)
                return "-999.25";

            double number;
            if (double.TryParse(value.ToString(), NumberStyles.Float,
                CultureInfo.InvariantCulture, out number))
            {
                return number.ToString("0.######", CultureInfo.InvariantCulture);
            }

            return "-999.25";
        }

        private string FormatOriginalValue(string column, JToken value)
        {
            if (value == null || value.Type == JTokenType.Null)
                return "-999.25";

            double number;
            if (!double.TryParse(value.ToString(), NumberStyles.Float,
                CultureInfo.InvariantCulture, out number))
                return "-999.25";

            var units = result["units"] as JObject;
            string sourceUnit = units == null || units[column] == null
                ? null
                : units[column].ToString();
            if ((string.Equals(column, "VP", StringComparison.OrdinalIgnoreCase)
                || string.Equals(column, "VS", StringComparison.OrdinalIgnoreCase))
                && string.Equals(sourceUnit, "m/s", StringComparison.OrdinalIgnoreCase))
            {
                number *= 1000.0;
            }

            return number.ToString("0.######", CultureInfo.InvariantCulture);
        }

        private string SanitizeFileName(string value)
        {
            var invalidCharacters = Path.GetInvalidFileNameChars();
            return new string(value.Select(character =>
                invalidCharacters.Contains(character) ? '_' : character).ToArray());
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
