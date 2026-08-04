using Newtonsoft.Json;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace missing_data
{
    public class PythonProcessResult
    {
        public int ExitCode { get; set; }

        public string Stdout { get; set; }

        public string Stderr { get; set; }
    }

    public class Utils
    {
        public static async Task<PythonProcessResult> RunPythonAnalysisAsync(string pythonExe, string runnerPath, string mode, string inputPath, string outputPath)
        {
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
                    "Python runner not found.",
                    runnerPath
                );
            }

            string workingDirectory = Path.GetDirectoryName(runnerPath);

            var arguments =
                "\"" + runnerPath + "\" " +
                mode + " " +
                "--input \"" + inputPath + "\" " +
                "--output \"" + outputPath + "\"";

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using (var process = new Process())
            {
                process.StartInfo = psi;
                process.Start();

                string stdout = await process.StandardOutput.ReadToEndAsync();
                string stderr = await process.StandardError.ReadToEndAsync();

                await Task.Run(() => process.WaitForExit());

                return new PythonProcessResult
                {
                    ExitCode = process.ExitCode,
                    Stdout = stdout,
                    Stderr = stderr
                };
            }
        }
        public static async Task<PythonProcessResult> RunPythonTrainingAsync(string pythonExe, string runnerPath, string inputPath, string outputPath, string clustersPath)
        {
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
                    "Python runner not found.",
                    runnerPath
                );
            }

            var arguments =
                $"\"{runnerPath}\" train " +
                $"--input \"{inputPath}\" " +
                $"--output \"{outputPath}\" " +
                $"--cluster \"{clustersPath}\"";

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = false
            };

            using (var process = new Process())
            {
                process.StartInfo = psi;
                process.Start();

                string stdout = await process.StandardOutput.ReadToEndAsync();
                string stderr = await process.StandardError.ReadToEndAsync();

                await Task.Run(() => process.WaitForExit());

                return new PythonProcessResult
                {
                    ExitCode = process.ExitCode,
                    Stdout = stdout,
                    Stderr = stderr
                };
            }
        }


        public static async Task<PythonProcessResult> RunPythonPredictionAsync(string pythonExe, string runnerPath, string inputPath, string outputPath, string clustersPath)
        {
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
                    "Python runner not found.",
                    runnerPath
                );
            }

            var arguments =
                $"\"{runnerPath}\" predict " +
                $"--input \"{inputPath}\" " +
                $"--output \"{outputPath}\" " +
                $"--cluster \"{clustersPath}\"";

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = false
            };

            using (var process = new Process())
            {
                process.StartInfo = psi;
                process.Start();

                string stdout = await process.StandardOutput.ReadToEndAsync();
                string stderr = await process.StandardError.ReadToEndAsync();

                await Task.Run(() => process.WaitForExit());

                return new PythonProcessResult
                {
                    ExitCode = process.ExitCode,
                    Stdout = stdout,
                    Stderr = stderr
                };
            }
        }
    }
}
