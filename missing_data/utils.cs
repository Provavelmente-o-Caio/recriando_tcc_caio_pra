using Newtonsoft.Json;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Text;

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
        public static async Task<PythonProcessResult> RunPythonAnalysisAsync(
            string pythonExe,
            string runnerPath,
            string mode,
            string inputPath,
            string outputPath,
            CancellationToken cancellationToken = default(CancellationToken),
            Action<string, bool> outputCallback = null)
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

            return await RunProcessAsync(psi, cancellationToken, outputCallback);
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

            return await RunProcessAsync(psi, CancellationToken.None, null);
        }

        public static async Task<PythonProcessResult> RunPythonOptunaTrainingAsync(
            string pythonExe,
            string optunaScriptPath,
            string inputPath,
            string clustersPath,
            string outputPath,
            int trials,
            int jobs,
            CancellationToken cancellationToken,
            Action<string, bool> outputCallback = null)
        {
            if (!File.Exists(pythonExe))
                throw new FileNotFoundException("Python executable not found.", pythonExe);

            if (!File.Exists(optunaScriptPath))
                throw new FileNotFoundException("Optuna runner not found.", optunaScriptPath);

            Directory.CreateDirectory(outputPath);

            var arguments =
                $"\"{optunaScriptPath}\" " +
                $"--data-source payload " +
                $"--input \"{inputPath}\" " +
                $"--cluster \"{clustersPath}\" " +
                $"--output \"{outputPath}\" " +
                $"--trials {trials} " +
                $"--jobs {jobs}";

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = false
            };

            return await RunProcessAsync(psi, cancellationToken, outputCallback);
        }


        public static async Task<PythonProcessResult> RunPythonPredictionAsync(string pythonExe, string runnerPath, string inputPath, string outputPath, string clustersPath, string trainedModelFolder)
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
                $"--cluster \"{clustersPath}\" " +
                $"--experiment_dir \"{trainedModelFolder}\"";

            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = false
            };

            return await RunProcessAsync(psi, CancellationToken.None, null);
        }

        private static async Task<PythonProcessResult> RunProcessAsync(
            ProcessStartInfo startInfo,
            CancellationToken cancellationToken,
            Action<string, bool> outputCallback)
        {
            using (var process = new Process())
            {
                process.StartInfo = startInfo;
                process.Start();

                var stdout = new StringBuilder();
                var stderr = new StringBuilder();
                Task stdoutTask = ReadStreamAsync(
                    process.StandardOutput,
                    stdout,
                    false,
                    outputCallback);
                Task stderrTask = ReadStreamAsync(
                    process.StandardError,
                    stderr,
                    true,
                    outputCallback);
                Task waitForExitTask = Task.Run(() => process.WaitForExit());
                using (cancellationToken.Register(() => TerminateProcessTree(process)))
                {
                    await Task.WhenAll(stdoutTask, stderrTask, waitForExitTask);
                }

                cancellationToken.ThrowIfCancellationRequested();

                return new PythonProcessResult
                {
                    ExitCode = process.ExitCode,
                    Stdout = stdout.ToString(),
                    Stderr = stderr.ToString()
                };
            }
        }

        private static async Task ReadStreamAsync(
            StreamReader reader,
            StringBuilder output,
            bool isError,
            Action<string, bool> outputCallback)
        {
            string line;
            while ((line = await reader.ReadLineAsync()) != null)
            {
                output.AppendLine(line);
                if (outputCallback != null)
                    outputCallback(line, isError);
            }
        }

        private static void TerminateProcessTree(Process process)
        {
            if (process.HasExited)
                return;

            try
            {
                using (var terminator = new Process())
                {
                    terminator.StartInfo = new ProcessStartInfo
                    {
                        FileName = "taskkill.exe",
                        Arguments = $"/PID {process.Id} /T /F",
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                    terminator.Start();
                    terminator.WaitForExit();
                }
            }
            catch (InvalidOperationException)
            {
                if (!process.HasExited)
                    process.Kill();
            }
            catch (Win32Exception)
            {
                if (!process.HasExited)
                    process.Kill();
            }
        }
    }
}
