using Newtonsoft.Json;
using System.ComponentModel;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
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
        public static async Task<PythonProcessResult> RunPythonAnalysisAsync(
            string pythonExe,
            string runnerPath,
            string mode,
            string inputPath,
            string outputPath,
            CancellationToken cancellationToken = default(CancellationToken))
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

            return await RunProcessAsync(psi, cancellationToken);
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

            return await RunProcessAsync(psi, CancellationToken.None);
        }

        public static async Task<PythonProcessResult> RunPythonOptunaTrainingAsync(
            string pythonExe,
            string optunaScriptPath,
            string inputPath,
            string clustersPath,
            string outputPath,
            int trials,
            int jobs,
            CancellationToken cancellationToken)
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

            return await RunProcessAsync(psi, cancellationToken);
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

            return await RunProcessAsync(psi, CancellationToken.None);
        }

        private static async Task<PythonProcessResult> RunProcessAsync(
            ProcessStartInfo startInfo,
            CancellationToken cancellationToken)
        {
            using (var process = new Process())
            {
                process.StartInfo = startInfo;
                process.Start();

                Task<string> stdoutTask = process.StandardOutput.ReadToEndAsync();
                Task<string> stderrTask = process.StandardError.ReadToEndAsync();
                Task waitForExitTask = Task.Run(() => process.WaitForExit());
                using (cancellationToken.Register(() => TerminateProcessTree(process)))
                {
                    await Task.WhenAll(stdoutTask, stderrTask, waitForExitTask);
                }

                cancellationToken.ThrowIfCancellationRequested();

                return new PythonProcessResult
                {
                    ExitCode = process.ExitCode,
                    Stdout = stdoutTask.Result,
                    Stderr = stderrTask.Result
                };
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
