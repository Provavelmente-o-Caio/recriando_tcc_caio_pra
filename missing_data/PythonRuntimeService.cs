using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Text;
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

    public class PythonRuntimeService
    {
        private readonly string projectDirectory;
        private readonly string pythonExecutable;
        private readonly string predictorScript;
        private readonly string optunaScript;

        public PythonRuntimeService()
        {
            projectDirectory = ResolveProjectDirectory();
            pythonExecutable = Path.Combine(projectDirectory, ".venv", "Scripts", "python.exe");
            predictorScript = Path.Combine(projectDirectory, "predictor.py");
            optunaScript = Path.Combine(projectDirectory, "optuna_experiment.py");
        }

        public string PythonExecutable { get { return pythonExecutable; } }
        public string PredictorScript { get { return predictorScript; } }
        public string OptunaScript { get { return optunaScript; } }

        public Task<PythonProcessResult> AnalyzeAsync(
            string inputPath, string outputPath, string jobId,
            CancellationToken cancellationToken = default(CancellationToken),
            Action<string, bool> outputCallback = null)
        {
            return RunPredictorAsync("analyze", inputPath, outputPath, null, null,
                jobId, cancellationToken, outputCallback);
        }

        public Task<PythonProcessResult> TrainAsync(
            string inputPath, string outputPath, string clustersPath, string jobId,
            CancellationToken cancellationToken,
            Action<string, bool> outputCallback = null)
        {
            return RunPredictorAsync("train", inputPath, outputPath, clustersPath, null,
                jobId, cancellationToken, outputCallback);
        }

        public Task<PythonProcessResult> PredictAsync(
            string inputPath, string outputPath, string clustersPath,
            string trainedModelFolder, string jobId,
            CancellationToken cancellationToken = default(CancellationToken),
            Action<string, bool> outputCallback = null)
        {
            return RunPredictorAsync("predict", inputPath, outputPath, clustersPath,
                trainedModelFolder, jobId, cancellationToken, outputCallback);
        }

        public Task<PythonProcessResult> RunOptunaTrainingAsync(
            string inputPath, string clustersPath, string outputPath,
            int trials, int jobs, string jobId, CancellationToken cancellationToken,
            Action<string, bool> outputCallback = null)
        {
            EnsureFileExists(pythonExecutable, "Python executable");
            EnsureFileExists(optunaScript, "Optuna runner");
            Directory.CreateDirectory(outputPath);

            string contractPath = Path.Combine(outputPath, "execution_contract.json");
            string progressPath = Path.Combine(outputPath, "progress.json");
            string arguments =
                Quote(optunaScript) + " --data-source payload " +
                "--input " + Quote(inputPath) + " " +
                "--cluster " + Quote(clustersPath) + " " +
                "--output " + Quote(outputPath) + " " +
                "--trials " + trials + " --jobs " + jobs + " " +
                "--job-id " + Quote(jobId) + " " +
                "--contract " + Quote(contractPath) + " " +
                "--progress " + Quote(progressPath);

            return RunAsync(CreateStartInfo(pythonExecutable, arguments, optunaScript),
                cancellationToken, outputCallback, contractPath, jobId, "optuna_training");
        }

        private Task<PythonProcessResult> RunPredictorAsync(
            string mode, string inputPath, string outputPath, string clustersPath,
            string trainedModelFolder, string jobId, CancellationToken cancellationToken,
            Action<string, bool> outputCallback)
        {
            EnsureFileExists(pythonExecutable, "Python executable");
            EnsureFileExists(predictorScript, "Python runner");

            string contractPath = GetContractPath(outputPath);
            string progressPath = GetProgressPath(outputPath);
            string arguments =
                Quote(predictorScript) + " " + mode + " " +
                "--input " + Quote(inputPath) + " " +
                "--output " + Quote(outputPath) + " " +
                "--job-id " + Quote(jobId) + " " +
                "--contract " + Quote(contractPath) + " " +
                "--progress " + Quote(progressPath);

            if (clustersPath != null)
                arguments += " --cluster " + Quote(clustersPath);
            if (trainedModelFolder != null)
                arguments += " --experiment_dir " + Quote(trainedModelFolder);

            return RunAsync(CreateStartInfo(pythonExecutable, arguments, predictorScript),
                cancellationToken, outputCallback, contractPath, jobId, mode);
        }

        private static ProcessStartInfo CreateStartInfo(
            string executable, string arguments, string scriptPath)
        {
            return new ProcessStartInfo
            {
                FileName = executable,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = Path.GetDirectoryName(scriptPath)
            };
        }

        private static async Task<PythonProcessResult> RunAsync(
            ProcessStartInfo startInfo, CancellationToken cancellationToken,
            Action<string, bool> outputCallback, string contractPath,
            string jobId, string operation)
        {
            using (var process = new Process())
            {
                process.StartInfo = startInfo;
                process.Start();
                var stdout = new StringBuilder();
                var stderr = new StringBuilder();
                Task stdoutTask = ReadStreamAsync(process.StandardOutput, stdout, false, outputCallback);
                Task stderrTask = ReadStreamAsync(process.StandardError, stderr, true, outputCallback);
                Task waitForExitTask = Task.Run(() => process.WaitForExit());

                using (cancellationToken.Register(() => TerminateProcessTree(process)))
                    await Task.WhenAll(stdoutTask, stderrTask, waitForExitTask);

                if (cancellationToken.IsCancellationRequested)
                {
                    WriteCancellationContract(contractPath, jobId, operation);
                    cancellationToken.ThrowIfCancellationRequested();
                }

                return new PythonProcessResult
                {
                    ExitCode = process.ExitCode,
                    Stdout = stdout.ToString(),
                    Stderr = stderr.ToString()
                };
            }
        }

        private static async Task ReadStreamAsync(
            StreamReader reader, StringBuilder output, bool isError,
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

        private static string ResolveProjectDirectory()
        {
            string configuredDirectory =
                Environment.GetEnvironmentVariable("RECRIANDO_TCC_HOME");
            if (!string.IsNullOrWhiteSpace(configuredDirectory))
                return Path.GetFullPath(configuredDirectory);

            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "recriando_tcc_caio_pra");
        }

        private static string GetContractPath(string outputPath)
        {
            return Path.Combine(
                Path.GetDirectoryName(Path.GetFullPath(outputPath)),
                "execution_contract.json");
        }

        private static string GetProgressPath(string outputPath)
        {
            return Path.Combine(
                Path.GetDirectoryName(Path.GetFullPath(outputPath)),
                "progress.json");
        }

        private static string Quote(string value)
        {
            return "\"" + value.Replace("\"", "\\\"") + "\"";
        }

        private static void EnsureFileExists(string path, string description)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException(description + " not found.", path);
        }

        private static void WriteCancellationContract(
            string contractPath, string jobId, string operation)
        {
            string directory = Path.GetDirectoryName(contractPath);
            if (!string.IsNullOrWhiteSpace(directory))
                Directory.CreateDirectory(directory);

            File.WriteAllText(
                contractPath,
                "{\n"
                + "  \"schema_version\": 1,\n"
                + "  \"job_id\": \"" + jobId.Replace("\"", "\\\"") + "\",\n"
                + "  \"operation\": \"" + operation + "\",\n"
                + "  \"status\": \"cancelled\",\n"
                + "  \"message\": \"Execution cancelled by the user.\"\n"
                + "}");
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
                        Arguments = "/PID " + process.Id + " /T /F",
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
