using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace missing_data
{
    public class PythonProcessResult
    {
        public int ExitCode { get; set; }
        public string Stdout { get; set; }
        public string Stderr { get; set; }
        public PythonExecutionContract ExecutionContract { get; set; }
    }

    public class PythonExecutionContract
    {
        [JsonProperty("schema_version")]
        public int SchemaVersion { get; set; }
        [JsonProperty("job_id")]
        public string JobId { get; set; }
        [JsonProperty("operation")]
        public string Operation { get; set; }
        [JsonProperty("status")]
        public string Status { get; set; }
        [JsonProperty("message")]
        public string Message { get; set; }
        [JsonProperty("error_type")]
        public string ErrorType { get; set; }
    }

    public class PythonProgressContract
    {
        [JsonProperty("schema_version")]
        public int SchemaVersion { get; set; }
        [JsonProperty("job_id")]
        public string JobId { get; set; }
        [JsonProperty("operation")]
        public string Operation { get; set; }
        [JsonProperty("status")]
        public string Status { get; set; }
        [JsonProperty("percent")]
        public double Percent { get; set; }
        [JsonProperty("stage")]
        public string Stage { get; set; }
        [JsonProperty("message")]
        public string Message { get; set; }
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

        public string GetExecutionContractPath(string outputPath)
        {
            return GetContractPath(outputPath);
        }

        public string GetProgressPath(string outputPath)
        {
            return GetProgressContractPath(outputPath);
        }

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
            Directory.CreateDirectory(outputPath);

            string contractPath = Path.Combine(outputPath, "execution_contract.json");
            string progressPath = Path.Combine(outputPath, "progress.json");
            try
            {
                EnsureFileExists(pythonExecutable, "Python executable");
                EnsureFileExists(optunaScript, "Optuna runner");
            }
            catch (Exception error)
            {
                WriteExecutionContract(
                    contractPath,
                    jobId,
                    "optuna_training",
                    "error",
                    error.Message,
                    error.GetType().Name);
                return Task.FromResult(
                    CreateContractErrorResult(contractPath, error.Message));
            }

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
            string contractPath = GetContractPath(outputPath);
            string progressPath = GetProgressContractPath(outputPath);
            try
            {
                EnsureFileExists(pythonExecutable, "Python executable");
                EnsureFileExists(predictorScript, "Python runner");
            }
            catch (Exception error)
            {
                WriteExecutionContract(
                    contractPath,
                    jobId,
                    mode,
                    "error",
                    error.Message,
                    error.GetType().Name);
                return Task.FromResult(
                    CreateContractErrorResult(contractPath, error.Message));
            }

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
            WriteExecutionContract(
                contractPath,
                jobId,
                operation,
                "running",
                "Starting Python process.");

            using (var process = new Process())
            {
                process.StartInfo = startInfo;
                try
                {
                    process.Start();
                }
                catch (Exception error)
                {
                    WriteExecutionContract(
                        contractPath,
                        jobId,
                        operation,
                        "error",
                        error.Message,
                        error.GetType().Name);
                    return new PythonProcessResult
                    {
                        ExitCode = -1,
                        Stdout = string.Empty,
                        Stderr = error.Message,
                        ExecutionContract = ReadExecutionContract(contractPath)
                    };
                }

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

                PythonExecutionContract executionContract =
                    ReadExecutionContract(contractPath);
                if (process.ExitCode != 0
                    && (executionContract == null
                        || executionContract.Status == "running"))
                {
                    WriteExecutionContract(
                        contractPath,
                        jobId,
                        operation,
                        "error",
                        string.IsNullOrWhiteSpace(stderr.ToString())
                            ? "Python process exited with code " + process.ExitCode + "."
                            : stderr.ToString().Trim(),
                        "ProcessExit");
                    executionContract = ReadExecutionContract(contractPath);
                }

                return new PythonProcessResult
                {
                    ExitCode = process.ExitCode,
                    Stdout = stdout.ToString(),
                    Stderr = stderr.ToString(),
                    ExecutionContract = executionContract
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

        private static string GetProgressContractPath(string outputPath)
        {
            return Path.Combine(
                Path.GetDirectoryName(Path.GetFullPath(outputPath)),
                "progress.json");
        }

        private static PythonExecutionContract ReadExecutionContract(string contractPath)
        {
            if (!File.Exists(contractPath))
                return null;

            return JsonConvert.DeserializeObject<PythonExecutionContract>(
                File.ReadAllText(contractPath));
        }

        private static PythonProcessResult CreateContractErrorResult(
            string contractPath,
            string message)
        {
            return new PythonProcessResult
            {
                ExitCode = -1,
                Stdout = string.Empty,
                Stderr = message,
                ExecutionContract = ReadExecutionContract(contractPath)
            };
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
            WriteExecutionContract(
                contractPath,
                jobId,
                operation,
                "cancelled",
                "Execution cancelled by the user.");
        }

        private static void WriteExecutionContract(
            string contractPath,
            string jobId,
            string operation,
            string status,
            string message,
            string errorType = null)
        {
            string directory = Path.GetDirectoryName(contractPath);
            if (!string.IsNullOrWhiteSpace(directory))
                Directory.CreateDirectory(directory);

            var payload = new
            {
                schema_version = 1,
                job_id = jobId,
                operation = operation,
                status = status,
                message = message,
                updated_at = DateTime.UtcNow.ToString("o"),
                error_type = errorType
            };

            string temporaryPath = contractPath + "." + Guid.NewGuid().ToString("N") + ".tmp";
            File.WriteAllText(
                temporaryPath,
                JsonConvert.SerializeObject(
                    payload,
                    Formatting.Indented));
            try
            {
                if (File.Exists(contractPath))
                {
                    File.Replace(temporaryPath, contractPath, null);
                }
                else
                {
                    File.Move(temporaryPath, contractPath);
                }
            }
            finally
            {
                if (File.Exists(temporaryPath))
                    File.Delete(temporaryPath);
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
