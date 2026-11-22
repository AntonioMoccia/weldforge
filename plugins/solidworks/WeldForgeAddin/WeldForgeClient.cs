// HTTP client for the WeldForge API.
//
// Uses only System.Net.Http and System.Text.Json: no NuGet packages beyond
// what the .NET Framework ships. A SolidWorks add-in gets installed on
// engineering workstations that are often locked down, and every extra
// assembly is one more thing to get through IT.
//
// A single static HttpClient, because creating one per call exhausts sockets
// under repeated use -- a well-known .NET trap and an easy one to hit when a
// user presses "Detect seams" twenty times.

using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace WeldForge
{
    /// <summary>An API call failed; carries the server's own message.</summary>
    public class WeldForgeException : Exception
    {
        public int StatusCode { get; }

        public WeldForgeException(string message, int statusCode = 0) : base(message)
        {
            StatusCode = statusCode;
        }
    }

    public class WeldForgeClient
    {
        private static readonly HttpClient Http = new HttpClient
        {
            // Planning a large weldment takes minutes; the default 100 s is
            // not enough and times out halfway through a real job.
            Timeout = TimeSpan.FromMinutes(20)
        };

        public string BaseUrl { get; }

        public WeldForgeClient(string baseUrl = null)
        {
            BaseUrl = (baseUrl
                       ?? Environment.GetEnvironmentVariable("WELDFORGE_URL")
                       ?? "http://127.0.0.1:8000").TrimEnd('/');
        }

        // -- plumbing --------------------------------------------------------

        private async Task<JsonDocument> SendAsync(HttpRequestMessage request)
        {
            HttpResponseMessage response;
            try
            {
                response = await Http.SendAsync(request).ConfigureAwait(false);
            }
            catch (Exception exc)
            {
                throw new WeldForgeException(
                    $"cannot reach the WeldForge server at {BaseUrl}: {exc.Message}. " +
                    "Start it with: weldforge serve");
            }

            string body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
            {
                string detail = body;
                try
                {
                    using (var parsed = JsonDocument.Parse(body))
                    {
                        if (parsed.RootElement.TryGetProperty("detail", out var element))
                        {
                            detail = element.ToString();
                        }
                    }
                }
                catch (JsonException) { }

                throw new WeldForgeException(
                    $"{request.Method} {request.RequestUri.PathAndQuery} failed " +
                    $"with HTTP {(int)response.StatusCode}: {detail}",
                    (int)response.StatusCode);
            }

            return string.IsNullOrWhiteSpace(body) ? null : JsonDocument.Parse(body);
        }

        private Task<JsonDocument> GetAsync(string path)
        {
            return SendAsync(new HttpRequestMessage(HttpMethod.Get, BaseUrl + path));
        }

        private Task<JsonDocument> PostJsonAsync(string path, object body)
        {
            var request = new HttpRequestMessage(HttpMethod.Post, BaseUrl + path)
            {
                Content = new StringContent(
                    JsonSerializer.Serialize(body), Encoding.UTF8, "application/json")
            };
            return SendAsync(request);
        }

        // -- service ---------------------------------------------------------

        public async Task<bool> IsReachableAsync()
        {
            try
            {
                using (await GetAsync("/health").ConfigureAwait(false)) { return true; }
            }
            catch (WeldForgeException)
            {
                return false;
            }
        }

        public async Task<string> HealthSummaryAsync()
        {
            using (var document = await GetAsync("/health").ConfigureAwait(false))
            {
                var root = document.RootElement;
                string backend = root.TryGetProperty("cad_backend", out var b) && b.ValueKind != JsonValueKind.Null
                    ? b.GetString() : "NONE";
                bool brep = root.TryGetProperty("brep_supported", out var s) && s.GetBoolean();
                return $"WeldForge {root.GetProperty("version").GetString()} at {BaseUrl}\r\n" +
                       $"CAD backend: {backend}\r\n" +
                       $"STEP support: {(brep ? "yes" : "no")}";
            }
        }

        public async Task<List<string>> ListAsync(string path, string arrayName, string key)
        {
            var items = new List<string>();
            using (var document = await GetAsync(path).ConfigureAwait(false))
            {
                foreach (var element in document.RootElement.GetProperty(arrayName).EnumerateArray())
                {
                    items.Add(element.GetProperty(key).GetString());
                }
            }
            return items;
        }

        public Task<List<string>> CellsAsync() => ListAsync("/cells", "cells", "key");
        public Task<List<string>> MaterialsAsync() => ListAsync("/materials", "materials", "key");
        public Task<List<string>> TargetsAsync() => ListAsync("/targets", "targets", "key");

        // -- parts -----------------------------------------------------------

        /// <summary>Upload a STEP file; returns the part id.</summary>
        public async Task<PartInfo> UploadPartAsync(string path, string name = null)
        {
            using (var content = new MultipartFormDataContent())
            using (var stream = File.OpenRead(path))
            {
                var file = new StreamContent(stream);
                file.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                content.Add(file, "file", Path.GetFileName(path));
                if (!string.IsNullOrEmpty(name))
                {
                    content.Add(new StringContent(name), "name");
                }

                var request = new HttpRequestMessage(HttpMethod.Post, BaseUrl + "/parts")
                {
                    Content = content
                };
                using (var document = await SendAsync(request).ConfigureAwait(false))
                {
                    var root = document.RootElement;
                    return new PartInfo
                    {
                        Id = root.GetProperty("id").GetString(),
                        Name = root.GetProperty("name").GetString(),
                        SolidCount = root.GetProperty("n_solids").GetInt32(),
                        FaceCount = root.GetProperty("n_faces").GetInt32(),
                        MassKg = root.GetProperty("mass_kg").GetDouble()
                    };
                }
            }
        }

        public async Task<SeamSummary> DetectSeamsAsync(string partId)
        {
            using (var document = await GetAsync($"/parts/{partId}/seams").ConfigureAwait(false))
            {
                var root = document.RootElement;
                var summary = new SeamSummary
                {
                    Count = root.GetProperty("n_seams").GetInt32(),
                    TotalLengthMm = root.GetProperty("total_length_mm").GetDouble()
                };
                foreach (var seam in root.GetProperty("seams").EnumerateArray())
                {
                    summary.Seams.Add(new SeamInfo
                    {
                        Id = seam.GetProperty("id").GetString(),
                        Label = seam.GetProperty("label").GetString(),
                        JointType = seam.GetProperty("joint_type").GetString(),
                        LengthMm = seam.GetProperty("length_mm").GetDouble(),
                        Position = seam.GetProperty("position").GetString(),
                        Confidence = seam.GetProperty("confidence").GetDouble()
                    });
                }
                return summary;
            }
        }

        // -- jobs ------------------------------------------------------------

        public async Task<string> StartPlanAsync(
            string partId, string cell, string material, double legRatio)
        {
            var body = new Dictionary<string, object>
            {
                ["part_id"] = partId,
                ["cell"] = cell,
                ["base_material"] = material,
                ["leg_thickness_ratio"] = legRatio
            };
            using (var document = await PostJsonAsync("/jobs", body).ConfigureAwait(false))
            {
                return document.RootElement.GetProperty("job_id").GetString();
            }
        }

        /// <summary>
        /// Poll until the job finishes. Polling rather than one long request:
        /// SolidWorks must stay responsive while a plan runs for minutes.
        /// </summary>
        public async Task<JobResult> WaitForJobAsync(
            string jobId,
            IProgress<string> progress = null,
            CancellationToken cancellation = default,
            int timeoutSeconds = 1800)
        {
            var deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellation.ThrowIfCancellationRequested();
                using (var document = await GetAsync($"/jobs/{jobId}").ConfigureAwait(false))
                {
                    var root = document.RootElement;
                    string status = root.GetProperty("status").GetString();
                    progress?.Report($"{status} ({root.GetProperty("elapsed_s").GetDouble():F0} s)");

                    if (status == "failed")
                    {
                        throw new WeldForgeException(
                            "planning failed: " + root.GetProperty("error").GetString());
                    }
                    if (status == "done")
                    {
                        return JobResult.FromJson(root.GetProperty("result"));
                    }
                }
                await Task.Delay(1000, cancellation).ConfigureAwait(false);
            }
            throw new WeldForgeException($"job {jobId} did not finish within {timeoutSeconds} s");
        }

        public async Task<Dictionary<string, string>> ExportAsync(
            string jobId, string target, string programName = "WELDJOB")
        {
            var body = new Dictionary<string, object>
            {
                ["target"] = target,
                ["program_name"] = programName
            };
            using (var document = await PostJsonAsync($"/jobs/{jobId}/export", body)
                       .ConfigureAwait(false))
            {
                var files = new Dictionary<string, string>();
                foreach (var file in document.RootElement.GetProperty("files").EnumerateObject())
                {
                    files[file.Name] = file.Value.GetString();
                }
                return files;
            }
        }
    }

    // -- data carriers -------------------------------------------------------

    public class PartInfo
    {
        public string Id;
        public string Name;
        public int SolidCount;
        public int FaceCount;
        public double MassKg;
    }

    public class SeamInfo
    {
        public string Id;
        public string Label;
        public string JointType;
        public double LengthMm;
        public string Position;
        public double Confidence;
    }

    public class SeamSummary
    {
        public int Count;
        public double TotalLengthMm;
        public List<SeamInfo> Seams = new List<SeamInfo>();
    }

    public class JobResult
    {
        public bool Ok;
        public bool Reachable;
        public bool CollisionFree;
        public double CycleTimeSeconds;
        public double ArcTimeSeconds;
        public double DutyCycle;
        public double WeldLengthMm;
        public double WireKg;
        public double GasLitres;
        public int SeamCount;
        public int PassCount;
        public List<string> Errors = new List<string>();
        public List<string> Warnings = new List<string>();

        public static JobResult FromJson(JsonElement result)
        {
            var report = result.GetProperty("report");
            var job = new JobResult
            {
                Ok = report.GetProperty("ok").GetBoolean(),
                Reachable = report.GetProperty("reachable").GetBoolean(),
                CollisionFree = report.GetProperty("collision_free").GetBoolean(),
                CycleTimeSeconds = report.GetProperty("cycle_time_s").GetDouble(),
                ArcTimeSeconds = report.GetProperty("arc_time_s").GetDouble(),
                DutyCycle = report.GetProperty("duty_cycle").GetDouble(),
                WeldLengthMm = report.GetProperty("total_weld_length_mm").GetDouble(),
                WireKg = report.GetProperty("wire_consumption_kg").GetDouble(),
                GasLitres = report.GetProperty("gas_consumption_l").GetDouble(),
                SeamCount = result.GetProperty("seams").GetArrayLength()
            };
            foreach (var procedure in result.GetProperty("procedures").EnumerateArray())
            {
                job.PassCount += procedure.GetProperty("passes").GetArrayLength();
            }
            foreach (var error in report.GetProperty("errors").EnumerateArray())
            {
                job.Errors.Add(error.GetString());
            }
            foreach (var warning in report.GetProperty("warnings").EnumerateArray())
            {
                job.Warnings.Add(warning.GetString());
            }
            return job;
        }
    }
}
