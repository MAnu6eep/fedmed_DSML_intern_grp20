import React, { useEffect, useState, useRef } from "react";
import { FlaskConical, Play, Square, RefreshCw, Terminal, CheckCircle2, Trash2, Cpu } from "lucide-react";

interface ExperimentConfig {
  filename: string;
  strategy: string;
  rounds: number;
  local_epochs: number;
  batch_size: number;
  learning_rate: number;
  partition: {
    type?: string;
    alpha?: number;
  };
  extra?: Record<string, any>;
}

interface TestSuite {
  id: string;
  title: string;
  description: string;
  path: string;
}

interface JobStatus {
  id: string | null;
  type: "experiment" | "test" | null;
  name: string | null;
  status: "idle" | "running" | "completed" | "failed" | "stopped";
  command: string[];
  start_time: string | null;
  end_time: string | null;
  return_code: number | null;
  log_count: number;
  logs: string[];
}

export const TestingAndExperiments: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"experiments" | "tests">("experiments");
  const [experiments, setExperiments] = useState<ExperimentConfig[]>([]);
  const [testSuites, setTestSuites] = useState<TestSuite[]>([]);
  const [jobStatus, setJobStatus] = useState<JobStatus>({
    id: null,
    type: null,
    name: null,
    status: "idle",
    command: [],
    start_time: null,
    end_time: null,
    return_code: null,
    log_count: 0,
    logs: [],
  });
  const [autoScroll, setAutoScroll] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const terminalEndRef = useRef<HTMLDivElement>(null);

  const fetchConfigs = () => {
    fetch("http://127.0.0.1:8000/api/jobs/experiments")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setExperiments(data);
      })
      .catch(() => {
        // Fallback default configs if API is offline
        setExperiments([
          { filename: "fedavg.yaml", strategy: "fedavg", rounds: 1, local_epochs: 1, batch_size: 2, learning_rate: 0.0001, partition: { type: "non_iid", alpha: 0.5 } },
          { filename: "fedprox.yaml", strategy: "fedprox", rounds: 1, local_epochs: 1, batch_size: 2, learning_rate: 0.0001, partition: { type: "non_iid", alpha: 0.5 }, extra: { mu: 0.01 } },
          { filename: "scaffold.yaml", strategy: "scaffold", rounds: 1, local_epochs: 1, batch_size: 2, learning_rate: 0.0001, partition: { type: "non_iid", alpha: 0.5 } },
          { filename: "centralized.yaml", strategy: "centralized", rounds: 1, local_epochs: 1, batch_size: 2, learning_rate: 0.0001, partition: { type: "centralized" } },
        ]);
      });

    fetch("http://127.0.0.1:8000/api/jobs/test-suites")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setTestSuites(data);
      })
      .catch(() => {});
  };

  const fetchStatus = () => {
    fetch("http://127.0.0.1:8000/api/jobs/status")
      .then((r) => r.json())
      .then((data) => {
        if (data && data.status) {
          setJobStatus(data);
        }
      })
      .catch(() => {});
  };

  useEffect(() => {
    fetchConfigs();
    fetchStatus();
    const interval = setInterval(fetchStatus, 1500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (autoScroll && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [jobStatus.logs, autoScroll]);

  const handleRunExperiment = async (filename: string) => {
    setActionLoading(`exp-${filename}`);
    try {
      await fetch("http://127.0.0.1:8000/api/jobs/run-experiment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config_file: filename }),
      });
      fetchStatus();
    } catch (e) {
      console.error(e);
    } finally {
      setActionLoading(null);
    }
  };

  const handleRunTest = async (suiteId: string) => {
    setActionLoading(`test-${suiteId}`);
    try {
      await fetch("http://127.0.0.1:8000/api/jobs/run-test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ suite_id: suiteId }),
      });
      fetchStatus();
    } catch (e) {
      console.error(e);
    } finally {
      setActionLoading(null);
    }
  };

  const handleStopJob = async () => {
    try {
      await fetch("http://127.0.0.1:8000/api/jobs/stop", { method: "POST" });
      fetchStatus();
    } catch (e) {
      console.error(e);
    }
  };

  const handleClearLogs = async () => {
    try {
      await fetch("http://127.0.0.1:8000/api/jobs/clear-logs", { method: "POST" });
      setJobStatus((prev) => ({ ...prev, logs: [] }));
    } catch (e) {
      console.error(e);
    }
  };

  const isRunning = jobStatus.status === "running";

  return (
    <div className="p-8 space-y-8 text-slate-200">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-navy-700 border border-navy-600 rounded-xl text-beige-100 shadow-sm">
              <FlaskConical className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white tracking-wide">Testings & Experimenting</h2>
              <p className="text-sm text-slate-400">
                On-demand benchmarking and validation suite decoupled from core startup
              </p>
            </div>
          </div>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center space-x-3 bg-navy-800 border border-navy-700 px-4 py-2.5 rounded-xl shadow-sm">
          <div className="flex items-center space-x-2">
            <span
              className={`w-3 h-3 rounded-full ${
                isRunning
                  ? "bg-amber-400 animate-ping"
                  : jobStatus.status === "completed"
                  ? "bg-green-400"
                  : jobStatus.status === "failed"
                  ? "bg-red-400"
                  : "bg-slate-500"
              }`}
            />
            <span className="text-xs uppercase font-bold tracking-wider text-slate-300">
              {jobStatus.status === "running"
                ? `Running: ${jobStatus.name}`
                : jobStatus.status === "completed"
                ? `Done: ${jobStatus.name}`
                : jobStatus.status === "failed"
                ? `Failed: ${jobStatus.name}`
                : "System Idle"}
            </span>
          </div>

          {isRunning && (
            <button
              onClick={handleStopJob}
              className="ml-2 flex items-center space-x-1.5 bg-red-900/40 hover:bg-red-900/70 border border-red-500/40 text-red-300 px-3 py-1 rounded-lg text-xs font-semibold transition"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Stop Job</span>
            </button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex space-x-2 border-b border-navy-700 pb-2">
        <button
          onClick={() => setActiveTab("experiments")}
          className={`flex items-center space-x-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition ${
            activeTab === "experiments"
              ? "bg-navy-700 text-white shadow-sm"
              : "text-slate-400 hover:text-white hover:bg-navy-800"
          }`}
        >
          <Cpu className="w-4 h-4" />
          <span>Federated Experiments ({experiments.length})</span>
        </button>

        <button
          onClick={() => setActiveTab("tests")}
          className={`flex items-center space-x-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition ${
            activeTab === "tests"
              ? "bg-navy-700 text-white shadow-sm"
              : "text-slate-400 hover:text-white hover:bg-navy-800"
          }`}
        >
          <CheckCircle2 className="w-4 h-4" />
          <span>Automated Test Suites ({testSuites.length})</span>
        </button>
      </div>

      {/* TAB 1: Federated Experiments */}
      {activeTab === "experiments" && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {experiments.map((exp) => (
            <div
              key={exp.filename}
              className="bg-navy-800 border border-navy-700 rounded-xl p-5 flex flex-col justify-between hover:border-navy-600 transition shadow-sm space-y-4"
            >
              <div>
                <div className="flex justify-between items-start mb-2">
                  <span className="text-xs font-mono uppercase bg-navy-900 text-blue-300 px-2 py-0.5 rounded border border-navy-700">
                    {exp.strategy.toUpperCase()}
                  </span>
                  <span className="text-xs text-slate-400 font-mono">{exp.filename}</span>
                </div>

                <h3 className="text-lg font-bold text-white capitalize mt-2">
                  {exp.strategy === "centralized"
                    ? "Centralized Baseline"
                    : `${exp.strategy} Segmentation`}
                </h3>

                <p className="text-xs text-slate-400 mt-1">
                  {exp.strategy === "fedavg" && "Baseline federated weight averaging across hospital nodes."}
                  {exp.strategy === "fedprox" && "Proximal regularization (mu=0.01) addressing non-IID drift."}
                  {exp.strategy === "scaffold" && "Control variates correction for heterogeneous 3D MRI data."}
                  {exp.strategy === "centralized" && "Pooled upper-bound reference model (non-federated)."}
                </p>

                {/* Hyperparameters pill group */}
                <div className="mt-4 pt-3 border-t border-navy-700/60 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-slate-500 block">Rounds</span>
                    <span className="font-semibold text-slate-200">{exp.rounds}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Local Epochs</span>
                    <span className="font-semibold text-slate-200">{exp.local_epochs}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Batch Size</span>
                    <span className="font-semibold text-slate-200">{exp.batch_size}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Learning Rate</span>
                    <span className="font-semibold text-slate-200">{exp.learning_rate}</span>
                  </div>
                  {exp.partition && (
                    <div className="col-span-2">
                      <span className="text-slate-500 block">Partition</span>
                      <span className="font-semibold text-blue-300">
                        {exp.partition.type || "IID"}{" "}
                        {exp.partition.alpha !== undefined ? `(alpha=${exp.partition.alpha})` : ""}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              <button
                onClick={() => handleRunExperiment(exp.filename)}
                disabled={isRunning}
                className={`w-full flex items-center justify-center space-x-2 py-2.5 rounded-lg text-xs font-bold transition shadow-sm ${
                  isRunning
                    ? "bg-navy-700 text-slate-500 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/30"
                }`}
              >
                {actionLoading === `exp-${exp.filename}` ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4 fill-current" />
                )}
                <span>Run {exp.strategy.toUpperCase()}</span>
              </button>
            </div>
          ))}
        </div>
      )}

      {/* TAB 2: Automated Test Suites */}
      {activeTab === "tests" && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {testSuites.map((suite) => (
            <div
              key={suite.id}
              className="bg-navy-800 border border-navy-700 rounded-xl p-5 flex flex-col justify-between hover:border-navy-600 transition shadow-sm space-y-4"
            >
              <div>
                <div className="flex justify-between items-start mb-2">
                  <span className="text-xs font-mono uppercase bg-navy-900 text-green-300 px-2.5 py-0.5 rounded border border-navy-700">
                    TEST SUITE
                  </span>
                  <span className="text-xs text-slate-400 font-mono truncate max-w-[150px]">
                    {suite.id}
                  </span>
                </div>

                <h3 className="text-base font-bold text-white mt-2">{suite.title}</h3>
                <p className="text-xs text-slate-400 mt-1">{suite.description}</p>

                <div className="mt-3 p-2 bg-navy-900/60 rounded border border-navy-700 font-mono text-[11px] text-slate-400 truncate">
                  pytest {suite.path}
                </div>
              </div>

              <button
                onClick={() => handleRunTest(suite.id)}
                disabled={isRunning}
                className={`w-full flex items-center justify-center space-x-2 py-2.5 rounded-lg text-xs font-bold transition shadow-sm ${
                  isRunning
                    ? "bg-navy-700 text-slate-500 cursor-not-allowed"
                    : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/30"
                }`}
              >
                {actionLoading === `test-${suite.id}` ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4 fill-current" />
                )}
                <span>Run Test Suite</span>
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Embedded Live Terminal & Console Output */}
      <div className="bg-navy-950 border border-navy-700 rounded-xl overflow-hidden shadow-xl">
        <div className="bg-navy-900 border-b border-navy-700 px-5 py-3 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Terminal className="w-4 h-4 text-blue-400" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono">
              Live Console Output {jobStatus.name ? `[${jobStatus.name}]` : ""}
            </span>
            <span className="text-xs text-slate-500 font-mono">({jobStatus.logs.length} lines)</span>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            <label className="flex items-center space-x-1.5 text-slate-400 cursor-pointer">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="rounded bg-navy-800 border-navy-600 text-blue-500 focus:ring-0"
              />
              <span>Auto-scroll</span>
            </label>

            <button
              onClick={handleClearLogs}
              className="flex items-center space-x-1 text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-navy-800 transition"
              title="Clear terminal buffer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear</span>
            </button>
          </div>
        </div>

        {/* Console Window */}
        <div className="p-4 h-72 overflow-y-auto font-mono text-xs leading-relaxed space-y-1 bg-black/50 select-text">
          {jobStatus.logs.length === 0 ? (
            <div className="text-slate-600 italic h-full flex flex-col items-center justify-center space-y-2">
              <Terminal className="w-8 h-8 opacity-30" />
              <span>No active job output. Select an experiment or test suite above to begin execution.</span>
            </div>
          ) : (
            jobStatus.logs.map((line, idx) => {
              const isError = line.toLowerCase().includes("failed") || line.toLowerCase().includes("error");
              const isPassed = line.includes("PASSED") || line.includes("completed");
              const isHeader = line.startsWith("===") || line.startsWith("Command:");

              return (
                <div
                  key={idx}
                  className={`${
                    isError
                      ? "text-red-400"
                      : isPassed
                      ? "text-emerald-400"
                      : isHeader
                      ? "text-blue-300 font-bold"
                      : "text-slate-300"
                  }`}
                >
                  {line}
                </div>
              );
            })
          )}
          <div ref={terminalEndRef} />
        </div>
      </div>
    </div>
  );
};
