import React, { useEffect, useState } from "react";
import { 
  Box, 
  Cpu, 
  HardDrive, 
  Plus, 
  ExternalLink, 
  Trash2, 
  X, 
  RefreshCw, 
  AlertTriangle,
  Server,
  Play,
  Square,
  RotateCcw,
  Activity
} from "lucide-react";

interface SystemResources {
  ram: {
    total_gb: number;
    used_gb: number;
    avail_gb: number;
    percent_used: number;
  };
  cpu?: {
    load_percent: number;
    logical_cores: number;
    physical_cores: number;
  };
  disk: {
    total_gb: number;
    used_gb: number;
    avail_gb: number;
    percent_used: number;
  };
  docker: {
    running: boolean;
    active_containers: number;
  };
}

interface SimulatedHospital {
  hospital_id: string;
  name: string;
  host: string;
  port: number;
  endpoint: string;
  registration_status: string;
  connection_status: string;
  approval_status: string;
  participation_status: string;
  model_status: string;
  container_id: string | null;
  current_round: number;
  last_heartbeat?: string | null;
}

interface DockerContainerItem {
  container_id: string;
  name: string;
  image: string;
  state: string;
  status: string;
  created_at: string;
  ports: string;
}

interface SimulateRemoteHospitalProps {
  onOpenHospital: (hospitalId: string) => void;
  onSwitchToStandby?: () => void;
}

export const SimulateRemoteHospital: React.FC<SimulateRemoteHospitalProps> = ({
  onOpenHospital,
  onSwitchToStandby,
}) => {
  const [hospitals, setHospitals] = useState<SimulatedHospital[]>([]);
  const [dockerContainers, setDockerContainers] = useState<{ active: DockerContainerItem[]; stopped: DockerContainerItem[] }>({
    active: [],
    stopped: [],
  });
  const [resources, setResources] = useState<SystemResources | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [dockerStartupError, setDockerStartupError] = useState<string | null>(null);

  // Form
  const [formName, setFormName] = useState("");
  const [formId, setFormId] = useState("");
  const [formPort, setFormPort] = useState<number | "">("");

  const loadData = () => {
    fetch("http://127.0.0.1:8000/api/system/resources")
      .then((r) => r.json())
      .then((data) => setResources(data))
      .catch(() => {});

    fetch("http://127.0.0.1:8000/api/docker/simulated-hospitals")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setHospitals(data);
      })
      .catch(() => {});

    fetch("http://127.0.0.1:8000/api/docker/containers")
      .then((r) => r.json())
      .then((data) => {
        if (data && Array.isArray(data.active)) {
          setDockerContainers(data);
        }
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleStartDockerSim = async () => {
    setDockerStartupError(null);
    setLoading("starting-docker");
    try {
      const res = await fetch("http://127.0.0.1:8000/api/docker/start-simulation", {
        method: "POST",
      });
      if (res.ok) {
        loadData();
      } else {
        const err = await res.json();
        setDockerStartupError(err.detail || "Failed to start Docker Compose infrastructure.");
      }
    } catch (e: any) {
      setDockerStartupError(e.message || "Failed to communicate with Docker backend.");
    } finally {
      setLoading(null);
    }
  };

  const handleInitiateContainer = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formName.trim()) return;

    setErrorMsg(null);
    setLoading("initiating");

    try {
      const res = await fetch("http://127.0.0.1:8000/api/docker/initiate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formName.trim(),
          hospital_id: formId.trim() || undefined,
          port: formPort ? Number(formPort) : undefined,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        setErrorMsg(err.detail || "Failed to initiate container");
      } else {
        const newHosp = await res.json();
        setShowModal(false);
        setFormName("");
        setFormId("");
        setFormPort("");
        loadData();
        // Immediately navigate to the newly created hospital UI
        if (newHosp?.hospital_id) {
          onOpenHospital(newHosp.hospital_id);
        }
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Network error");
    } finally {
      setLoading(null);
    }
  };

  // Hospital-level Docker container controls (using stable hospital_id)
  const handleStartHospital = async (hospitalId: string) => {
    setLoading(`start-${hospitalId}`);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/hospitals/${encodeURIComponent(hospitalId)}/docker/start`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || "Failed to start container in Docker Desktop");
      }
      loadData();
    } catch (e: any) {
      console.error(e);
      alert(e.message || "Network error");
    } finally {
      setLoading(null);
    }
  };

  const handleStopHospital = async (hospitalId: string) => {
    setLoading(`stop-${hospitalId}`);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/hospitals/${encodeURIComponent(hospitalId)}/docker/stop`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || "Failed to stop container in Docker Desktop");
      }
      loadData();
    } catch (e: any) {
      console.error(e);
      alert(e.message || "Network error");
    } finally {
      setLoading(null);
    }
  };

  const handleRestartHospital = async (hospitalId: string) => {
    setLoading(`restart-${hospitalId}`);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/hospitals/${encodeURIComponent(hospitalId)}/docker/restart`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || "Failed to restart container in Docker Desktop");
      }
      loadData();
    } catch (e: any) {
      console.error(e);
      alert(e.message || "Network error");
    } finally {
      setLoading(null);
    }
  };

  // Container lifecycle controls (for raw container list)
  const handleStartContainer = async (containerId: string) => {
    setLoading(`start-${containerId}`);
    try {
      await fetch(`http://127.0.0.1:8000/api/docker/containers/${encodeURIComponent(containerId)}/start`, { method: "POST" });
      loadData();
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(null);
    }
  };


  const handleRemoveContainer = async (containerId: string) => {
    if (!confirm(`Are you sure you want to remove container ${containerId}?`)) return;
    setLoading(`remove-${containerId}`);
    try {
      await fetch(`http://127.0.0.1:8000/api/docker/containers/${containerId}`, { method: "DELETE" });
      loadData();
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(null);
    }
  };

  const handleDeleteHospital = async (hid: string) => {
    if (!confirm(`Are you sure you want to remove hospital ${hid}?`)) return;
    try {
      await fetch(`http://127.0.0.1:8000/api/hospitals/${hid}`, { method: "DELETE" });
      loadData();
    } catch (e) {
      console.error(e);
    }
  };

  const isDockerRunning = resources?.docker?.running ?? false;
  const isMemorySufficient = (resources?.ram?.avail_gb ?? 1.0) >= 0.2;

  // Check if hospital container is actively running in Docker or connected
  const isContainerRunning = (h: SimulatedHospital) => {
    if (!h.container_id) return h.connection_status === "CONNECTED";
    return dockerContainers.active.some(
      (x) => x.container_id.startsWith(h.container_id!.slice(0, 12)) ||
             (h.container_id && x.container_id === h.container_id) ||
             x.name.includes(h.container_id!) ||
             x.name.toLowerCase().includes(h.hospital_id.toLowerCase().replace(/[^a-z0-9]/g, ""))
    ) || h.connection_status === "CONNECTED";
  };

  return (
    <div className="p-8 space-y-8 text-slate-200">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-2xl font-bold text-white tracking-wide">
              Simulate Remote Hospital
            </h2>
            <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase font-mono border ${
              isDockerRunning
                ? "bg-blue-500/20 text-blue-300 border-blue-500/40"
                : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
            }`}>
              {isDockerRunning ? "Docker Simulation Mode" : "Native Standby Mode"}
            </span>
          </div>
          <p className="text-sm text-slate-400">
            Real multi-container simulation of remote hospital nodes running isolated federated learning environments
          </p>
        </div>

        <button
          onClick={() => {
            setErrorMsg(null);
            setShowModal(true);
          }}
          className="bg-blue-600 hover:bg-blue-500 text-white flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-semibold transition shadow-sm"
        >
          <Plus className="w-4 h-4" />
          <span>Initiate Remote Hospital</span>
        </button>
      </div>

      {/* Docker Failure Fallback Banner */}
      {!isDockerRunning && (
        <div className="bg-amber-950/30 border border-amber-500/40 rounded-xl p-5 space-y-4">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-bold text-white">
                Docker Desktop is currently not running.
              </h3>
              <p className="text-xs text-slate-300 mt-1">
                FedMed can continue in Standby Mode without Docker, or you can start Docker to simulate remote hospitals using isolated containers.
              </p>
            </div>
          </div>

          {dockerStartupError && (
            <div className="bg-red-950/40 border border-red-500/40 p-2.5 rounded text-[11px] text-red-200 font-mono">
              <strong>Error:</strong> {dockerStartupError}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3 pt-1">
            <button
              onClick={handleStartDockerSim}
              disabled={loading === "starting-docker"}
              className="bg-blue-600 hover:bg-blue-500 text-white font-bold px-4 py-2 rounded-lg text-xs flex items-center space-x-1.5 transition shadow-sm"
            >
              {loading === "starting-docker" ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Starting Docker...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Start Docker Simulation</span>
                </>
              )}
            </button>

            {onSwitchToStandby && (
              <button
                onClick={onSwitchToStandby}
                className="bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 font-bold px-4 py-2 rounded-lg text-xs transition"
              >
                Continue in Standby Mode
              </button>
            )}

            <button
              onClick={loadData}
              className="bg-navy-800 hover:bg-navy-700 text-slate-300 border border-navy-700 font-medium px-3 py-2 rounded-lg text-xs flex items-center space-x-1 transition"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Retry</span>
            </button>
          </div>
        </div>
      )}

      {/* Real System Resources Cards */}
      {resources && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* RAM */}
          <div className="bg-navy-800 border border-navy-700 rounded-xl p-4 space-y-2 shadow-sm">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 font-semibold uppercase flex items-center">
                <Cpu className="w-4 h-4 mr-1 text-blue-400" /> Host RAM
              </span>
              <span className="font-mono text-white font-bold">
                {resources.ram.used_gb} / {resources.ram.total_gb} GB
              </span>
            </div>
            <div className="w-full bg-navy-900 rounded-full h-1.5 overflow-hidden border border-navy-700">
              <div
                className={`h-full transition-all duration-500 ${
                  resources.ram.percent_used > 85 ? "bg-red-500" : "bg-blue-500"
                }`}
                style={{ width: `${Math.min(resources.ram.percent_used, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-400 font-mono">
              <span>Free: {resources.ram.avail_gb} GB</span>
              <span>Load: {resources.ram.percent_used}%</span>
            </div>
          </div>

          {/* CPU */}
          <div className="bg-navy-800 border border-navy-700 rounded-xl p-4 space-y-2 shadow-sm">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 font-semibold uppercase flex items-center">
                <Activity className="w-4 h-4 mr-1 text-cyan-400" /> Host CPU
              </span>
              <span className="font-mono text-white font-bold">
                {resources.cpu ? `${resources.cpu.load_percent}%` : "--"}
              </span>
            </div>
            <div className="w-full bg-navy-900 rounded-full h-1.5 overflow-hidden border border-navy-700">
              <div
                className="h-full bg-cyan-500 transition-all duration-500"
                style={{ width: `${Math.min(resources.cpu?.load_percent ?? 15, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-400 font-mono">
              <span>Logical: {resources.cpu?.logical_cores ?? 16} Cores</span>
              <span>Physical: {resources.cpu?.physical_cores ?? 12}</span>
            </div>
          </div>

          {/* Disk */}
          <div className="bg-navy-800 border border-navy-700 rounded-xl p-4 space-y-2 shadow-sm">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 font-semibold uppercase flex items-center">
                <HardDrive className="w-4 h-4 mr-1 text-emerald-400" /> Host Disk
              </span>
              <span className="font-mono text-white font-bold">
                {resources.disk.avail_gb} GB Free
              </span>
            </div>
            <div className="w-full bg-navy-900 rounded-full h-1.5 overflow-hidden border border-navy-700">
              <div
                className="h-full bg-emerald-500 transition-all duration-500"
                style={{ width: `${Math.min(resources.disk.percent_used, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-400 font-mono">
              <span>Used: {resources.disk.used_gb} GB</span>
              <span>Total: {resources.disk.total_gb} GB</span>
            </div>
          </div>

          {/* Docker Status */}
          <div className="bg-navy-800 border border-navy-700 rounded-xl p-4 space-y-2 shadow-sm">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 font-semibold uppercase flex items-center">
                <Server className="w-4 h-4 mr-1 text-purple-400" /> Docker Engine
              </span>
              <span
                className={`w-2 h-2 rounded-full ${
                  resources.docker.running ? "bg-emerald-400" : "bg-amber-400"
                }`}
              />
            </div>
            <div className="text-sm font-bold text-white truncate">
              {resources.docker.running ? "Active & Running" : "Stopped / Standby"}
            </div>
            <div className="flex justify-between text-[10px] text-slate-400 font-mono">
              <span>Active Containers: {resources.docker.active_containers}</span>
              <span>Stopped: {dockerContainers.stopped.length}</span>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 1: ACTIVE HOSPITAL CONTAINERS */}
      <div className="bg-navy-800 border border-navy-700 rounded-xl overflow-hidden shadow-sm space-y-0">
        <div className="p-5 border-b border-navy-700 flex justify-between items-center bg-navy-850">
          <div>
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>ACTIVE HOSPITAL CONTAINERS</span>
            </h3>
            <p className="text-xs text-slate-400">
              Live independent hospital instances participating in federated learning cycles
            </p>
          </div>
          <span className="text-xs font-mono bg-navy-900 text-emerald-300 px-3 py-1 rounded-full border border-navy-700">
            {hospitals.length} node(s) ({hospitals.filter(isContainerRunning).length} active)
          </span>
        </div>

        {hospitals.length === 0 ? (
          <div className="p-10 text-center space-y-3">
            <Box className="w-8 h-8 mx-auto text-slate-500 opacity-40" />
            <h4 className="text-sm font-bold text-slate-300">
              No remote hospital containers are currently registered.
            </h4>
            <p className="text-xs text-slate-500 max-w-sm mx-auto">
              Initiate a new container or start a previously configured hospital node below.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-navy-900/60 text-slate-400 uppercase text-[10px] font-semibold border-b border-navy-700">
                <tr>
                  <th className="px-5 py-3">Hospital Name & ID</th>
                  <th className="px-4 py-3">Container ID</th>
                  <th className="px-4 py-3">Connection</th>
                  <th className="px-4 py-3">Approval Status</th>
                  <th className="px-4 py-3">Training Status</th>
                  <th className="px-4 py-3">Model State</th>
                  <th className="px-5 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-700/60">
                {hospitals.map((h) => {
                  const running = isContainerRunning(h);
                  return (
                  <tr key={h.hospital_id} className="hover:bg-navy-700/20 transition-colors">
                    <td className="px-5 py-3">
                      <div className="font-bold text-white text-sm">{h.name}</div>
                      <div className="text-[11px] text-slate-400 font-mono">
                        {h.hospital_id} • Port :{h.port}
                      </div>
                    </td>

                    <td className="px-4 py-3 font-mono text-cyan-300 text-[11px]">
                      {h.container_id ? (
                        <span className="px-2 py-0.5 rounded bg-navy-900 border border-navy-700">
                          {h.container_id.slice(0, 12)}
                        </span>
                      ) : (
                        <span className="text-slate-500">Native Process</span>
                      )}
                    </td>

                    <td className="px-4 py-3">
                      {running ? (
                        <span className="inline-flex items-center space-x-1.5 text-emerald-400 font-semibold text-[11px]">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                          <span>CONNECTED</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center space-x-1.5 text-amber-400 font-semibold text-[11px]">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                          <span>STOPPED</span>
                        </span>
                      )}
                    </td>

                    <td className="px-4 py-3">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                          h.approval_status === "APPROVED"
                            ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                            : "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                        }`}
                      >
                        {h.approval_status}
                      </span>
                    </td>

                    <td className="px-4 py-3 font-mono text-slate-300 text-[11px]">
                      {h.participation_status}
                    </td>

                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-navy-900 border border-navy-700 text-slate-300">
                        {h.model_status}
                      </span>
                    </td>

                    <td className="px-5 py-3 text-right">
                      <div className="flex items-center justify-end space-x-1.5">
                        <button
                          onClick={() => onOpenHospital(h.hospital_id)}
                          className="bg-blue-600 hover:bg-blue-500 text-white px-3 py-1.5 rounded text-xs font-semibold flex items-center space-x-1 transition shadow-sm"
                        >
                          <span>Open Hospital</span>
                          <ExternalLink className="w-3.5 h-3.5" />
                        </button>
                        {running ? (
                          <>
                            <button
                              onClick={() => handleStopHospital(h.hospital_id)}
                              disabled={loading === `stop-${h.hospital_id}`}
                              className="text-amber-400 hover:text-amber-300 p-1.5 rounded hover:bg-navy-700 transition"
                              title="Stop container in Docker Desktop"
                            >
                              <Square className="w-3.5 h-3.5" />
                            </button>
                            <button
                              onClick={() => handleRestartHospital(h.hospital_id)}
                              disabled={loading === `restart-${h.hospital_id}`}
                              className="text-slate-400 hover:text-white p-1.5 rounded hover:bg-navy-700 transition"
                              title="Restart container in Docker Desktop"
                            >
                              <RotateCcw className="w-3.5 h-3.5" />
                            </button>
                          </>
                        ) : (
                          <button
                            onClick={() => handleStartHospital(h.hospital_id)}
                            disabled={loading === `start-${h.hospital_id}`}
                            className="text-emerald-400 hover:text-emerald-300 p-1.5 rounded hover:bg-navy-700 transition"
                            title="Start container in Docker Desktop"
                          >
                            <Play className="w-3.5 h-3.5 fill-current" />
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteHospital(h.hospital_id)}
                          className="text-slate-500 hover:text-red-400 p-1.5 rounded hover:bg-navy-700 transition"
                          title="Remove hospital node"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* SECTION 2: PREVIOUS / STOPPED HOSPITAL CONTAINERS */}
      <div className="bg-navy-800 border border-navy-700 rounded-xl overflow-hidden shadow-sm space-y-0">
        <div className="p-5 border-b border-navy-700 flex justify-between items-center bg-navy-850">
          <div>
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-500" />
              <span>PREVIOUS / STOPPED HOSPITAL CONTAINERS</span>
            </h3>
            <p className="text-xs text-slate-400">
              Container images and stopped hospital instances available to resume or clean up
            </p>
          </div>
          <span className="text-xs font-mono bg-navy-900 text-slate-400 px-3 py-1 rounded-full border border-navy-700">
            {dockerContainers.stopped.length} stopped container(s)
          </span>
        </div>

        {dockerContainers.stopped.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-xs">
            No stopped hospital containers on host.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-navy-900/60 text-slate-400 uppercase text-[10px] font-semibold border-b border-navy-700">
                <tr>
                  <th className="px-5 py-3">Container ID & Name</th>
                  <th className="px-4 py-3">Docker Image</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Created</th>
                  <th className="px-5 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-700/60">
                {dockerContainers.stopped.map((c) => (
                  <tr key={c.container_id} className="hover:bg-navy-700/20 transition-colors">
                    <td className="px-5 py-3">
                      <div className="font-bold text-slate-200 text-xs">{c.name}</div>
                      <div className="text-[10px] text-slate-500 font-mono">{c.container_id}</div>
                    </td>

                    <td className="px-4 py-3 font-mono text-slate-400 text-[11px] truncate max-w-xs">
                      {c.image}
                    </td>

                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-navy-900 border border-navy-700 text-slate-400">
                        {c.status}
                      </span>
                    </td>

                    <td className="px-4 py-3 text-slate-500 text-[11px]">
                      {c.created_at}
                    </td>

                    <td className="px-5 py-3 text-right">
                      <div className="flex items-center justify-end space-x-2">
                        <button
                          onClick={() => handleStartContainer(c.container_id)}
                          disabled={loading === `start-${c.container_id}`}
                          className="bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 px-3 py-1.5 rounded text-xs font-semibold flex items-center space-x-1 transition"
                        >
                          <Play className="w-3 h-3 fill-current" />
                          <span>Start Container</span>
                        </button>
                        <button
                          onClick={() => handleRemoveContainer(c.container_id)}
                          disabled={loading === `remove-${c.container_id}`}
                          className="text-slate-500 hover:text-red-400 p-1.5 rounded hover:bg-navy-700 transition"
                          title="Remove container"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* INITIATE REMOTE HOSPITAL MODAL */}
      {showModal && (
        <div className="fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4">
          <div className="bg-navy-800 border border-navy-700 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-5">
            <div className="flex justify-between items-center border-b border-navy-700 pb-3">
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <Box className="w-4 h-4 text-blue-400" />
                <span>Initiate Remote Hospital</span>
              </h3>
              <button
                onClick={() => setShowModal(false)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Real Resource Check in Modal */}
            <div className={`p-3 rounded-lg border text-xs space-y-1.5 ${
              isMemorySufficient
                ? "bg-navy-900 border-navy-700 text-slate-300"
                : "bg-red-950/40 border-red-500/40 text-red-200"
            }`}>
              <div className="flex justify-between font-semibold">
                <span>System Resource Evaluation:</span>
                <span className={isMemorySufficient ? "text-emerald-400" : "text-red-400"}>
                  {isMemorySufficient ? "Resources Sufficient" : "Memory Constrained"}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Available RAM: <strong className="text-white font-mono">{resources?.ram.avail_gb ?? 0} GB</strong> (Minimum required: 0.20 GB).
              </p>
              {!isMemorySufficient && (
                <p className="text-[11px] text-red-300">
                  Container initiation blocked to avoid workstation memory exhaustion.
                </p>
              )}
            </div>

            {errorMsg && (
              <div className="bg-red-950/40 border border-red-500/40 p-2.5 rounded text-xs text-red-200">
                {errorMsg}
              </div>
            )}

            <form onSubmit={handleInitiateContainer} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Hospital Name *
                </label>
                <input
                  type="text"
                  placeholder="e.g. Mayo Clinic Neurology"
                  value={formName}
                  onChange={(e) => {
                    setFormName(e.target.value);
                    if (!formId) {
                      setFormId(e.target.value.toLowerCase().replace(/\s+/g, "_"));
                    }
                  }}
                  required
                  className="w-full bg-navy-900 border border-navy-600 rounded px-3 py-2 text-white text-xs focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Hospital ID (Slug)
                </label>
                <input
                  type="text"
                  placeholder="e.g. mayo_neuro"
                  value={formId}
                  onChange={(e) => setFormId(e.target.value)}
                  className="w-full bg-navy-900 border border-navy-600 rounded px-3 py-2 text-white font-mono text-xs focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Container Port (Optional)
                </label>
                <input
                  type="number"
                  placeholder="Auto-allocated if blank (e.g. 8081)"
                  value={formPort}
                  onChange={(e) => setFormPort(e.target.value ? Number(e.target.value) : "")}
                  className="w-full bg-navy-900 border border-navy-600 rounded px-3 py-2 text-white font-mono text-xs focus:outline-none focus:border-blue-500"
                />
              </div>

              <div className="pt-2 flex space-x-3">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="flex-1 bg-navy-900 hover:bg-navy-700 text-slate-400 py-2.5 rounded-lg text-xs font-semibold transition border border-navy-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading === "initiating" || !isMemorySufficient}
                  className={`flex-1 py-2.5 rounded-lg text-xs font-bold transition flex items-center justify-center space-x-1.5 shadow-sm ${
                    isMemorySufficient
                      ? "bg-blue-600 hover:bg-blue-500 text-white"
                      : "bg-navy-700 text-slate-500 cursor-not-allowed"
                  }`}
                >
                  {loading === "initiating" ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Allocating Node...</span>
                    </>
                  ) : (
                    <span>Create Hospital</span>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
