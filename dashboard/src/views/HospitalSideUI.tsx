import React, { useEffect, useState } from "react";
import { 
  ArrowLeft, 
  HardDrive, 
  Database, 
  BrainCircuit, 
  ShieldCheck, 
  Upload, 
  CheckCircle2, 
  RefreshCw,
  FolderOpen,
  LineChart as LineChartIcon,
  Activity
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";

interface TrainingMetricPoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
  dice: number;
  elapsed_seconds: number;
}

interface HospitalSideUIProps {
  hospitalId: string | null;
  onBack: () => void;
}

export const HospitalSideUI: React.FC<HospitalSideUIProps> = ({ hospitalId, onBack }) => {
  const [hospital, setHospital] = useState<any>(null);
  const [loadingAction, setLoadingAction] = useState<string | null>(null);
  const [localPathInput, setLocalPathInput] = useState("");
  const [targetEpochs, setTargetEpochs] = useState<number>(2);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const loadHospitalData = () => {
    if (!hospitalId) return;
    fetch("http://127.0.0.1:8000/api/hospitals")
      .then((r) => r.json())
      .then((list) => {
        if (Array.isArray(list)) {
          const found = list.find((h: any) => h.hospital_id === hospitalId);
          if (found) setHospital(found);
        }
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadHospitalData();
    const interval = setInterval(loadHospitalData, 2000);
    return () => clearInterval(interval);
  }, [hospitalId]);

  if (!hospitalId || !hospital) {
    return (
      <div className="p-8 text-center text-slate-400">
        <p>Loading hospital node environment...</p>
        <button onClick={onBack} className="mt-4 text-blue-400 underline text-xs">
          Return to Registry
        </button>
      </div>
    );
  }

  // 1. Data Configuration
  const handleConfigureDemoData = async () => {
    setLoadingAction("data-demo");
    setStatusMessage(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/hospitals/${hospitalId}/configure-data`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "demo" }),
      });
      const data = await res.json();
      setHospital(data);
      setStatusMessage("Demo synthetic 3D MRI dataset configured successfully.");
    } catch (e: any) {
      setStatusMessage(`Error: ${e.message}`);
    } finally {
      setLoadingAction(null);
    }
  };

  const handleConfigureLocalData = async () => {
    if (!localPathInput.trim()) return;
    setLoadingAction("data-local");
    setStatusMessage(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/hospitals/${hospitalId}/configure-data`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "local", local_path: localPathInput.trim() }),
      });
      if (res.ok) {
        const data = await res.json();
        setHospital(data);
        setStatusMessage(`Local dataset connected (${data.sample_count} volumes validated).`);
      } else {
        const err = await res.json();
        setStatusMessage(`Error: ${err.detail}`);
      }
    } catch (e: any) {
      setStatusMessage(`Error: ${e.message}`);
    } finally {
      setLoadingAction(null);
    }
  };

  // 2. Model Initialization
  const handleInitModel = async () => {
    setLoadingAction("init-model");
    setStatusMessage(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/hospitals/${hospitalId}/init-model`, {
        method: "POST",
      });
      const data = await res.json();
      setHospital(data);
      setStatusMessage("Global Model weights instantiated and synchronized into local memory.");
    } finally {
      setLoadingAction(null);
    }
  };

  // 3. Request Approval
  const handleRequestApproval = async () => {
    setLoadingAction("req-approval");
    setStatusMessage(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/hospitals/${hospitalId}/request-approval`, {
        method: "POST",
      });
      const data = await res.json();
      setHospital(data);
      setStatusMessage("Federation approval requested from central administrator.");
    } finally {
      setLoadingAction(null);
    }
  };

  // 4. Start Real Local Training
  const handleStartTraining = async () => {
    setLoadingAction("training");
    setStatusMessage(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/hospitals/${hospitalId}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs: targetEpochs }),
      });
      if (!res.ok) {
        const err = await res.json();
        setStatusMessage(`Training failed: ${err.detail}`);
      } else {
        const data = await res.json();
        setHospital(data);
        setStatusMessage(`Local training completed (${targetEpochs} epoch(s)). Loss: ${data.current_loss}, Dice: ${data.current_dice}`);
      }
    } catch (e: any) {
      setStatusMessage(`Training error: ${e.message}`);
    } finally {
      setLoadingAction(null);
    }
  };

  // 5. Submit Model Update
  const handleSubmitModel = async () => {
    setLoadingAction("submit-model");
    setStatusMessage(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/hospitals/${hospitalId}/submit-model`, {
        method: "POST",
      });
      if (res.ok) {
        setStatusMessage("Permitted model update submitted to central federation server.");
        loadHospitalData();
      } else {
        const err = await res.json();
        setStatusMessage(`Error submitting model: ${err.detail}`);
      }
    } catch (e: any) {
      setStatusMessage(`Error: ${e.message}`);
    } finally {
      setLoadingAction(null);
    }
  };

  const isApproved = hospital.approval_status === "APPROVED";
  const hasData = hospital.sample_count > 0;
  const isModelReady = hospital.model_status !== "NOT_INITIALIZED";
  const isTrained = hospital.model_status === "TRAINED";
  const isSubmitted = hospital.model_status === "SUBMITTED";

  // Derive explicit 12-state status
  const getLifecycleState = () => {
    if (!hasData) return "DATA NOT CONFIGURED";
    if (!isModelReady) return "WAITING FOR MODEL";
    if (!isApproved) return "APPROVAL PENDING";
    if (loadingAction === "training" || hospital.model_status === "TRAINING") return "TRAINING";
    if (isTrained) return "TRAINING COMPLETED";
    if (isSubmitted) return "MODEL UPDATE SENT";
    return "READY FOR TRAINING";
  };

  const currentState = getLifecycleState();
  const metricsHistory: TrainingMetricPoint[] = hospital.training_metrics || [];

  return (
    <div className="p-8 space-y-6 text-slate-200 max-w-7xl mx-auto">
      {/* Top Bar with Navigation */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-navy-700 pb-4">
        <div className="flex items-center space-x-3">
          <button
            onClick={onBack}
            className="p-2 rounded-lg bg-navy-800 hover:bg-navy-700 text-slate-400 hover:text-white transition"
            title="Return to Registry"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold text-white tracking-wide">{hospital.name}</h1>
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase font-mono border ${
                hospital.type === "docker"
                  ? "bg-blue-500/20 text-blue-300 border-blue-500/40"
                  : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
              }`}>
                {hospital.type === "docker" ? "Docker Hospital Node" : "Standby Hospital Node"}
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Node ID: {hospital.hospital_id} • Endpoint: {hospital.endpoint} • gRPC: :{hospital.grpc_port}
            </p>
          </div>
        </div>

        {/* Global Node State Badge */}
        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-400 font-semibold uppercase">Status:</span>
          <span className="px-3 py-1 rounded-full text-xs font-bold font-mono bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
            {currentState}
          </span>
        </div>
      </div>

      {/* Notification banner */}
      {statusMessage && (
        <div className="p-3 bg-blue-950/40 border border-blue-500/50 rounded-lg text-xs text-blue-200 flex items-center justify-between">
          <span>{statusMessage}</span>
          <button onClick={() => setStatusMessage(null)} className="text-blue-400 hover:text-white text-xs">
            Dismiss
          </button>
        </div>
      )}

      {/* 12-Step Lifecycle Progression Flow */}
      <div className="bg-navy-800 border border-navy-700 rounded-xl p-4 shadow-sm">
        <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2.5 flex items-center justify-between">
          <span>Hospital Operational Workflow Progression</span>
          <span className="font-mono text-cyan-400 font-bold">{currentState}</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-[10px] font-mono text-center">
          <div className={`p-2 rounded border ${hasData ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" : "bg-navy-900 border-navy-700 text-slate-500"}`}>
            1. DATA CONFIGURED
          </div>
          <div className={`p-2 rounded border ${isModelReady ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" : "bg-navy-900 border-navy-700 text-slate-500"}`}>
            2. MODEL INITIALIZED
          </div>
          <div className={`p-2 rounded border ${isApproved ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" : "bg-navy-900 border-navy-700 text-slate-500"}`}>
            3. HUB APPROVED
          </div>
          <div className={`p-2 rounded border ${loadingAction === "training" ? "bg-blue-500/30 text-blue-300 border-blue-400 animate-pulse" : (isTrained || isSubmitted) ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" : "bg-navy-900 border-navy-700 text-slate-500"}`}>
            4. LOCAL TRAINING
          </div>
          <div className={`p-2 rounded border ${isTrained ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" : isSubmitted ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" : "bg-navy-900 border-navy-700 text-slate-500"}`}>
            5. UPDATE READY
          </div>
          <div className={`p-2 rounded border ${isSubmitted ? "bg-purple-500/20 text-purple-300 border-purple-500/40" : "bg-navy-900 border-navy-700 text-slate-500"}`}>
            6. SUBMITTED TO HUB
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Data & Model Setup */}
        <div className="space-y-6">
          {/* Section A: Local Medical Data */}
          <div className="bg-navy-800 border border-navy-700 rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex justify-between items-center">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center">
                <HardDrive className="w-4 h-4 mr-2 text-blue-400" />
                Local Medical Dataset
              </h2>
              {hasData && (
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-navy-900 text-emerald-300 border border-emerald-500/40">
                  {hospital.sample_count} Volumes Configured
                </span>
              )}
            </div>

            <div className="bg-red-950/20 border border-red-500/30 rounded-lg p-2.5 text-[11px] text-red-200">
              <strong>STRICT PRIVACY GUARANTEE:</strong> Raw patient MRI scans remain strictly on-premise in this hospital environment. The central server never receives raw patient data.
            </div>

            <div className="space-y-3">
              {/* Option A: Connect Existing Dataset */}
              <div className="bg-navy-900 p-3.5 rounded-lg border border-navy-700 space-y-2">
                <h3 className="font-bold text-white text-xs flex items-center">
                  <FolderOpen className="w-3.5 h-3.5 mr-1.5 text-blue-400" />
                  Option A: Connect Existing Dataset
                </h3>
                <p className="text-slate-400 text-[11px]">
                  Specify path to an on-premise directory of 3D NIfTI (.nii.gz) scans:
                </p>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    placeholder="e.g. C:/HospitalData/BraTS2021"
                    value={localPathInput}
                    onChange={(e) => setLocalPathInput(e.target.value)}
                    className="flex-1 bg-navy-800 border border-navy-600 rounded px-2.5 py-1.5 text-white font-mono text-xs focus:outline-none"
                  />
                  <button
                    onClick={handleConfigureLocalData}
                    disabled={loadingAction === "data-local" || !localPathInput.trim()}
                    className="bg-navy-700 hover:bg-navy-600 text-white font-semibold px-3 py-1.5 rounded text-xs transition"
                  >
                    {loadingAction === "data-local" ? "Loading..." : "Connect"}
                  </button>
                </div>
              </div>

              {/* Option B: Create Demo Synthetic Dataset */}
              <div className="bg-navy-900 p-3.5 rounded-lg border border-navy-700 flex justify-between items-center">
                <div>
                  <h3 className="font-bold text-white text-xs flex items-center">
                    <Database className="w-3.5 h-3.5 mr-1.5 text-purple-400" />
                    Option B: Create Demo Dataset
                  </h3>
                  <p className="text-slate-400 text-[11px]">
                    Generate an isolated synthetic 3D MRI dataset for testing.
                  </p>
                </div>
                <button
                  onClick={handleConfigureDemoData}
                  disabled={loadingAction === "data-demo"}
                  className="bg-blue-600 hover:bg-blue-500 text-white font-semibold px-3 py-1.5 rounded text-xs transition shadow-sm"
                >
                  {loadingAction === "data-demo" ? "Generating..." : "Generate Demo"}
                </button>
              </div>
            </div>
          </div>

          {/* Section B: Global Model Synchronization & Approval */}
          <div className="bg-navy-800 border border-navy-700 rounded-xl p-5 shadow-sm space-y-4">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center">
              <Database className="w-4 h-4 mr-2 text-purple-400" />
              Global Model Synchronization & Node Approval
            </h2>

            <div className="grid grid-cols-2 gap-3 text-center">
              <div className="bg-navy-900 p-3 rounded-lg border border-navy-700">
                <span className="text-slate-400 text-[11px] block">Model State</span>
                <span className={`text-xs font-bold font-mono mt-1 block ${isModelReady ? "text-emerald-400" : "text-slate-400"}`}>
                  {hospital.model_status}
                </span>
              </div>
              <div className="bg-navy-900 p-3 rounded-lg border border-navy-700">
                <span className="text-slate-400 text-[11px] block">Approval State</span>
                <span className={`text-xs font-bold font-mono mt-1 block ${isApproved ? "text-emerald-400" : "text-amber-400"}`}>
                  {hospital.approval_status}
                </span>
              </div>
            </div>

            <div className="flex space-x-3 pt-1">
              {!isModelReady ? (
                <button
                  onClick={handleInitModel}
                  disabled={loadingAction === "init-model" || !hasData}
                  className={`flex-1 py-2 rounded-lg font-bold text-xs transition shadow-sm ${
                    hasData
                      ? "bg-purple-600 hover:bg-purple-500 text-white"
                      : "bg-navy-700 text-slate-500 cursor-not-allowed"
                  }`}
                >
                  {loadingAction === "init-model" ? "Instantiating..." : "Instantiate Global Model Locally"}
                </button>
              ) : (
                <div className="flex-1 p-2 rounded-lg bg-navy-900 border border-navy-700 text-center text-xs text-emerald-400 font-semibold flex items-center justify-center space-x-1">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Model Synchronized (v{hospital.current_round})</span>
                </div>
              )}

              {!isApproved ? (
                <button
                  onClick={handleRequestApproval}
                  disabled={loadingAction === "req-approval"}
                  className="bg-amber-600 hover:bg-amber-500 text-white font-bold px-4 py-2 rounded-lg text-xs transition"
                >
                  {loadingAction === "req-approval" ? "Requesting..." : "Request Approval"}
                </button>
              ) : (
                <div className="px-3 py-2 rounded-lg bg-navy-900 border border-navy-700 text-center text-xs text-emerald-400 font-semibold flex items-center space-x-1">
                  <ShieldCheck className="w-4 h-4" />
                  <span>Approved Node</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Training Execution & Telemetry Graph */}
        <div className="space-y-6">
          {/* Training Execution Card */}
          <div className="bg-navy-800 border border-navy-700 rounded-xl p-5 shadow-sm space-y-4">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center">
              <BrainCircuit className="w-4 h-4 mr-2 text-emerald-400" />
              Real PyTorch / MONAI Local Training
            </h2>

            <div className="flex items-center space-x-4 bg-navy-900 p-3 rounded-lg border border-navy-700">
              <div className="flex items-center space-x-2">
                <span className="text-xs text-slate-400 font-semibold">Local Epochs:</span>
                <select
                  value={targetEpochs}
                  onChange={(e) => setTargetEpochs(Number(e.target.value))}
                  className="bg-navy-800 border border-navy-600 rounded px-2 py-1 text-white text-xs font-mono focus:outline-none"
                >
                  <option value={1}>1 Epoch</option>
                  <option value={2}>2 Epochs</option>
                  <option value={3}>3 Epochs</option>
                  <option value={5}>5 Epochs</option>
                </select>
              </div>

              <div className="flex items-center space-x-2 text-xs text-slate-400">
                <span>Architecture:</span>
                <strong className="text-slate-200">3D U-Net (MONAI)</strong>
              </div>
            </div>

            <button
              onClick={handleStartTraining}
              disabled={loadingAction === "training" || !isApproved || !isModelReady}
              className={`w-full py-3 rounded-xl font-bold text-xs flex items-center justify-center space-x-2 transition shadow-sm ${
                isApproved && isModelReady
                  ? "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/30"
                  : "bg-navy-700 text-slate-500 cursor-not-allowed"
              }`}
            >
              {loadingAction === "training" ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Executing PyTorch Local Training ({targetEpochs} epoch(s))...</span>
                </>
              ) : (
                <>
                  <BrainCircuit className="w-4 h-4" />
                  <span>Start Local Training</span>
                </>
              )}
            </button>

            {!isApproved && (
              <p className="text-[11px] text-amber-400 text-center">
                * Participation requires Central Administrator approval before training begins.
              </p>
            )}

            {/* Model Submission Section */}
            {isTrained && (
              <div className="p-4 bg-navy-900 rounded-lg border border-navy-700 space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-white">Trained Weights Ready</span>
                  <span className="font-mono text-emerald-400">
                    Loss: {hospital.current_loss} • Dice: {hospital.current_dice}
                  </span>
                </div>
                <button
                  onClick={handleSubmitModel}
                  disabled={loadingAction === "submit-model"}
                  className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-2.5 rounded-lg text-xs transition shadow-sm flex items-center justify-center space-x-2"
                >
                  <Upload className="w-4 h-4" />
                  <span>{loadingAction === "submit-model" ? "Submitting..." : "Send Model Update to Central Hub"}</span>
                </button>
              </div>
            )}

            {isSubmitted && (
              <div className="p-3 bg-purple-950/20 border border-purple-500/30 rounded-lg text-center space-y-1">
                <p className="text-xs font-bold text-purple-300">Model Update Transmitted</p>
                <p className="text-[11px] text-slate-400">
                  Permitted model weights received by Central Hub. Awaiting "Federation of Models" aggregation round.
                </p>
              </div>
            )}
          </div>

          {/* Real Local Training Telemetry Graph */}
          <div className="bg-navy-800 border border-navy-700 rounded-xl p-5 shadow-sm space-y-3">
            <div className="flex justify-between items-center">
              <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center">
                <LineChartIcon className="w-3.5 h-3.5 mr-1.5 text-cyan-400" />
                Local Training Convergence Telemetry
              </h3>
              {metricsHistory.length > 0 && (
                <span className="text-[10px] text-slate-400 font-mono">
                  {metricsHistory.length} Epoch(s) Logged
                </span>
              )}
            </div>

            {metricsHistory.length === 0 ? (
              <div className="p-8 text-center text-slate-500 text-xs space-y-1">
                <Activity className="w-6 h-6 mx-auto opacity-30" />
                <p>No training metrics recorded yet.</p>
                <p className="text-[10px]">Execute local training to observe genuine loss convergence and Dice score progression.</p>
              </div>
            ) : (
              <div className="w-full h-56 pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="epoch"
                      stroke="#94a3b8"
                      tick={{ fontSize: 10 }}
                      label={{ value: "Local Epoch", position: "insideBottom", offset: -5, fontSize: 10, fill: "#94a3b8" }}
                    />
                    <YAxis
                      stroke="#94a3b8"
                      domain={[0, 1]}
                      tick={{ fontSize: 10 }}
                    />
                    <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", fontSize: 11 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line
                      type="monotone"
                      dataKey="train_loss"
                      name="Train Loss"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot
                    />
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      name="Val Loss"
                      stroke="#f97316"
                      strokeWidth={2}
                      dot
                    />
                    <Line
                      type="monotone"
                      dataKey="dice"
                      name="Dice Score"
                      stroke="#22c55e"
                      strokeWidth={2}
                      dot
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
