import React, { useEffect, useState } from "react";
import { 
  Activity, 
  Server, 
  Users, 
  Database, 
  ShieldCheck, 
  Plus, 
  ArrowRight, 
  Clock, 
  LineChart as LineChartIcon,
  TrendingUp,
  Award
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
import type { ViewState } from "../App";

interface CentralDashboardProps {
  onNavigate?: (view: ViewState) => void;
}

export const CentralDashboard: React.FC<CentralDashboardProps> = ({ onNavigate }) => {
  const [metrics, setMetrics] = useState<any>({ round: 0, status: "FRESH", loss: 0, dice: 0 });
  const [health, setHealth] = useState<any>({ status: "healthy", nodes_connected: 0 });
  const [hospitals, setHospitals] = useState<any[]>([]);
  const [incomingCount, setIncomingCount] = useState<number>(0);
  const [recentLogs, setRecentLogs] = useState<any[]>([]);
  const [globalModel, setGlobalModel] = useState<any>({ current_version: 0, history: [] });

  const loadDashboardData = () => {
    fetch("http://127.0.0.1:8000/api/metrics")
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => {});

    fetch("http://127.0.0.1:8000/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => {});

    fetch("http://127.0.0.1:8000/api/hospitals")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setHospitals(data);
      })
      .catch(() => {});

    fetch("http://127.0.0.1:8000/api/federation/incoming-models")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setIncomingCount(data.length);
      })
      .catch(() => {});

    fetch("http://127.0.0.1:8000/api/federation/global-model")
      .then((r) => r.json())
      .then((data) => {
        if (data) setGlobalModel(data);
      })
      .catch(() => {});

    fetch("http://127.0.0.1:8000/api/activity/logs")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setRecentLogs(data.slice(0, 4));
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadDashboardData();
    const interval = setInterval(loadDashboardData, 3000);
    return () => clearInterval(interval);
  }, []);

  const isFresh = !metrics.round || metrics.round === 0 || metrics.status === "FRESH";
  const approvedCount = hospitals.filter((h) => h.approval_status === "APPROVED").length;
  const trainingCount = hospitals.filter((h) => h.participation_status === "TRAINING" || h.status === "training").length;

  // Process history from round 1 to latest round for learning curve
  const learningCurveData = (globalModel.history || [])
    .slice()
    .reverse()
    .map((h: any) => ({
      round: `R${h.round}`,
      roundNum: h.round,
      dice: h.metrics?.dice ?? null,
      loss: h.metrics?.loss ?? null,
      hospitals: Array.isArray(h.participating_hospitals) ? h.participating_hospitals.join(", ") : "",
      strategy: h.strategy || "FedAvg",
    }));

  const bestDice = learningCurveData.reduce((max: number, d: any) => (d.dice && d.dice > max ? d.dice : max), 0);

  return (
    <div className="p-8 space-y-8 text-slate-200">
      {/* Top Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-wide">Central Dashboard</h2>
          <p className="text-sm text-slate-400">
            Real-time federation overview and state-driven control hub
          </p>
        </div>
        <div className="flex items-center space-x-2 text-xs bg-navy-800 px-4 py-2 rounded-full border border-navy-700">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-semibold text-slate-300 uppercase">Hub {health.status}</span>
        </div>
      </div>

      {/* Global Model Status Banner */}
      <div className="bg-navy-800 border border-navy-700 rounded-xl p-6 shadow-sm">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
            Global Model Status
          </h3>
          {metrics.last_updated && (
            <span className="text-xs text-slate-400 flex items-center font-mono">
              <Clock className="w-3.5 h-3.5 mr-1 text-blue-400" />
              Last updated: {new Date(metrics.last_updated).toLocaleString()}
            </span>
          )}
        </div>

        <div
          className={`p-6 rounded-xl border flex flex-col items-center justify-center space-y-2 text-center transition-all ${
            isFresh
              ? "bg-navy-900 border-blue-500/30 text-blue-200"
              : "bg-emerald-950/20 border-emerald-500/40 text-emerald-200"
          }`}
        >
          <Database className={`w-10 h-10 mb-1 ${isFresh ? "text-blue-400" : "text-emerald-400"}`} />
          <p className="text-2xl font-extrabold tracking-wide uppercase">
            {isFresh
              ? "FRESH / NEVER TRAINED"
              : `FEDERATED MODEL — VERSION ${metrics.round}`}
          </p>
          <p className="text-xs text-slate-400 font-mono">
            {isFresh
              ? "No federation rounds executed yet. Ready to initialize with registered hospitals."
              : `Current Round: ${metrics.round} • Strategy: ${metrics.active_strategy || "FedAvg"} • Loss: ${metrics.loss || "--"} • Dice: ${metrics.dice || "--"}`}
          </p>
        </div>
      </div>

      {/* Key State Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-navy-800 border border-navy-700 rounded-xl p-5 shadow-sm">
          <div className="flex items-center space-x-2 text-slate-400 text-xs uppercase font-semibold">
            <Users className="w-4 h-4 text-blue-400" />
            <span>Registered Hospitals</span>
          </div>
          <p className="text-3xl font-extrabold text-white mt-2">{hospitals.length}</p>
          <p className="text-xs text-slate-500 mt-1">
            {approvedCount} approved for federation
          </p>
        </div>

        <div className="bg-navy-800 border border-navy-700 rounded-xl p-5 shadow-sm">
          <div className="flex items-center space-x-2 text-slate-400 text-xs uppercase font-semibold">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Approved Quorum</span>
          </div>
          <p className="text-3xl font-extrabold text-white mt-2">{approvedCount}</p>
          <p className="text-xs text-slate-500 mt-1">
            {hospitals.length - approvedCount} pending authorization
          </p>
        </div>

        <div className="bg-navy-800 border border-navy-700 rounded-xl p-5 shadow-sm">
          <div className="flex items-center space-x-2 text-slate-400 text-xs uppercase font-semibold">
            <Activity className="w-4 h-4 text-amber-400" />
            <span>Local Training Active</span>
          </div>
          <p className="text-3xl font-extrabold text-white mt-2">{trainingCount}</p>
          <p className="text-xs text-slate-500 mt-1">
            Hospital nodes currently in epoch loop
          </p>
        </div>

        <div className="bg-navy-800 border border-navy-700 rounded-xl p-5 shadow-sm">
          <div className="flex items-center space-x-2 text-slate-400 text-xs uppercase font-semibold">
            <Server className="w-4 h-4 text-purple-400" />
            <span>Incoming Model Updates</span>
          </div>
          <p className="text-3xl font-extrabold text-white mt-2">{incomingCount}</p>
          <p className="text-xs text-slate-500 mt-1">
            Ready for aggregation in Federation
          </p>
        </div>
      </div>

      {/* GLOBAL MODEL LEARNING CURVE & CONVERGENCE TELEMETRY GRAPH */}
      <div className="bg-navy-800 border border-navy-700 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center">
              <LineChartIcon className="w-4 h-4 mr-2 text-cyan-400" />
              Global Model Learning Curve & Convergence Telemetry
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Tracks actual aggregated Dice score and validation loss across all completed federation rounds
            </p>
          </div>

          {learningCurveData.length > 0 && (
            <div className="flex items-center space-x-3 text-xs font-mono">
              <div className="bg-navy-900 border border-navy-700 px-3 py-1 rounded-lg text-emerald-400 flex items-center space-x-1.5">
                <Award className="w-3.5 h-3.5" />
                <span>Peak Dice: <strong>{bestDice.toFixed(4)}</strong></span>
              </div>
              <div className="bg-navy-900 border border-navy-700 px-3 py-1 rounded-lg text-blue-300">
                <span>Rounds: <strong>{learningCurveData.length}</strong></span>
              </div>
            </div>
          )}
        </div>

        {learningCurveData.length === 0 ? (
          <div className="p-10 text-center text-slate-500 text-xs space-y-2 border border-dashed border-navy-700 rounded-xl bg-navy-900/50">
            <TrendingUp className="w-8 h-8 mx-auto opacity-30 text-slate-400" />
            <h4 className="font-semibold text-slate-300">No Federation Rounds Completed Yet</h4>
            <p className="text-[11px] max-w-sm mx-auto">
              The global model learning curve will appear automatically once the first federation round is aggregated.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="w-full h-72 pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={learningCurveData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="round"
                    stroke="#94a3b8"
                    tick={{ fontSize: 11 }}
                    label={{ value: "Federation Round", position: "insideBottom", offset: -5, fontSize: 11, fill: "#94a3b8" }}
                  />
                  <YAxis
                    stroke="#94a3b8"
                    domain={[0, 1]}
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", fontSize: 12 }}
                    formatter={(val: any, name: any) => [typeof val === "number" ? val.toFixed(4) : val, name]}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey="dice"
                    name="Global Dice Score"
                    stroke="#22c55e"
                    strokeWidth={2.5}
                    dot={{ r: 4, fill: "#22c55e" }}
                    activeDot={{ r: 6 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    name="Global Loss"
                    stroke="#ef4444"
                    strokeWidth={2}
                    dot={{ r: 4, fill: "#ef4444" }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Quick Metrics Breakdown Strip */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
              <div className="p-3 bg-navy-900 border border-navy-700/70 rounded-lg text-xs space-y-1">
                <span className="text-slate-400 text-[11px]">Latest Global Dice Score</span>
                <p className="text-base font-bold text-emerald-400 font-mono">
                  {learningCurveData[learningCurveData.length - 1]?.dice?.toFixed(4) ?? "--"}
                </p>
                <p className="text-[10px] text-slate-500">Sørensen-Dice 3D overlap metric (0 - 1)</p>
              </div>

              <div className="p-3 bg-navy-900 border border-navy-700/70 rounded-lg text-xs space-y-1">
                <span className="text-slate-400 text-[11px]">Latest Aggregated Loss</span>
                <p className="text-base font-bold text-rose-400 font-mono">
                  {learningCurveData[learningCurveData.length - 1]?.loss?.toFixed(4) ?? "--"}
                </p>
                <p className="text-[10px] text-slate-500">Dice + Focal Cross-Entropy Loss</p>
              </div>

              <div className="p-3 bg-navy-900 border border-navy-700/70 rounded-lg text-xs space-y-1">
                <span className="text-slate-400 text-[11px]">Active Aggregation Strategy</span>
                <p className="text-base font-bold text-blue-300 font-mono">
                  {learningCurveData[learningCurveData.length - 1]?.strategy ?? "FedAvg"}
                </p>
                <p className="text-[10px] text-slate-500">Federated Averaging weighted by samples</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Empty State Banner if 0 hospitals */}
      {hospitals.length === 0 && (
        <div className="bg-navy-900 border border-dashed border-navy-600 rounded-xl p-8 text-center space-y-4">
          <div className="w-12 h-12 bg-navy-800 rounded-full flex items-center justify-center mx-auto text-blue-400 border border-navy-700">
            <Users className="w-6 h-6" />
          </div>
          <h4 className="text-lg font-bold text-white">No Hospitals Registered</h4>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            The federation is currently empty. Register a real remote hospital or initiate a simulated Docker container to begin.
          </p>
          {onNavigate && (
            <div className="flex justify-center space-x-3 pt-2">
              <button
                onClick={() => onNavigate("registry")}
                className="bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2 rounded-lg transition flex items-center space-x-1.5"
              >
                <Plus className="w-4 h-4" />
                <span>Register Remote Hospital</span>
              </button>
              <button
                onClick={() => onNavigate("simulate_remote")}
                className="bg-navy-700 hover:bg-navy-600 text-slate-200 text-xs font-semibold px-4 py-2 rounded-lg transition border border-navy-600 flex items-center space-x-1.5"
              >
                <span>Simulate Remote Hospital</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      )}

      {/* Recent Activity Log Preview */}
      <div className="bg-navy-800 border border-navy-700 rounded-xl p-6 shadow-sm">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
            Recent Federation Activity
          </h3>
          {onNavigate && (
            <button
              onClick={() => onNavigate("activity")}
              className="text-xs text-blue-400 hover:text-blue-300 font-semibold flex items-center space-x-1"
            >
              <span>View All Logs</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {recentLogs.length === 0 ? (
          <div className="text-center py-6 text-xs text-slate-500 italic">
            No hospital activity recorded yet.
          </div>
        ) : (
          <div className="space-y-2.5">
            {recentLogs.map((log) => (
              <div
                key={log.id}
                className="bg-navy-900 border border-navy-700/60 rounded-lg p-3 flex justify-between items-center text-xs"
              >
                <div className="flex items-center space-x-3 truncate mr-4">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-navy-800 text-blue-300 border border-navy-700 uppercase flex-shrink-0">
                    {log.event_type.replace(/_/g, " ")}
                  </span>
                  <span className="text-slate-200 truncate">{log.description}</span>
                </div>
                <span className="text-slate-500 font-mono text-[11px] flex-shrink-0">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
