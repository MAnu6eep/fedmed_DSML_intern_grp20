import React, { useEffect, useState } from "react";
import { Database, Clock, HardDrive, FileCheck } from "lucide-react";

export const GlobalModelRegistry: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<any>({
    current_version: 0,
    status: "FRESH",
    last_updated: null,
    history: [],
  });

  const loadData = () => {
    fetch("http://127.0.0.1:8000/api/federation/global-model")
      .then((r) => r.json())
      .then(setModelInfo)
      .catch(() => {});
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 3000);
    return () => clearInterval(interval);
  }, []);

  const history = modelInfo.history || [];
  const isFresh = modelInfo.current_version === 0;

  return (
    <div className="p-8 space-y-8 text-slate-200">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-wide">Global Model Registry</h2>
          <p className="text-sm text-slate-400">
            Persistent repository of central 3D U-Net checkpoints, version lineage, and benchmark evaluation history
          </p>
        </div>

        <div className="bg-navy-800 border border-navy-700 px-4 py-2 rounded-lg text-xs font-mono">
          <span className="text-slate-400 mr-2">Current Active:</span>
          <span className={`font-bold ${isFresh ? "text-blue-300" : "text-emerald-400"}`}>
            {isFresh ? "FRESH / NEVER TRAINED" : `v${modelInfo.current_version}`}
          </span>
        </div>
      </div>

      {/* Model History List or Empty State */}
      <div className="space-y-4">
        {history.length === 0 ? (
          <div className="bg-navy-800 border border-navy-700 rounded-xl p-12 text-center space-y-3 shadow-sm">
            <div className="w-14 h-14 bg-navy-900 border border-navy-700 rounded-full flex items-center justify-center mx-auto text-blue-400">
              <Database className="w-7 h-7" />
            </div>
            <h3 className="text-lg font-bold text-white">Global Model — FRESH</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              No previous federated model versions recorded. The global model is in its initial un-trained state and will increment to v1 upon the first aggregation round.
            </p>
          </div>
        ) : (
          history.map((entry: any) => (
            <div
              key={entry.version}
              className="bg-navy-800 border border-navy-700 rounded-xl p-6 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-sm hover:border-navy-600 transition"
            >
              <div className="flex items-start space-x-5">
                <div className="bg-navy-900 p-3.5 rounded-xl border border-navy-700 flex-shrink-0">
                  <Database
                    className={`w-7 h-7 ${
                      entry.status === "Active" ? "text-emerald-400" : "text-blue-400"
                    }`}
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex items-center space-x-3">
                    <h3 className="text-lg font-bold text-white">
                      Global Model v{entry.version}
                    </h3>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                        entry.status === "Active"
                          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                          : "bg-slate-700/40 text-slate-400"
                      }`}
                    >
                      {entry.status}
                    </span>
                  </div>

                  <p className="text-xs text-slate-400 font-mono">
                    Round: {entry.round} • Strategy: {entry.strategy} • Participants:{" "}
                    {entry.participating_hospitals?.join(", ") || "None"}
                  </p>

                  <div className="flex items-center space-x-4 text-xs font-mono pt-1 text-slate-300">
                    <span>
                      Avg Loss: <strong className="text-white">{entry.metrics?.loss}</strong>
                    </span>
                    <span>•</span>
                    <span>
                      Validation Dice:{" "}
                      <strong className="text-emerald-300">{entry.metrics?.dice}</strong>
                    </span>
                  </div>

                  {entry.checkpoint_path && (
                    <div className="flex items-center space-x-1.5 text-[11px] font-mono text-slate-500 pt-1">
                      <HardDrive className="w-3.5 h-3.5" />
                      <span className="truncate max-w-md">{entry.checkpoint_path}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="text-right text-xs font-mono text-slate-500 flex md:flex-col justify-between items-end">
                <span className="flex items-center">
                  <Clock className="w-3.5 h-3.5 mr-1" />
                  {new Date(entry.created_at).toLocaleString()}
                </span>
                <span className="mt-1 flex items-center text-blue-400">
                  <FileCheck className="w-3.5 h-3.5 mr-1" /> Persisted to Disk
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
