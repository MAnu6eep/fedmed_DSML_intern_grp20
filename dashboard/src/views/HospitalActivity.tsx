import React, { useEffect, useState } from "react";
import { History, Clock, FileText, CheckCircle2, ShieldCheck, Database, BrainCircuit, Upload, Server } from "lucide-react";

interface ActivityLog {
  id: string;
  timestamp: string;
  event_type: string;
  hospital_id: string | null;
  hospital_name: string | null;
  description: string;
  details: Record<string, any>;
}

export const HospitalActivity: React.FC = () => {
  const [logs, setLogs] = useState<ActivityLog[]>([]);

  const loadLogs = () => {
    fetch("http://127.0.0.1:8000/api/activity/logs")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setLogs(data);
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadLogs();
    const interval = setInterval(loadLogs, 2500);
    return () => clearInterval(interval);
  }, []);

  const getEventIcon = (type: string) => {
    switch (type) {
      case "HOSPITAL_REGISTERED":
      case "HOSPITAL_CONNECTED":
        return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
      case "HOSPITAL_APPROVED":
        return <ShieldCheck className="w-4 h-4 text-emerald-400" />;
      case "DATASET_CONFIGURED":
      case "MODEL_INITIALIZED":
        return <Database className="w-4 h-4 text-blue-400" />;
      case "TRAINING_STARTED":
      case "TRAINING_COMPLETED":
        return <BrainCircuit className="w-4 h-4 text-amber-400" />;
      case "MODEL_SUBMITTED":
        return <Upload className="w-4 h-4 text-purple-400" />;
      case "FEDERATION_COMPLETED":
      case "GLOBAL_MODEL_UPDATED":
        return <Server className="w-4 h-4 text-emerald-400" />;
      default:
        return <FileText className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div className="p-8 space-y-8 text-slate-200">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-wide">Hospital Activity</h2>
          <p className="text-sm text-slate-400">
            Persistent audit trail tracking the chronological lifecycle of hospital nodes and federation events
          </p>
        </div>

        <div className="bg-navy-800 border border-navy-700 px-4 py-2 rounded-lg text-xs font-mono">
          <span className="text-slate-400 mr-2">Total Events:</span>
          <span className="font-bold text-white">{logs.length}</span>
        </div>
      </div>

      {/* Activity Timeline / Empty State */}
      <div className="bg-navy-800 border border-navy-700 rounded-xl overflow-hidden shadow-sm">
        {logs.length === 0 ? (
          <div className="p-12 text-center space-y-3">
            <div className="w-14 h-14 bg-navy-900 border border-navy-700 rounded-full flex items-center justify-center mx-auto text-slate-500">
              <History className="w-7 h-7" />
            </div>
            <h3 className="text-lg font-bold text-white">No Hospital Activity Recorded</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              No historical events recorded yet. As hospitals are registered, authenticated, train models, and aggregate weights, their real lifecycle events will be permanently tracked here.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-navy-700/60">
            {logs.map((log) => (
              <div
                key={log.id}
                className="p-5 flex items-start space-x-4 hover:bg-navy-700/20 transition-colors text-xs"
              >
                <div className="p-2 rounded-lg bg-navy-900 border border-navy-700/70 flex-shrink-0 mt-0.5">
                  {getEventIcon(log.event_type)}
                </div>

                <div className="flex-1 space-y-1">
                  <div className="flex items-center space-x-3">
                    <span className="font-bold text-white text-sm">
                      {log.description}
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-navy-900 text-blue-300 border border-navy-700">
                      {log.event_type.replace(/_/g, " ")}
                    </span>
                  </div>

                  {log.hospital_name && (
                    <div className="text-[11px] text-slate-400 font-mono">
                      Target Node: <strong className="text-slate-300">{log.hospital_name}</strong>{" "}
                      ({log.hospital_id})
                    </div>
                  )}

                  {log.details && Object.keys(log.details).length > 0 && (
                    <div className="pt-1">
                      <pre className="text-[11px] font-mono bg-navy-900/80 border border-navy-700/60 p-2 rounded text-slate-300 overflow-x-auto">
                        {JSON.stringify(log.details, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>

                <div className="text-right text-[11px] font-mono text-slate-500 flex items-center space-x-1 flex-shrink-0">
                  <Clock className="w-3.5 h-3.5" />
                  <span>{new Date(log.timestamp).toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
