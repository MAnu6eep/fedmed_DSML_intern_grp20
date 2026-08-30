
import {
  Activity,
  Database,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

interface NodeStatusCardProps {
  id: string;
  name: string;
  port: number;
  sampleCount: number;
  status: "online" | "training" | "offline";
}

export const NodeStatusCard: React.FC<NodeStatusCardProps> = ({
  name,
  port,
  sampleCount,
  status,
}) => {
  const isOnline = status !== "offline";

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-medium text-white text-base">
            {name}
          </h3>

          <p className="text-xs text-slate-400">
            Port: {port} • gRPC
          </p>
        </div>

        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
            isOnline
              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
              : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
          }`}
        >
          {isOnline ? (
            <CheckCircle2 className="w-3 h-3" />
          ) : (
            <AlertCircle className="w-3 h-3" />
          )}

          {status}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-800/80 text-xs">
        <div className="flex items-center gap-2 text-slate-300">
          <Database className="w-3.5 h-3.5 text-slate-500" />
          <span>{sampleCount} Samples</span>
        </div>

        <div className="flex items-center gap-2 text-slate-300">
          <Activity className="w-3.5 h-3.5 text-emerald-500" />
          <span>Healthy</span>
        </div>
      </div>
    </div>
  );
};