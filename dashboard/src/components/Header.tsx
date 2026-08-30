
import { Shield, RefreshCw } from "lucide-react";

export const Header: React.FC = () => {
  return (
    <header className="h-16 bg-slate-900/60 backdrop-blur border-b border-slate-800 px-6 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          Strategy: FedAvg + SecAgg
        </span>

        <span className="text-slate-400 text-sm">
          Round: 0 / 20
        </span>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-xs text-slate-300 bg-slate-800 px-3 py-1.5 rounded-md border border-slate-700">
          <Shield className="w-3.5 h-3.5 text-emerald-400" />
          <span>mTLS Encrypted</span>
        </div>

        <button className="text-slate-400 hover:text-white transition-colors">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
};