import {
    ShieldCheck,
    HardDrive,
    Cpu,
    AlertTriangle,
    CheckCircle2,
    RefreshCw,
  } from 'lucide-react';
  
  import type { HospitalTelemetry } from '../types/telemetry';
  interface HospitalNodeCardProps {
    node: HospitalTelemetry;
    onSimulateDropout?: (nodeId: string) => void;
  }
  
  export const HospitalNodeCard = ({
    node,
    onSimulateDropout,
  }: HospitalNodeCardProps) => {
    const getStatusBadge = (status: HospitalTelemetry['status']) => {
      switch (status) {
        case 'training':
          return (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <RefreshCw className="w-3 h-3 animate-spin" />
              Training
            </span>
          );
  
        case 'online':
          return (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <CheckCircle2 className="w-3 h-3" />
              Online
            </span>
          );
  
        case 'offline':
          return (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">
              <AlertTriangle className="w-3 h-3" />
              Offline
            </span>
          );
  
        default:
          return (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
              Error
            </span>
          );
      }
    };
  
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm transition-all hover:border-slate-700">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="font-semibold text-white text-base tracking-wide">
              {node.name}
            </h3>
  
            <p className="text-xs text-slate-400 mt-0.5 font-mono">
              {node.host}:{node.port} • gRPC: {node.grpcPort}
            </p>
          </div>
  
          {getStatusBadge(node.status)}
        </div>
  
        <div className="grid grid-cols-2 gap-3 py-3 border-y border-slate-800/80 text-xs">
          <div className="flex items-center gap-2 text-slate-300">
            <HardDrive className="w-3.5 h-3.5 text-slate-500" />
            <span>{node.sampleCount} MRI Volumes</span>
          </div>
  
          <div className="flex items-center gap-2 text-slate-300">
            <ShieldCheck
              className={`w-3.5 h-3.5 ${
                node.isSecAggActive
                  ? 'text-emerald-400'
                  : 'text-slate-600'
              }`}
            />
            <span>
              {node.isSecAggActive ? 'SecAgg+ Active' : 'Plaintext'}
            </span>
          </div>
  
          <div className="flex items-center gap-2 text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-blue-400" />
            <span>Loss: {node.localLoss.toFixed(4)}</span>
          </div>
  
          <div className="flex items-center gap-2 text-slate-300">
            <span className="font-semibold text-emerald-400">
              Dice:
            </span>
            <span>{(node.localDice * 100).toFixed(1)}%</span>
          </div>
        </div>
  
        <div className="mt-4 flex items-center justify-between">
          <span className="text-[11px] text-slate-400 font-mono">
            Round {node.currentRound}
          </span>
  
          {onSimulateDropout && (
            <button
              onClick={() => onSimulateDropout(node.id)}
              className="text-xs text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 px-2.5 py-1 rounded transition-colors border border-rose-500/20"
            >
              Simulate Drop
            </button>
          )}
        </div>
      </div>
    );
  };