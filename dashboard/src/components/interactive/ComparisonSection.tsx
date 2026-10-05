import React from "react";
import { BarChart3, FileText, CheckCircle2 } from "lucide-react";
import { REAL_BENCHMARK_RESULTS } from "../../data/demoData";

export const ComparisonSection: React.FC = () => {
  return (
    <section id="comparison" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <BarChart3 className="w-3.5 h-3.5" />
            SECTION 11 — BENCHMARK COMPARISON
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Empirical Benchmark: Plaintext FedAvg vs. SecAgg+ / CKKS
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Exact empirical execution metrics extracted from repository benchmark runs (<code className="text-cyan-400">docs/secagg_benchmark_results.json</code>).
          </p>
        </div>

        {/* REAL REPO BENCHMARK CARD */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-2 text-sm text-white font-bold">
              <FileText className="w-4 h-4 text-cyan-400" />
              <span>Repository Benchmark Execution Logs</span>
            </div>
            <span className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-3 py-1 rounded-full border border-cyan-500/20">
              5 Hospitals • 10,000 Parameter Vectors
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* PLAINTEXT FEDAVG */}
            <div className="p-6 rounded-xl bg-slate-950 border border-slate-800 space-y-4">
              <div className="flex justify-between items-center border-b border-slate-800 pb-2">
                <strong className="text-base font-bold text-slate-200">Plaintext FedAvg Baseline</strong>
                <span className="text-xs font-mono text-slate-400">Fast / Low Privacy</span>
              </div>
              <div className="space-y-2 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-slate-400">Aggregation Time:</span>
                  <strong className="text-white">{REAL_BENCHMARK_RESULTS.plaintextFedAvgTimeSec.toFixed(4)}s</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Payload Size:</span>
                  <strong className="text-white">{REAL_BENCHMARK_RESULTS.plaintextFedAvgPayloadKB} KB</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Privacy Guarantee:</span>
                  <span className="text-red-400 font-bold">Plaintext Unencrypted</span>
                </div>
              </div>
            </div>

            {/* SECAGG+ / CKKS ENCRYPTED */}
            <div className="p-6 rounded-xl bg-slate-950 border border-cyan-500/40 space-y-4">
              <div className="flex justify-between items-center border-b border-slate-800 pb-2">
                <strong className="text-base font-bold text-cyan-400">SecAgg+ / CKKS Encrypted</strong>
                <span className="text-xs font-mono text-cyan-400">Max Cryptographic Security</span>
              </div>
              <div className="space-y-2 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-slate-400">Total Execution Time:</span>
                  <strong className="text-emerald-400">{REAL_BENCHMARK_RESULTS.secaggPlusTimeSec.toFixed(4)}s</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Encrypted Payload:</span>
                  <strong className="text-cyan-400">{REAL_BENCHMARK_RESULTS.secaggPlusPayloadKB.toFixed(1)} KB</strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Communication Overhead:</span>
                  <strong className="text-indigo-400">+{REAL_BENCHMARK_RESULTS.communicationOverheadPercent.toFixed(0)}%</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 flex items-center justify-between">
            <span className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <strong>Trade-off Analysis:</strong> SecAgg+ incurs a 24.8x payload overhead to guarantee zero plaintext model leakage during transfer.
            </span>
          </div>
        </div>
      </div>
    </section>
  );
};
