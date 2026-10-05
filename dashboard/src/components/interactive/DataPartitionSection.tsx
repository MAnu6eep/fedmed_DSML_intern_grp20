import React, { useState } from "react";
import { PieChart, Sliders } from "lucide-react";

export const DataPartitionSection: React.FC = () => {
  const [alpha, setAlpha] = useState(0.5);
  const [partitionMode, setPartitionMode] = useState<"iid" | "dirichlet">("dirichlet");

  return (
    <section id="data-partition" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <PieChart className="w-3.5 h-3.5" />
            SECTION 9 — DATA PARTITIONING & HETEROGENEITY
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            IID vs. Dirichlet Non-IID Hospital Data Splitting
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Real hospitals encounter non-identical patient populations. FedMed models Non-IID clinical distributions using Dirichlet concentration sampling.
          </p>
        </div>

        {/* TOGGLE IID vs DIRICHLET */}
        <div className="flex justify-center">
          <div className="bg-slate-900 p-1 rounded-xl border border-slate-800 flex gap-2">
            <button
              onClick={() => setPartitionMode("iid")}
              className={`px-5 py-2.5 rounded-lg text-xs font-bold transition-all ${
                partitionMode === "iid"
                  ? "bg-blue-500/20 text-blue-300 border border-blue-500/40 shadow-lg"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              IID Uniform Partitioning (partition_iid)
            </button>
            <button
              onClick={() => setPartitionMode("dirichlet")}
              className={`px-5 py-2.5 rounded-lg text-xs font-bold transition-all ${
                partitionMode === "dirichlet"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-lg"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Dirichlet Non-IID Partitioning (partition_dirichlet)
            </button>
          </div>
        </div>

        {/* INTERACTIVE CONTROLS */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h3 className="text-lg font-bold text-white">Dirichlet Concentration Parameter (α) Inspector</h3>
              <p className="text-xs text-slate-400">Smaller α values induce higher distribution skew across hospital sites.</p>
            </div>
            <span className="text-xs font-mono text-cyan-400">fedmed/data/partitioner.py</span>
          </div>

          {partitionMode === "dirichlet" && (
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Sliders className="w-4 h-4 text-cyan-400" />
                  Dirichlet Concentration Parameter (α):
                </span>
                <strong className="text-cyan-400 text-sm">{alpha.toFixed(2)}</strong>
              </div>
              <input
                type="range"
                min="0.05"
                max="5.0"
                step="0.05"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 bg-slate-800 h-2 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>α = 0.05 (Extreme Heterogeneity)</span>
                <span>α = 0.5 (Realistic Clinical Heterogeneity)</span>
                <span>α = 5.0 (Near-IID Uniform)</span>
              </div>
            </div>
          )}

          {/* SIMULATED HOSPITAL DISTRIBUTION BAR CHARTS */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { name: "Hospital A", et: partitionMode === "iid" ? 33 : Math.round(50 * (1 / (alpha + 0.5))), tc: 35, wt: 85 },
              { name: "Hospital B", et: partitionMode === "iid" ? 33 : Math.round(15 * alpha), tc: 25, wt: 65 },
              { name: "Hospital C", et: partitionMode === "iid" ? 33 : Math.round(60 * (alpha / (alpha + 0.2))), tc: 55, wt: 95 },
            ].map((hosp, idx) => (
              <div key={idx} className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                <strong className="text-sm text-white font-bold block">{hosp.name} Sample Split</strong>
                <div className="space-y-2 text-xs">
                  <div>
                    <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                      <span>Enhancing Tumor (ET)</span>
                      <span className="text-cyan-400 font-mono">{hosp.et}%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-cyan-400 transition-all duration-300" style={{ width: `${Math.min(hosp.et, 100)}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                      <span>Tumor Core (TC)</span>
                      <span className="text-indigo-400 font-mono">{hosp.tc}%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-indigo-400 transition-all duration-300" style={{ width: `${Math.min(hosp.tc, 100)}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                      <span>Whole Tumor (WT)</span>
                      <span className="text-emerald-400 font-mono">{hosp.wt}%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-400 transition-all duration-300" style={{ width: `${Math.min(hosp.wt, 100)}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
