import React, { useState, useEffect } from "react";
import { Play, RotateCcw, Lock, CheckCircle2 } from "lucide-react";
import { HOSPITALS_DATA } from "../../data/demoData";

export const LocalTrainingSection: React.FC = () => {
  const [progressA, setProgressA] = useState(0);
  const [progressB, setProgressB] = useState(0);
  const [progressC, setProgressC] = useState(0);
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (isTraining) {
      timer = setInterval(() => {
        setProgressA((prev) => Math.min(prev + 12, 100));
        setProgressB((prev) => Math.min(prev + 8, 100));
        setProgressC((prev) => Math.min(prev + 15, 100));
      }, 300);
    }
    return () => clearInterval(timer);
  }, [isTraining]);

  const handleStart = () => {
    setProgressA(0);
    setProgressB(0);
    setProgressC(0);
    setIsTraining(true);
  };

  const isComplete = progressA === 100 && progressB === 100 && progressC === 100;

  return (
    <section id="local-training" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
            <Play className="w-3.5 h-3.5" />
            SECTION 5 — SIMULTANEOUS LOCAL TRAINING
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            On-Premise 3D U-Net Training Passes
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            All hospital nodes optimize local weights simultaneously using MONAI Dice-BCE loss. Only local model deltas (Δw) are computed.
          </p>
        </div>

        {/* TRAINING SIMULATOR CONTAINER */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h3 className="text-lg font-bold text-white">Local Epoch Simulation</h3>
              <p className="text-xs text-slate-400">Watch local gradient updates accumulate inside hospital firewalls.</p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleStart}
                disabled={isTraining && !isComplete}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500 text-slate-950 font-bold text-xs hover:bg-emerald-400 transition-all disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5 fill-slate-950" />
                {isTraining ? "Training Active..." : "Run Concurrent Local Training"}
              </button>

              <button
                onClick={() => { setIsTraining(false); setProgressA(0); setProgressB(0); setProgressC(0); }}
                className="p-2 rounded-lg bg-slate-800 text-slate-300 hover:text-white"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* PROGRESS BARS PER HOSPITAL */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { hosp: HOSPITALS_DATA[0], prog: progressA },
              { hosp: HOSPITALS_DATA[1], prog: progressB },
              { hosp: HOSPITALS_DATA[2], prog: progressC },
            ].map(({ hosp, prog }) => (
              <div key={hosp.id} className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <strong className="text-white font-bold">{hosp.name}</strong>
                  <span className="text-cyan-400 font-mono">{prog}%</span>
                </div>

                <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-300"
                    style={{ width: `${prog}%` }}
                  />
                </div>

                <div className="flex justify-between text-[10px] text-slate-400 pt-2 border-t border-slate-900">
                  <span>Epoch 1 / 1</span>
                  <span className="flex items-center gap-1 text-emerald-400">
                    <Lock className="w-3 h-3" />
                    Data Stays Local
                  </span>
                </div>

                {prog === 100 && (
                  <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20 text-[10px] text-emerald-300 font-mono text-center flex items-center justify-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                    Model Update Δw Generated
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* BOTTOM VERIFICATION BAR */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between text-xs text-slate-300">
            <span className="flex items-center gap-2">
              <Lock className="w-4 h-4 text-cyan-400" />
              <strong>Privacy Assurance:</strong> Zero MRI voxels leave local memory buffers during training iterations.
            </span>
            <span className="text-slate-400 font-mono">FedMed Local Engine</span>
          </div>
        </div>
      </div>
    </section>
  );
};
