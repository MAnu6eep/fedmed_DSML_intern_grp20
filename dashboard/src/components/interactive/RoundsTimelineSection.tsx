import React, { useState } from "react";
import { Clock, FileText } from "lucide-react";
import { DEMO_ROUND_HISTORY } from "../../data/demoData";

export const RoundsTimelineSection: React.FC = () => {
  const [selectedRoundIndex, setSelectedRoundIndex] = useState(4); // Default Round 5
  const [useRealData, setUseRealData] = useState(false);

  const currentRound = DEMO_ROUND_HISTORY[selectedRoundIndex];

  return (
    <section id="fed-rounds" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
            <Clock className="w-3.5 h-3.5" />
            SECTION 8 — FEDERATED ROUND TIMELINE
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Communication Round Iterative Progress
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Track global model metrics across federated training rounds. Each round executes training, update collection, secure aggregation, and global broadcasting.
          </p>
        </div>

        {/* DATA SOURCE TOGGLE BADGE */}
        <div className="flex justify-between items-center bg-slate-900 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center gap-2 text-xs">
            <FileText className="w-4 h-4 text-cyan-400" />
            <span>Data Source: <strong className="text-white">{useRealData ? "Repo Test Result Log (fedavg_fedprox_comparison.json)" : "Illustrative Presentation Demo Data"}</strong></span>
          </div>

          <button
            onClick={() => setUseRealData(!useRealData)}
            className="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 text-xs font-semibold hover:bg-slate-700 transition-all border border-slate-700"
          >
            Switch to {useRealData ? "Demo Values" : "Repo Log File Values"}
          </button>
        </div>

        {/* TIMELINE STEPPER BUTTONS */}
        <div className="flex justify-between items-center gap-2 overflow-x-auto pb-2">
          {DEMO_ROUND_HISTORY.map((item, idx) => (
            <button
              key={item.round}
              onClick={() => setSelectedRoundIndex(idx)}
              className={`flex-1 p-4 rounded-xl border text-center transition-all min-w-[120px] ${
                selectedRoundIndex === idx
                  ? "bg-cyan-500/10 border-cyan-500 text-white shadow-lg shadow-cyan-500/10 scale-105"
                  : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              <span className="text-[10px] font-bold uppercase text-slate-400 block mb-1">Communication</span>
              <strong className="text-base font-extrabold block">Round {item.round}</strong>
              <span className="text-xs font-mono text-emerald-400 block mt-1">Dice: {item.valDice}</span>
            </button>
          ))}
        </div>

        {/* ACTIVE ROUND DISPLAY METRICS */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <span className="h-3 w-3 rounded-full bg-emerald-500 animate-pulse" />
              <h3 className="text-2xl font-bold text-white">Federated Round {currentRound.round} State</h3>
            </div>
            <span className="text-xs font-mono text-slate-400">
              Active Hospitals: <strong className="text-white">{currentRound.activeHospitals}/3</strong>
            </span>
          </div>

          {/* METRIC SCORECARDS */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">Global Val Loss</span>
              <strong className="text-2xl font-bold text-red-400 font-mono mt-1 block">{currentRound.valLoss}</strong>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">Global Val Dice</span>
              <strong className="text-2xl font-bold text-emerald-400 font-mono mt-1 block">{currentRound.valDice}</strong>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">HD95 Distance</span>
              <strong className="text-2xl font-bold text-cyan-400 font-mono mt-1 block">{currentRound.hd95} mm</strong>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center">
              <span className="text-[10px] text-slate-500 font-bold uppercase block">Round Time</span>
              <strong className="text-2xl font-bold text-indigo-400 font-mono mt-1 block">{currentRound.roundTimeSec}s</strong>
            </div>
          </div>

          {/* CLASS DICE BREAKDOWN */}
          <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
            <span className="text-xs font-mono uppercase text-slate-400 font-bold block">Tumor Sub-Region Validation Dice Breakdown</span>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="p-3 rounded bg-slate-900 border border-slate-800 flex justify-between items-center">
                <span>Enhancing Tumor (ET):</span>
                <strong className="text-cyan-400 font-mono text-sm">{currentRound.etDice}</strong>
              </div>
              <div className="p-3 rounded bg-slate-900 border border-slate-800 flex justify-between items-center">
                <span>Tumor Core (TC):</span>
                <strong className="text-indigo-400 font-mono text-sm">{currentRound.tcDice}</strong>
              </div>
              <div className="p-3 rounded bg-slate-900 border border-slate-800 flex justify-between items-center">
                <span>Whole Tumor (WT):</span>
                <strong className="text-emerald-400 font-mono text-sm">{currentRound.wtDice}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
