import React, { useState } from "react";
import { Server, CheckCircle2 } from "lucide-react";
import { STRATEGIES_DATA, type StrategyInfo } from "../../data/demoData";

export const FederatedServerSection: React.FC = () => {
  const [selectedStrategyId, setSelectedStrategyId] = useState<"fedavg" | "fedprox" | "scaffold">("fedavg");

  const currentStrategy = STRATEGIES_DATA.find((s) => s.id === selectedStrategyId) || STRATEGIES_DATA[0];

  return (
    <section id="fed-server" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold uppercase tracking-wider">
            <Server className="w-3.5 h-3.5" />
            SECTION 7 — FEDERATED SERVER & STRATEGIES
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Flower Federation Aggregation Engine
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            The central server merges incoming parameter updates from Hospital A, B, and C into a single global model using specialized federated optimization strategies.
          </p>
        </div>

        {/* STRATEGY SWITCHER TABS */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {STRATEGIES_DATA.map((strategy: StrategyInfo) => {
            const isSelected = strategy.id === selectedStrategyId;
            return (
              <button
                key={strategy.id}
                onClick={() => setSelectedStrategyId(strategy.id)}
                className={`p-5 rounded-2xl border text-left transition-all ${
                  isSelected
                    ? "bg-slate-900 border-cyan-500 shadow-xl shadow-cyan-500/10"
                    : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs font-mono font-bold text-cyan-400 uppercase">{strategy.name}</span>
                  {isSelected && <CheckCircle2 className="w-4 h-4 text-cyan-400" />}
                </div>
                <strong className="block text-base text-white font-bold">{strategy.fullName}</strong>
                <p className="text-xs text-slate-400 mt-2 line-clamp-2">{strategy.description}</p>
              </button>
            );
          })}
        </div>

        {/* ACTIVE STRATEGY DETAILS DISPLAY */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <span className="text-xs font-mono uppercase text-cyan-400 font-bold">Flower Server Strategy</span>
              <h3 className="text-2xl font-bold text-white mt-1">{currentStrategy.fullName} ({currentStrategy.name})</h3>
            </div>
            <span className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-3 py-1 rounded-full border border-cyan-500/20">
              fedmed/federation/server.py
            </span>
          </div>

          {/* FORMULA & USE CASE */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
              <span className="text-xs font-mono uppercase text-slate-400 font-bold block">Mathematical Aggregation Formula</span>
              <div className="p-4 rounded bg-slate-900 border border-slate-800 text-center font-mono text-cyan-300 text-sm overflow-x-auto">
                {currentStrategy.formula}
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">{currentStrategy.description}</p>
            </div>

            <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
              <span className="text-xs font-mono uppercase text-slate-400 font-bold block">Clinical Application & Target Distribution</span>
              <p className="text-sm text-slate-200 leading-relaxed font-semibold">{currentStrategy.useCase}</p>
              
              <div className="pt-3 border-t border-slate-900 space-y-2">
                <span className="text-[10px] uppercase font-bold text-slate-400 block">Repository Hyperparameters</span>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  {Object.entries(currentStrategy.hyperparameters).map(([key, val]) => (
                    <div key={key} className="p-2 rounded bg-slate-900 flex justify-between">
                      <span className="text-slate-400">{key}:</span>
                      <strong className="text-emerald-400">{val}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
