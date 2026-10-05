import React, { useState, useEffect } from "react";
import { ShieldCheck, Play, ArrowDown, Database, Cpu, Lock, Network, Brain } from "lucide-react";
import { HOSPITALS_DATA } from "../../data/demoData";

interface HeroSectionProps {
  onExplore: () => void;
  onStartPresentation: () => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  onExplore,
  onStartPresentation,
}) => {
  const [pulseIndex, setPulseIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setPulseIndex((prev) => (prev + 1) % 3);
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  return (
    <section className="relative min-h-[85vh] flex flex-col items-center justify-center py-12 px-6 overflow-hidden bg-slate-950 border-b border-slate-800/80">
      {/* BACKGROUND GRAPHICS */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(14,165,233,0.15),rgba(255,255,255,0))]" />
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-500/5 blur-[120px] rounded-full pointer-events-none" />

      {/* HERO BADGE */}
      <div className="relative z-10 flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/80 border border-cyan-500/30 text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-6 backdrop-blur-sm">
        <ShieldCheck className="w-4 h-4 text-cyan-400" />
        <span>Hybrid FL Framework • MONAI + TenSEAL + Flower</span>
      </div>

      {/* MAIN TITLE */}
      <div className="relative z-10 max-w-4xl text-center space-y-4">
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-white">
          FED<span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400">MED</span>
        </h1>
        <p className="text-xl md:text-3xl font-semibold text-slate-200">
          Privacy-Preserving Federated Learning for 3D Brain Tumor Segmentation
        </p>
        <p className="text-base md:text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          Train a shared medical AI model across distributed hospitals without centralizing patient MRI data.
        </p>
      </div>

      {/* ANIMATED NETWORK VISUALIZATION */}
      <div className="relative z-10 w-full max-w-5xl my-10 p-8 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md shadow-2xl">
        <div className="text-center mb-6">
          <span className="text-xs font-bold uppercase tracking-widest text-slate-400 flex items-center justify-center gap-2">
            <Network className="w-4 h-4 text-cyan-400" />
            Live Distributed Architecture Particle Flow
          </span>
        </div>

        {/* HOSPITALS ROW */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 relative">
          {HOSPITALS_DATA.map((hospital, index) => {
            const isActive = pulseIndex === index;
            return (
              <div
                key={hospital.id}
                className={`relative p-5 rounded-xl border transition-all duration-500 ${
                  isActive
                    ? "bg-slate-800/80 border-cyan-500/80 shadow-lg shadow-cyan-500/10 scale-105"
                    : "bg-slate-950/80 border-slate-800 opacity-90"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Database className="w-4 h-4 text-cyan-400" />
                    <span className="font-bold text-sm text-white">{hospital.name}</span>
                  </div>
                  <span className="px-2 py-0.5 text-[10px] font-semibold uppercase rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    Local Data Isolated
                  </span>
                </div>

                <div className="space-y-1 text-xs text-slate-400">
                  <div className="flex justify-between">
                    <span>Sample Count:</span>
                    <strong className="text-slate-200">{hospital.samples} MRI Scans</strong>
                  </div>
                  <div className="flex justify-between">
                    <span>Dataset:</span>
                    <span className="text-slate-300 truncate max-w-[140px]">{hospital.datasetName}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Local Dice:</span>
                    <strong className="text-cyan-400">{hospital.dice}</strong>
                  </div>
                </div>

                {/* ANIMATED PULSE PARTICLE TO SERVER */}
                {isActive && (
                  <div className="absolute -bottom-6 left-1/2 -translate-x-1/2 flex flex-col items-center">
                    <div className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                    <span className="text-[9px] font-bold text-cyan-400 uppercase tracking-tighter mt-1">
                      Encrypted Update Packet
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* FLOW CONNECTOR ARROWS */}
        <div className="flex justify-center my-4 relative">
          <div className="flex items-center gap-8 text-slate-500 text-xs font-mono">
            <span className="flex items-center gap-1 text-cyan-400/90 font-semibold">
              <Lock className="w-3.5 h-3.5" />
              SecAgg+ / CKKS Encryption
            </span>
            <span>↓</span>
            <span className="flex items-center gap-1 text-cyan-400/90 font-semibold">
              <Lock className="w-3.5 h-3.5" />
              Zero-Knowledge Parameter Updates
            </span>
          </div>
        </div>

        {/* FEDERATED SERVER */}
        <div className="max-w-md mx-auto p-5 rounded-xl bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border border-blue-500/40 shadow-xl text-center relative overflow-hidden">
          <div className="absolute top-0 right-0 px-3 py-1 bg-blue-500/20 text-blue-400 text-[10px] font-bold uppercase rounded-bl-lg">
            Flower Federated Server
          </div>

          <div className="flex items-center justify-center gap-3 mb-2">
            <Cpu className="w-5 h-5 text-blue-400 animate-pulse" />
            <h3 className="font-bold text-white text-base">Federated Aggregation Engine</h3>
          </div>

          <p className="text-xs text-slate-400 mb-3">
            Strategy: <strong className="text-cyan-400">FedAvg / FedProx / SCAFFOLD</strong> • Weighted Parameter Fusion
          </p>

          <div className="flex justify-center items-center gap-6 text-xs text-slate-300 border-t border-slate-700/60 pt-3">
            <div>
              <span className="block text-[10px] text-slate-500">AGGREGATED MODEL</span>
              <strong className="text-emerald-400">3D U-Net Global Weights</strong>
            </div>
            <div>
              <span className="block text-[10px] text-slate-500">SECURITY</span>
              <strong className="text-cyan-400">Zero Raw Data Leakage</strong>
            </div>
          </div>
        </div>

        {/* GLOBAL MODEL BROADCAST ARROWS */}
        <div className="flex justify-center my-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20">
            <Brain className="w-3.5 h-3.5 text-emerald-400" />
            Global Model Broadcast to Hospitals & Inference Engine
          </div>
        </div>
      </div>

      {/* CALL TO ACTION BUTTONS */}
      <div className="relative z-10 flex flex-wrap items-center justify-center gap-4 mt-2">
        <button
          onClick={onExplore}
          className="flex items-center gap-2.5 px-6 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 text-white font-bold text-sm shadow-xl shadow-cyan-500/25 hover:brightness-110 active:scale-95 transition-all"
        >
          <span>Explore System Pipeline</span>
          <ArrowDown className="w-4 h-4" />
        </button>

        <button
          onClick={onStartPresentation}
          className="flex items-center gap-2.5 px-6 py-3.5 rounded-xl bg-slate-900 border border-slate-700 text-slate-200 font-bold text-sm shadow-lg hover:bg-slate-800 hover:text-white active:scale-95 transition-all"
        >
          <Play className="w-4 h-4 text-emerald-400 fill-emerald-400/20" />
          <span>Launch PM Presentation Mode</span>
        </button>
      </div>
    </section>
  );
};
