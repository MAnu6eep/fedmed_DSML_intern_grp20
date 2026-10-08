import React, { useState, useEffect } from "react";
import { Play, Pause, ChevronLeft, ChevronRight, X, ShieldCheck, Keyboard } from "lucide-react";
import { PIPELINE_STAGES, HOSPITALS_DATA, STRATEGIES_DATA, REAL_BENCHMARK_RESULTS } from "../../data/demoData";

interface PresentationModeProps {
  onExit: () => void;
}

export const PresentationMode: React.FC<PresentationModeProps> = ({ onExit }) => {
  const [currentStageId, setCurrentStageId] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-play timer
  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStageId((prev) => (prev < 14 ? prev + 1 : 1));
      }, 7000); // 7s per stage in auto-play
    }
    return () => clearInterval(timer);
  }, [isPlaying]);

  // Keyboard navigation shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        setIsPlaying((prev) => !prev);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        setCurrentStageId((prev) => (prev < 14 ? prev + 1 : 1));
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        setCurrentStageId((prev) => (prev > 1 ? prev - 1 : 14));
      } else if (e.code === "Escape") {
        e.preventDefault();
        onExit();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onExit]);

  const currentStage = PIPELINE_STAGES.find((s) => s.id === currentStageId) || PIPELINE_STAGES[0];

  return (
    <div className="fixed inset-0 z-50 bg-slate-950 text-slate-100 flex flex-col overflow-hidden font-sans">
      {/* PRESENTATION TOP CONTROL BAR */}
      <div className="bg-slate-900/90 border-b border-slate-800 px-8 py-4 flex items-center justify-between gap-4 backdrop-blur-md">
        {/* BRANDING */}
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 p-0.5">
            <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <ShieldCheck className="h-5 w-5 text-cyan-400" />
            </div>
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              FED<span className="text-cyan-400">MED</span> Project Manager Presentation
            </h1>
            <p className="text-xs text-slate-400">
              Stage {currentStageId} of 14 • <strong className="text-cyan-400">{currentStage.title}</strong>
            </p>
          </div>
        </div>

        {/* PROGRESS BAR */}
        <div className="hidden md:flex flex-1 max-w-md items-center gap-3">
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
            <div
              className="bg-gradient-to-r from-cyan-500 to-emerald-400 h-full transition-all duration-500"
              style={{ width: `${(currentStageId / 14) * 100}%` }}
            />
          </div>
          <span className="text-xs font-mono font-bold text-slate-400">{Math.round((currentStageId / 14) * 100)}%</span>
        </div>

        {/* CONTROLS */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
              isPlaying
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                : "bg-emerald-500 text-slate-950 hover:bg-emerald-400 shadow-lg shadow-emerald-500/20"
            }`}
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-amber-300" /> : <Play className="w-4 h-4 fill-slate-950" />}
            <span>{isPlaying ? "Pause Auto-Slide" : "Auto-Play Presentation"}</span>
          </button>

          <div className="flex items-center bg-slate-800 rounded-xl p-1 border border-slate-700">
            <button
              onClick={() => setCurrentStageId((prev) => (prev > 1 ? prev - 1 : 14))}
              className="p-2 hover:text-cyan-400 transition-all text-slate-300"
              title="Previous Stage (← Left Arrow)"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
            <span className="px-2 text-xs font-mono font-bold text-slate-400">{currentStageId}/14</span>
            <button
              onClick={() => setCurrentStageId((prev) => (prev < 14 ? prev + 1 : 1))}
              className="p-2 hover:text-cyan-400 transition-all text-slate-300"
              title="Next Stage (→ Right Arrow)"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>

          <button
            onClick={onExit}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 text-xs font-bold border border-slate-700 transition-all"
            title="Exit Presentation Mode (Esc)"
          >
            <X className="w-4 h-4 text-red-400" />
            <span>Exit (Esc)</span>
          </button>
        </div>
      </div>

      {/* MAIN PRESENTATION CANVAS */}
      <div className="flex-1 p-10 overflow-y-auto flex flex-col justify-center items-center max-w-6xl mx-auto w-full">
        {/* STAGE CONTAINER */}
        <div className="w-full bg-slate-900/90 border border-slate-800 rounded-3xl p-10 shadow-2xl space-y-8 backdrop-blur-lg">
          {/* STAGE TITLE BADGE */}
          <div className="flex items-center justify-between border-b border-slate-800 pb-6">
            <div>
              <span className="px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-mono font-bold uppercase tracking-wider">
                System Stage {currentStageId} • {currentStage.slug}
              </span>
              <h2 className="text-3xl md:text-4xl font-extrabold text-white mt-2 tracking-tight">
                {currentStage.title}
              </h2>
              <p className="text-lg font-semibold text-cyan-300 mt-1">{currentStage.subtitle}</p>
            </div>

            <div className="hidden md:flex items-center gap-2 text-xs text-slate-400 bg-slate-950 px-4 py-2 rounded-xl border border-slate-800">
              <Keyboard className="w-4 h-4 text-cyan-400" />
              <span>Use <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-200 font-mono">Space</kbd> to Pause/Play • <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-200 font-mono">←</kbd> <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-200 font-mono">→</kbd> Nav</span>
            </div>
          </div>

          {/* STAGE DESCRIPTION */}
          <p className="text-base md:text-lg text-slate-300 leading-relaxed max-w-4xl">
            {currentStage.shortDesc}
          </p>

          {/* STAGE VISUAL BODY CONTENT */}
          <div className="p-6 rounded-2xl bg-slate-950 border border-slate-800">
            {currentStageId === 1 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                <div className="p-6 rounded-xl bg-red-950/20 border border-red-900/40 space-y-3">
                  <h3 className="font-bold text-red-400 text-lg">Centralized Data Transfer</h3>
                  <p className="text-slate-300 text-xs">Hospitals transmit raw MRI files across network firewalls. Severe HIPAA / GDPR privacy risk.</p>
                </div>
                <div className="p-6 rounded-xl bg-emerald-950/20 border border-emerald-900/40 space-y-3">
                  <h3 className="font-bold text-emerald-400 text-lg">FedMed Isolated Architecture</h3>
                  <p className="text-slate-300 text-xs">Patient scans remain inside hospital firewalls. Only encrypted model parameter updates are transmitted.</p>
                </div>
              </div>
            )}

            {currentStageId === 2 && (
              <div className="space-y-4">
                <strong className="text-xs font-mono uppercase text-cyan-400 block">MONAI Dictionary Transforms Pipeline</strong>
                <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-xs font-mono">
                  {["LoadImaged", "Orientationd (RAS)", "Spacingd (1.0mm³)", "NormalizeIntensityd", "RandCrop (4,64,64,32)"].map((tf, i) => (
                    <div key={i} className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
                      <span className="text-[10px] text-slate-500 block">Step {i + 1}</span>
                      <strong className="text-cyan-300 font-semibold">{tf}</strong>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {currentStageId === 3 && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {HOSPITALS_DATA.map((h) => (
                  <div key={h.id} className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                    <strong className="text-white text-base font-bold">{h.name}</strong>
                    <div className="text-xs text-slate-400 space-y-1 font-mono">
                      <div>Samples: <strong className="text-white">{h.samples}</strong></div>
                      <div>Dataset: <strong className="text-slate-300">{h.datasetName}</strong></div>
                      <div>Local Dice: <strong className="text-emerald-400">{h.dice}</strong></div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {currentStageId === 4 && (
              <div className="space-y-4">
                <strong className="text-xs font-mono uppercase text-indigo-400 block">FedMedUNet3D Network Structure</strong>
                <div className="grid grid-cols-5 gap-2 text-center text-xs font-mono">
                  <div className="p-3 rounded bg-blue-950/40 border border-blue-500 text-blue-300">Input (4 Ch)</div>
                  <div className="p-3 rounded bg-indigo-950/40 border border-indigo-500 text-indigo-300">Encoder (16-128 Ch)</div>
                  <div className="p-3 rounded bg-purple-950/60 border border-purple-500 text-purple-300 font-bold">Bottleneck (256 Ch)</div>
                  <div className="p-3 rounded bg-cyan-950/40 border border-cyan-500 text-cyan-300">Decoder (128-16 Ch)</div>
                  <div className="p-3 rounded bg-emerald-950/40 border border-emerald-500 text-emerald-300">Output Mask (1 Ch)</div>
                </div>
              </div>
            )}

            {currentStageId === 5 && (
              <div className="space-y-3 text-xs">
                <strong className="text-xs font-mono uppercase text-emerald-400 block">Simultaneous On-Premise Training Iterations</strong>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center font-bold text-white text-base">
                  Data Stays Local • Model Updates Move
                </div>
              </div>
            )}

            {currentStageId === 6 && (
              <div className="grid grid-cols-3 gap-4 text-xs font-mono">
                <div className="p-4 rounded-xl bg-slate-900 border border-cyan-500/40 space-y-1">
                  <strong className="text-cyan-400 font-bold block">TenSEAL CKKS</strong>
                  <span className="text-slate-400 block">Poly Degree: 8192</span>
                  <span className="text-slate-400 block">Scale: 2^40</span>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-blue-500/40 space-y-1">
                  <strong className="text-blue-400 font-bold block">SecAgg+ Protocol</strong>
                  <span className="text-slate-400 block">Threshold: 2/3</span>
                  <span className="text-slate-400 block">Quantization: 16-bit</span>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-purple-500/40 space-y-1">
                  <strong className="text-purple-400 font-bold block">DP-SGD</strong>
                  <span className="text-slate-400 block">Clipping Norm: 1.0</span>
                  <span className="text-slate-400 block">Noise Multiplier: 1.0</span>
                </div>
              </div>
            )}

            {currentStageId === 7 && (
              <div className="grid grid-cols-3 gap-4 text-xs font-mono">
                {STRATEGIES_DATA.map((s) => (
                  <div key={s.id} className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                    <strong className="text-cyan-400 text-sm font-bold block">{s.name}</strong>
                    <p className="text-slate-300 text-[11px] font-sans">{s.description}</p>
                  </div>
                ))}
              </div>
            )}

            {currentStageId === 8 && (
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center text-xs font-mono">
                <span className="text-slate-400 block">Round 5 Final Global Dice Score:</span>
                <strong className="text-emerald-400 text-2xl font-bold">0.9120</strong>
              </div>
            )}

            {currentStageId === 9 && (
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center text-xs font-mono">
                <span className="text-slate-400 block">Dirichlet Concentration Parameter (α):</span>
                <strong className="text-cyan-400 text-lg font-bold">α = 0.5 (Realistic Non-IID Hospital Split)</strong>
              </div>
            )}

            {currentStageId === 10 && (
              <div className="grid grid-cols-4 gap-3 text-center font-mono">
                <div className="p-3 rounded bg-slate-900 border border-slate-800"><span className="text-[10px] text-slate-500 block">Dice</span><strong className="text-emerald-400 text-lg">0.912</strong></div>
                <div className="p-3 rounded bg-slate-900 border border-slate-800"><span className="text-[10px] text-slate-500 block">IoU</span><strong className="text-cyan-400 text-lg">0.838</strong></div>
                <div className="p-3 rounded bg-slate-900 border border-slate-800"><span className="text-[10px] text-slate-500 block">HD95</span><strong className="text-indigo-400 text-lg">4.2mm</strong></div>
                <div className="p-3 rounded bg-slate-900 border border-slate-800"><span className="text-[10px] text-slate-500 block">Loss</span><strong className="text-red-400 text-lg">0.251</strong></div>
              </div>
            )}

            {currentStageId === 11 && (
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-xs font-mono space-y-1">
                <div>SecAgg+ Execution Time: <strong className="text-emerald-400">{REAL_BENCHMARK_RESULTS.secaggPlusTimeSec}s</strong></div>
                <div>Encrypted Payload Size: <strong className="text-cyan-400">{REAL_BENCHMARK_RESULTS.secaggPlusPayloadKB} KB</strong></div>
              </div>
            )}

            {currentStageId === 12 && (
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center text-xs font-mono text-emerald-400">
                Fault Tolerance Quorum: Minimum 2/3 Hospital Nodes Active. Federation Proceeds Uninterrupted.
              </div>
            )}

            {currentStageId === 13 && (
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center text-xs font-mono text-cyan-400">
                Multi-Class Tumor Mask Prediction: Enhancing Tumor (ET), Tumor Core (TC), Whole Tumor (WT).
              </div>
            )}

            {currentStageId === 14 && (
              <div className="p-4 rounded-xl bg-slate-900 border border-cyan-500 text-center text-xs font-mono text-white font-bold">
                Complete FedMed Hybrid Pipeline Verified End-to-End.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
