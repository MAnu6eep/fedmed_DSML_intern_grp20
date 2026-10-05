import React, { useState } from "react";
import { Brain, Layers } from "lucide-react";

export const SegmentationSection: React.FC = () => {
  const [sliceIndex, setSliceIndex] = useState(8);
  const [selectedModality, setSelectedModality] = useState<"FLAIR" | "T1" | "T1ce" | "T2">("FLAIR");
  const [activeLayer, setActiveLayer] = useState<"all" | "gt" | "pred">("all");

  return (
    <section id="segmentation" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <Brain className="w-3.5 h-3.5" />
            SECTION 13 — 3D BRAIN TUMOR SEGMENTATION OUTPUT
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Multi-Modal Volumetric Tumor Prediction Viewer
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Render 3D U-Net segmentation predictions across 2D cross-sectional axial slices. Compares ground truth expert annotations with global model predictions.
          </p>
        </div>

        {/* INTERACTIVE CONTROLS BAR */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex flex-wrap justify-between items-center gap-4 border-b border-slate-800 pb-4">
            {/* MODALITY SELECTOR */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-400 uppercase">MRI Modality:</span>
              <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
                {(["FLAIR", "T1", "T1ce", "T2"] as const).map((mod) => (
                  <button
                    key={mod}
                    onClick={() => setSelectedModality(mod)}
                    className={`px-3 py-1 text-xs font-bold rounded ${
                      selectedModality === mod ? "bg-cyan-500 text-slate-950" : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {mod}
                  </button>
                ))}
              </div>
            </div>

            {/* LAYER OVERLAY TOGGLE */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-400 uppercase">Mask Layer:</span>
              <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setActiveLayer("all")}
                  className={`px-3 py-1 text-xs font-bold rounded ${activeLayer === "all" ? "bg-emerald-500 text-slate-950" : "text-slate-400 hover:text-white"}`}
                >
                  Both Masks
                </button>
                <button
                  onClick={() => setActiveLayer("gt")}
                  className={`px-3 py-1 text-xs font-bold rounded ${activeLayer === "gt" ? "bg-cyan-500 text-slate-950" : "text-slate-400 hover:text-white"}`}
                >
                  Ground Truth
                </button>
                <button
                  onClick={() => setActiveLayer("pred")}
                  className={`px-3 py-1 text-xs font-bold rounded ${activeLayer === "pred" ? "bg-indigo-500 text-white" : "text-slate-400 hover:text-white"}`}
                >
                  3D U-Net Prediction
                </button>
              </div>
            </div>
          </div>

          {/* SLICE DEPTH SLIDER */}
          <div className="space-y-2 font-mono">
            <div className="flex justify-between text-xs text-slate-300">
              <span className="flex items-center gap-1.5 text-cyan-400 font-bold">
                <Layers className="w-4 h-4" />
                Axial Volumetric Depth Slice:
              </span>
              <strong className="text-cyan-400">Slice {sliceIndex} / 16 (Depth Z=32)</strong>
            </div>
            <input
              type="range"
              min="1"
              max="16"
              value={sliceIndex}
              onChange={(e) => setSliceIndex(parseInt(e.target.value))}
              className="w-full accent-cyan-400 bg-slate-800 h-2 rounded-lg cursor-pointer"
            />
          </div>

          {/* VISUAL MRI SCAN VIEWPORT CANVAS / RENDERING */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
            {/* RAW MRI SLICE */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 text-center">
              <span className="text-xs font-mono font-bold text-slate-300 block">1. Raw {selectedModality} MRI Slice</span>
              <div className="relative aspect-square rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center overflow-hidden">
                <div className="w-40 h-40 rounded-full bg-slate-800 border-4 border-slate-700/60 relative flex items-center justify-center">
                  <div className="w-24 h-24 rounded-full bg-slate-700/80 blur-xs" />
                </div>
                <span className="absolute bottom-2 left-2 text-[10px] font-mono text-slate-400 bg-slate-950/80 px-2 py-0.5 rounded">
                  {selectedModality} • Slice #{sliceIndex}
                </span>
              </div>
            </div>

            {/* GROUND TRUTH MASK */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 text-center">
              <span className="text-xs font-mono font-bold text-cyan-400 block">2. Expert Ground Truth Mask</span>
              <div className="relative aspect-square rounded-lg bg-slate-900 border border-cyan-900/40 flex items-center justify-center overflow-hidden">
                <div className="w-40 h-40 rounded-full bg-slate-800 border-4 border-slate-700/60 relative flex items-center justify-center">
                  {/* SIMULATED GROUND TRUTH TUMOR REGIONS */}
                  <div className="w-16 h-16 rounded-full bg-cyan-500/40 border-2 border-cyan-400 flex items-center justify-center">
                    <div className="w-8 h-8 rounded-full bg-indigo-500/60 border-2 border-indigo-400" />
                  </div>
                </div>
                <span className="absolute bottom-2 left-2 text-[10px] font-mono text-cyan-300 bg-slate-950/80 px-2 py-0.5 rounded">
                  BraTS Manual Annotation
                </span>
              </div>
            </div>

            {/* 3D UNET PREDICTION */}
            <div className="p-4 rounded-xl bg-slate-950 border border-emerald-500/40 space-y-3 text-center">
              <span className="text-xs font-mono font-bold text-emerald-400 block">3. FedMed 3D U-Net Prediction</span>
              <div className="relative aspect-square rounded-lg bg-slate-900 border border-emerald-900/40 flex items-center justify-center overflow-hidden">
                <div className="w-40 h-40 rounded-full bg-slate-800 border-4 border-slate-700/60 relative flex items-center justify-center">
                  {/* SIMULATED PREDICTED TUMOR REGIONS */}
                  <div className="w-16 h-16 rounded-full bg-emerald-500/40 border-2 border-emerald-400 flex items-center justify-center">
                    <div className="w-8 h-8 rounded-full bg-teal-400/70 border-2 border-teal-300" />
                  </div>
                </div>
                <span className="absolute bottom-2 left-2 text-[10px] font-mono text-emerald-300 bg-slate-950/80 px-2 py-0.5 rounded">
                  Voxel Dice: 0.912
                </span>
              </div>
            </div>
          </div>

          {/* COLOR LEGEND */}
          <div className="flex flex-wrap items-center justify-center gap-6 text-xs text-slate-300 border-t border-slate-800 pt-4">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-emerald-500 border border-emerald-300" />
              <span>Whole Tumor (WT)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-teal-400 border border-teal-200" />
              <span>Tumor Core (TC)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-cyan-400 border border-cyan-200" />
              <span>Enhancing Tumor (ET)</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
