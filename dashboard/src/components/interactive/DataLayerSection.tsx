import React, { useState } from "react";
import { Layers, CheckCircle2, Sliders } from "lucide-react";
import { PREPROCESSING_PIPELINE } from "../../data/demoData";

export const DataLayerSection: React.FC = () => {
  const [selectedStep, setSelectedStep] = useState(0);

  const modalities = [
    { name: "FLAIR", desc: "Fluid Attenuated Inversion Recovery (Edema / Hyperintensity)", color: "from-cyan-500 to-blue-600" },
    { name: "T1", desc: "Native T1-Weighted (Anatomical Structure & Boundaries)", color: "from-blue-600 to-indigo-600" },
    { name: "T1ce", desc: "T1 Post-Contrast / Gd (Enhancing Tumor Core & Active Edges)", color: "from-indigo-600 to-purple-600" },
    { name: "T2", desc: "T2-Weighted (Fluid / Cystic & Necrotic Regions)", color: "from-purple-600 to-pink-600" },
  ];

  return (
    <section id="data-layer" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <Layers className="w-3.5 h-3.5" />
            SECTION 2 — MONAI DATA PREPROCESSING PIPELINE
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Raw 3D MRI Volumetric Ingestion to Standardized Tensors
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Each patient dataset consists of 4 complementary MRI pulse sequences. MONAI dictionary transforms harmonize voxel resolution, orientation, and intensity range.
          </p>
        </div>

        {/* 4 MRI MODALITIES GRID */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {modalities.map((mod, idx) => (
            <div key={idx} className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 relative overflow-hidden group hover:border-cyan-500/40 transition-all">
              <div className={`h-1.5 w-full bg-gradient-to-r ${mod.color} rounded-full mb-2`} />
              <div className="flex items-center justify-between">
                <span className="font-extrabold text-lg text-white">{mod.name}</span>
                <span className="text-[10px] uppercase font-bold text-slate-400 bg-slate-800 px-2 py-0.5 rounded">Channel {idx}</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">{mod.desc}</p>
            </div>
          ))}
        </div>

        {/* INTERACTIVE PREPROCESSING PIPELINE FLOW */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-cyan-400" />
              MONAI Preprocessing Pipeline Step Inspector
            </h3>
            <span className="text-xs text-slate-400 font-mono">fedmed/data/loader.py</span>
          </div>

          {/* STEP TABS */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {PREPROCESSING_PIPELINE.map((step, idx) => (
              <button
                key={step.step}
                onClick={() => setSelectedStep(idx)}
                className={`p-3 rounded-xl border text-left transition-all ${
                  selectedStep === idx
                    ? "bg-cyan-500/10 border-cyan-500 text-white shadow-lg shadow-cyan-500/10"
                    : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] font-bold uppercase text-cyan-400">Step {step.step}</span>
                  {selectedStep === idx && <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />}
                </div>
                <strong className="block text-xs font-semibold text-slate-200 truncate">{step.name}</strong>
              </button>
            ))}
          </div>

          {/* ACTIVE STEP DETAILS DISPLAY */}
          {PREPROCESSING_PIPELINE[selectedStep] && (
            <div className="p-6 rounded-xl bg-slate-950 border border-cyan-500/30 grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
              <div className="md:col-span-2 space-y-2">
                <span className="text-xs font-mono text-cyan-400 uppercase tracking-wider">
                  MONAI Transform: {PREPROCESSING_PIPELINE[selectedStep].name}
                </span>
                <h4 className="text-xl font-bold text-white">
                  {PREPROCESSING_PIPELINE[selectedStep].title}
                </h4>
                <p className="text-sm text-slate-300 leading-relaxed">
                  {PREPROCESSING_PIPELINE[selectedStep].description}
                </p>
              </div>

              <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 text-center space-y-1">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block">Output Shape</span>
                <strong className="text-lg font-mono text-emerald-400 block">
                  {PREPROCESSING_PIPELINE[selectedStep].outputShape}
                </strong>
                <span className="text-[10px] text-slate-400 block">Ready for 3D U-Net Ingestion</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
