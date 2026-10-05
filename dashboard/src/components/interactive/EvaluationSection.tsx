import React from "react";
import { Activity, Target, Award, BarChart2 } from "lucide-react";

export const EvaluationSection: React.FC = () => {
  return (
    <section id="evaluation" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
            <Activity className="w-3.5 h-3.5" />
            SECTION 10 — MODEL EVALUATION & CLINICAL METRICS
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Volumetric 3D MRI Segmentation Metrics
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Evaluates the aggregated global 3D U-Net against validation ground truth masks using standard medical imaging metrics.
          </p>
        </div>

        {/* METRIC SCORE CARDS */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-2 relative overflow-hidden group hover:border-emerald-500/40 transition-all">
            <div className="flex justify-between items-center text-slate-400">
              <span className="text-xs font-bold uppercase">Dice Score (DSC)</span>
              <Award className="w-5 h-5 text-emerald-400" />
            </div>
            <strong className="text-4xl font-extrabold text-white block font-mono">0.912</strong>
            <p className="text-[11px] text-slate-400">Voxel overlap agreement (0 to 1)</p>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-2 relative overflow-hidden group hover:border-cyan-500/40 transition-all">
            <div className="flex justify-between items-center text-slate-400">
              <span className="text-xs font-bold uppercase">IoU / Jaccard</span>
              <Target className="w-5 h-5 text-cyan-400" />
            </div>
            <strong className="text-4xl font-extrabold text-white block font-mono">0.838</strong>
            <p className="text-[11px] text-slate-400">Intersection over Union ratio</p>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-2 relative overflow-hidden group hover:border-indigo-500/40 transition-all">
            <div className="flex justify-between items-center text-slate-400">
              <span className="text-xs font-bold uppercase">HD95 Distance</span>
              <BarChart2 className="w-5 h-5 text-indigo-400" />
            </div>
            <strong className="text-4xl font-extrabold text-white block font-mono">4.2 mm</strong>
            <p className="text-[11px] text-slate-400">95th percentile boundary error</p>
          </div>

          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-2 relative overflow-hidden group hover:border-red-500/40 transition-all">
            <div className="flex justify-between items-center text-slate-400">
              <span className="text-xs font-bold uppercase">Validation Loss</span>
              <Activity className="w-5 h-5 text-red-400" />
            </div>
            <strong className="text-4xl font-extrabold text-white block font-mono">0.251</strong>
            <p className="text-[11px] text-slate-400">MONAI Dice-BCE combined loss</p>
          </div>
        </div>

        {/* DETAILED REGION BREAKDOWN */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <div className="flex justify-between items-center border-b border-slate-800 pb-3">
            <h3 className="text-lg font-bold text-white">BraTS Clinical Sub-Region Performance</h3>
            <span className="text-xs font-mono text-cyan-400">fedmed/core/evaluation.py</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
              <span className="text-[10px] uppercase font-bold text-cyan-400">Enhancing Tumor (ET)</span>
              <strong className="text-xl font-bold text-white block font-mono">DSC: 0.884</strong>
              <p className="text-[10px] text-slate-400">Gd-enhancing active neoplastic tissue</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
              <span className="text-[10px] uppercase font-bold text-indigo-400">Tumor Core (TC)</span>
              <strong className="text-xl font-bold text-white block font-mono">DSC: 0.902</strong>
              <p className="text-[10px] text-slate-400">Necrotic core + Enhancing tumor</p>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
              <span className="text-[10px] uppercase font-bold text-emerald-400">Whole Tumor (WT)</span>
              <strong className="text-xl font-bold text-white block font-mono">DSC: 0.935</strong>
              <p className="text-[10px] text-slate-400">Edema + Necrotic core + Enhancing tumor</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
