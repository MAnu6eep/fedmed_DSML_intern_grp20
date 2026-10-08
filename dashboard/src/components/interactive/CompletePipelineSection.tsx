import React, { useState } from "react";
import { Network, ChevronRight } from "lucide-react";
import { PIPELINE_STAGES } from "../../data/demoData";

export const CompletePipelineSection: React.FC = () => {
  const [activeStageId, setActiveStageId] = useState(1);

  const handleNext = () => {
    setActiveStageId((prev) => (prev < 14 ? prev + 1 : 1));
  };

  const handlePrev = () => {
    setActiveStageId((prev) => (prev > 1 ? prev - 1 : 14));
  };

  const currentStage = PIPELINE_STAGES.find((s) => s.id === activeStageId) || PIPELINE_STAGES[0];

  return (
    <section id="complete-pipeline" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <Network className="w-3.5 h-3.5" />
            SECTION 14 — COMPLETE SYSTEM PIPELINE
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            End-to-End Integrated FedMed Architecture
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Interactive system diagram connecting all 14 stages from raw MRI ingestion to local training, secure aggregation, global update, and final tumor prediction.
          </p>
        </div>

        {/* PIPELINE NAVIGATION BAR */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <span className="text-xs font-mono uppercase text-cyan-400 font-bold">Stage {activeStageId} of 14</span>
              <h3 className="text-2xl font-bold text-white mt-1">{currentStage.title}</h3>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={handlePrev}
                className="px-4 py-2 rounded-lg bg-slate-800 text-slate-300 font-bold text-xs hover:bg-slate-700"
              >
                Previous Stage
              </button>

              <button
                onClick={handleNext}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500 text-slate-950 font-bold text-xs hover:bg-cyan-400 shadow-md shadow-cyan-500/20"
              >
                <span>Next Stage</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* ACTIVE STAGE CARD */}
          <div className="p-6 rounded-xl bg-slate-950 border border-cyan-500/40 space-y-3">
            <strong className="text-lg text-cyan-300 font-bold block">{currentStage.subtitle}</strong>
            <p className="text-sm text-slate-300 leading-relaxed">{currentStage.shortDesc}</p>
          </div>

          {/* 14-STEP VISUAL PROGRESSION GRID */}
          <div className="grid grid-cols-2 md:grid-cols-7 gap-3 pt-2">
            {PIPELINE_STAGES.map((stg) => {
              const isSelected = stg.id === activeStageId;
              return (
                <button
                  key={stg.id}
                  onClick={() => setActiveStageId(stg.id)}
                  className={`p-3 rounded-xl border text-left transition-all ${
                    isSelected
                      ? "bg-cyan-500/20 border-cyan-500 text-white shadow-lg shadow-cyan-500/10 scale-105"
                      : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <span className="text-[9px] font-bold uppercase text-cyan-400 block">Stage {stg.id}</span>
                  <strong className="text-xs block truncate my-0.5 text-slate-200">{stg.title.split(". ")[1]}</strong>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};
