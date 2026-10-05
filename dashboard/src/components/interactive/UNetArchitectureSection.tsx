import React, { useState } from "react";
import { Cpu, Info } from "lucide-react";
import { UNET_LAYERS, type UNetLayerInfo } from "../../data/demoData";

export const UNetArchitectureSection: React.FC = () => {
  const [hoveredLayer, setHoveredLayer] = useState<UNetLayerInfo | null>(UNET_LAYERS[0]);

  return (
    <section id="unet-arch" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
            <Cpu className="w-3.5 h-3.5" />
            SECTION 4 — MONAI 3D U-NET ARCHITECTURE
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            FedMedUNet3D Volumetric Segmentation Network
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Deep neural network architecture utilizing MONAI 3D convolutions, residual blocks, batch normalization, and skip connections for multi-class brain tumor extraction.
          </p>
        </div>

        {/* INTERACTIVE U-NET DIAGRAM */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-8">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h3 className="text-lg font-bold text-white">Interactive 3D U-Net Layer Explorer</h3>
              <p className="text-xs text-slate-400">Hover over any stage block to view channel dimensions and spatial resolution.</p>
            </div>
            <span className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-3 py-1 rounded-full border border-cyan-500/20">
              fedmed/core/model.py
            </span>
          </div>

          {/* VISUAL U-SHAPED LAYOUT */}
          <div className="grid grid-cols-1 md:grid-cols-11 gap-2 items-center text-center">
            {UNET_LAYERS.map((layer: UNetLayerInfo, idx) => {
              const isSelected = hoveredLayer?.name === layer.name;
              return (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredLayer(layer)}
                  className={`p-3 rounded-xl border transition-all cursor-pointer ${
                    layer.type === "input"
                      ? "bg-blue-950/40 border-blue-500 text-blue-300"
                      : layer.type === "encoder"
                      ? "bg-indigo-950/40 border-indigo-500 text-indigo-300"
                      : layer.type === "bottleneck"
                      ? "bg-purple-950/60 border-purple-500 text-purple-300 shadow-lg shadow-purple-500/20 scale-105"
                      : layer.type === "decoder"
                      ? "bg-cyan-950/40 border-cyan-500 text-cyan-300"
                      : "bg-emerald-950/40 border-emerald-500 text-emerald-300"
                  } ${isSelected ? "ring-2 ring-white scale-105 shadow-xl" : "opacity-80 hover:opacity-100"}`}
                >
                  <span className="text-[9px] font-bold uppercase block text-slate-400 truncate">{layer.type}</span>
                  <strong className="text-xs font-bold block my-1 truncate">{layer.channels} Ch</strong>
                  <span className="text-[9px] font-mono block text-slate-300 truncate">{layer.resolution}</span>
                </div>
              );
            })}
          </div>

          {/* LAYER SPECIFICATIONS CARD */}
          {hoveredLayer && (
            <div className="p-6 rounded-xl bg-slate-950 border border-slate-800 flex flex-wrap items-start justify-between gap-4">
              <div className="space-y-2 max-w-xl">
                <div className="flex items-center gap-2">
                  <Info className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs font-mono uppercase text-cyan-400 font-bold">{hoveredLayer.type} Stage</span>
                </div>
                <h4 className="text-lg font-bold text-white">{hoveredLayer.name}</h4>
                <p className="text-xs text-slate-300 leading-relaxed">{hoveredLayer.description}</p>
              </div>

              <div className="flex items-center gap-4 text-xs font-mono">
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-500 block uppercase">Shape</span>
                  <strong className="text-cyan-400 text-sm">{hoveredLayer.shape}</strong>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-500 block uppercase">Channels</span>
                  <strong className="text-indigo-400 text-sm">{hoveredLayer.channels}</strong>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
                  <span className="text-[10px] text-slate-500 block uppercase">Resolution</span>
                  <strong className="text-emerald-400 text-sm">{hoveredLayer.resolution}</strong>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
