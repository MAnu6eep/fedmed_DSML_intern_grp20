import React, { useState } from "react";
import { ShieldCheck, Key, Lock, EyeOff } from "lucide-react";
import { PRIVACY_SPECS } from "../../data/demoData";

export const SecureCommSection: React.FC = () => {
  const [activePrivacyTab, setActivePrivacyTab] = useState<"tenseal" | "secagg" | "dp">("tenseal");

  return (
    <section id="secure-comm" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <ShieldCheck className="w-3.5 h-3.5" />
            SECTION 6 — SECURE COMMUNICATION & PRIVACY LAYERS
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Homomorphic Encryption, SecAgg+, and DP-SGD
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Model parameter updates leaving hospital boundaries are protected by three layered privacy guarantees implemented in the codebase.
          </p>
        </div>

        {/* PRIVACY TAB SWITCHER */}
        <div className="flex justify-center">
          <div className="bg-slate-900 p-1 rounded-xl border border-slate-800 flex flex-wrap gap-2">
            <button
              onClick={() => setActivePrivacyTab("tenseal")}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs font-bold transition-all ${
                activePrivacyTab === "tenseal"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-lg shadow-cyan-500/10"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Key className="w-4 h-4 text-cyan-400" />
              1. TenSEAL CKKS Encryption
            </button>
            <button
              onClick={() => setActivePrivacyTab("secagg")}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs font-bold transition-all ${
                activePrivacyTab === "secagg"
                  ? "bg-blue-500/20 text-blue-300 border border-blue-500/40 shadow-lg shadow-blue-500/10"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Lock className="w-4 h-4 text-blue-400" />
              2. SecAgg+ Secret Sharing
            </button>
            <button
              onClick={() => setActivePrivacyTab("dp")}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs font-bold transition-all ${
                activePrivacyTab === "dp"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-lg shadow-purple-500/10"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <EyeOff className="w-4 h-4 text-purple-400" />
              3. Differential Privacy (DP-SGD)
            </button>
          </div>
        </div>

        {/* DETAILS CARD */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          {activePrivacyTab === "tenseal" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-xl font-bold text-cyan-400 flex items-center gap-2">
                  <Key className="w-5 h-5" />
                  TenSEAL CKKS Homomorphic Encryption Engine
                </h3>
                <span className="text-xs font-mono text-cyan-400">fedmed/privacy/tenseal_engine.py</span>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">
                Encrypts floating-point model weights using the CKKS homomorphic scheme. Allows the server to perform vector addition directly on ciphertexts without secret keys.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Poly Degree</span>
                  <strong className="text-base text-white font-mono">{PRIVACY_SPECS.tensealCKKS.polyModulusDegree}</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Global Scale</span>
                  <strong className="text-base text-cyan-400 font-mono">{PRIVACY_SPECS.tensealCKKS.globalScale}</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Coeff Bit Sizes</span>
                  <strong className="text-base text-indigo-400 font-mono">[60, 40, 40, 60]</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Strategy</span>
                  <strong className="text-xs text-emerald-400 font-mono">{PRIVACY_SPECS.tensealCKKS.selectiveEncryption}</strong>
                </div>
              </div>
            </div>
          )}

          {activePrivacyTab === "secagg" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-xl font-bold text-blue-400 flex items-center gap-2">
                  <Lock className="w-5 h-5" />
                  SecAgg+ Zero-Sum Pairwise Masking Protocol
                </h3>
                <span className="text-xs font-mono text-blue-400">fedmed/privacy/secagg_config.py</span>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">
                Applies Shamir secret sharing and zero-sum pairwise noise masks across hospital nodes. Individual client updates are cryptographically unreadable; masks cancel out upon server aggregation.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Threshold (k/n)</span>
                  <strong className="text-base text-white font-mono">{PRIVACY_SPECS.secaggPlus.threshold} / {PRIVACY_SPECS.secaggPlus.totalClients} Hospitals</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Quantization</span>
                  <strong className="text-base text-blue-400 font-mono">{PRIVACY_SPECS.secaggPlus.quantizationBits}-bit Integer</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Modulus Range</span>
                  <strong className="text-base text-indigo-400 font-mono">{PRIVACY_SPECS.secaggPlus.modulusRange}</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Clipping Bound</span>
                  <strong className="text-base text-emerald-400 font-mono">{PRIVACY_SPECS.secaggPlus.clippingBound}</strong>
                </div>
              </div>
            </div>
          )}

          {activePrivacyTab === "dp" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-xl font-bold text-purple-400 flex items-center gap-2">
                  <EyeOff className="w-5 h-5" />
                  Differential Privacy (DP-SGD & Gaussian Noise)
                </h3>
                <span className="text-xs font-mono text-purple-400">fedmed/privacy/differential_privacy.py</span>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">
                Clips gradient norm to L2 threshold and injects calibrated Gaussian noise to prevent membership inference and training data extraction attacks.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">L2 Max Grad Norm</span>
                  <strong className="text-base text-white font-mono">{PRIVACY_SPECS.differentialPrivacy.maxGradNorm}</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Noise Multiplier</span>
                  <strong className="text-base text-purple-400 font-mono">{PRIVACY_SPECS.differentialPrivacy.noiseMultiplier}</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Privacy Budget (ε)</span>
                  <strong className="text-base text-indigo-400 font-mono">{PRIVACY_SPECS.differentialPrivacy.epsilon}</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-[10px] text-slate-500 uppercase block font-bold">Delta (δ)</span>
                  <strong className="text-base text-emerald-400 font-mono">{PRIVACY_SPECS.differentialPrivacy.delta}</strong>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
