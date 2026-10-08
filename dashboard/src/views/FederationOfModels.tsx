import React, { useEffect, useState } from "react";
import { Server, ArrowDown, Play, CheckCircle2, Box } from "lucide-react";

interface IncomingModel {
  update_id: string;
  hospital_id: string;
  hospital_name: string;
  model_version: number;
  federation_round: number;
  submission_time: string;
  training_loss: number;
  validation_dice: number;
  sample_count: number;
  status: string;
  eligibility: string;
}

export const FederationOfModels: React.FC = () => {
  const [incomingModels, setIncomingModels] = useState<IncomingModel[]>([]);
  const [globalModel, setGlobalModel] = useState<any>({ current_version: 0, status: "FRESH" });
  const [isAggregating, setIsAggregating] = useState(false);
  const [feedbackMsg, setFeedbackMsg] = useState<string | null>(null);

  const loadData = () => {
    fetch("http://127.0.0.1:8000/api/federation/incoming-models")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setIncomingModels(data);
      })
      .catch(() => {});

    fetch("http://127.0.0.1:8000/api/federation/global-model")
      .then((r) => r.json())
      .then(setGlobalModel)
      .catch(() => {});
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 2500);
    return () => clearInterval(interval);
  }, []);

  const handleStartFederation = async () => {
    if (incomingModels.length === 0) return;
    setIsAggregating(true);
    setFeedbackMsg(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/federation/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: "FedAvg" }),
      });

      if (res.ok) {
        const result = await res.json();
        setFeedbackMsg(`Federation complete! New Global Model v${result.version} created and persisted.`);
        loadData();
      } else {
        const err = await res.json();
        setFeedbackMsg(`Error: ${err.detail}`);
      }
    } catch (e: any) {
      setFeedbackMsg(`Aggregation failed: ${e.message}`);
    } finally {
      setIsAggregating(false);
    }
  };

  const hasModels = incomingModels.length > 0;

  return (
    <div className="p-8 space-y-8 text-slate-200">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-wide">Federation of Models</h2>
          <p className="text-sm text-slate-400">
            Secure aggregation of incoming local model weights into the next global model version
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <div className="bg-navy-800 border border-navy-700 px-4 py-2 rounded-lg text-xs font-mono">
            <span className="text-slate-400 mr-2">Current Model:</span>
            <span className="font-bold text-white">
              {globalModel.current_version === 0 ? "FRESH" : `v${globalModel.current_version}`}
            </span>
          </div>

          <button
            onClick={handleStartFederation}
            disabled={!hasModels || isAggregating}
            className={`px-5 py-2.5 rounded-lg text-xs font-bold flex items-center space-x-2 transition shadow-sm ${
              hasModels && !isAggregating
                ? "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/30"
                : "bg-navy-700 text-slate-500 cursor-not-allowed"
            }`}
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isAggregating ? "Aggregating Weights..." : "Start Federation"}</span>
          </button>
        </div>
      </div>

      {feedbackMsg && (
        <div className="p-3.5 rounded-lg bg-emerald-950/40 border border-emerald-500/40 text-emerald-200 text-xs">
          {feedbackMsg}
        </div>
      )}

      {/* Visual Aggregation Flow */}
      <div className="bg-navy-800 border border-navy-700 rounded-xl p-8 space-y-8 shadow-sm">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 text-center">
          Incoming Model Updates & Weight Aggregator
        </h3>

        {/* Incoming Model Cards or Empty State */}
        {!hasModels ? (
          <div className="py-10 text-center space-y-3">
            <div className="w-12 h-12 bg-navy-900 border border-navy-700 rounded-full flex items-center justify-center mx-auto text-slate-500">
              <Box className="w-6 h-6" />
            </div>
            <h4 className="text-base font-bold text-white">No Model Updates Received</h4>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              Hospitals must finish local training and submit model updates before federation can be executed.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {incomingModels.map((m) => (
              <div
                key={m.update_id}
                className="bg-navy-900 border border-navy-700 rounded-xl p-4 space-y-3"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-bold text-white text-sm">{m.hospital_name}</h4>
                    <span className="text-[11px] font-mono text-slate-400">{m.hospital_id}</span>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    ELIGIBLE
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
                  <div>
                    <span className="text-slate-500 block">Training Loss</span>
                    <span className="font-mono text-slate-200 font-semibold">{m.training_loss}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Validation Dice</span>
                    <span className="font-mono text-emerald-300 font-semibold">{m.validation_dice}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Local Samples</span>
                    <span className="font-mono text-slate-200">{m.sample_count} vols</span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">Submission</span>
                    <span className="font-mono text-slate-400 truncate block">
                      {new Date(m.submission_time).toLocaleTimeString()}
                    </span>
                  </div>
                </div>

                <div className="flex items-center space-x-1.5 text-[11px] text-blue-300 pt-2 border-t border-navy-800">
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
                  <span>Weights ready for FedAvg aggregation</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Central Aggregator Box */}
        <div className="flex flex-col items-center space-y-4">
          <ArrowDown className={`w-6 h-6 ${hasModels ? "text-blue-400 animate-bounce" : "text-slate-600"}`} />

          <div className="bg-navy-900 border border-navy-600 rounded-xl p-6 w-full max-w-xl text-center space-y-3 shadow-lg">
            <Server className="w-10 h-10 text-blue-400 mx-auto" />
            <h4 className="text-lg font-bold text-white">Central Aggregator (FedAvg)</h4>
            <p className="text-xs text-slate-400">
              Aggregates parameter tensors from approved hospital nodes to produce the next Global Model.
            </p>

            <div className="grid grid-cols-2 gap-4 pt-3 border-t border-navy-800 text-xs">
              <div>
                <span className="text-slate-500 block">Participating Updates</span>
                <span className="text-white font-bold text-sm">{incomingModels.length}</span>
              </div>
              <div>
                <span className="text-slate-500 block">Target Output Version</span>
                <span className="text-emerald-400 font-bold text-sm">
                  v{globalModel.current_version + 1}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
