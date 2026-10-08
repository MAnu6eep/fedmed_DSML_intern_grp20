import React, { useEffect, useState } from "react";
import { 
  Users, 
  Plus, 
  Network, 
  Trash2, 
  ExternalLink, 
  X, 
  RefreshCw,
  Box
} from "lucide-react";

interface HospitalNode {
  hospital_id: string;
  name: string;
  type: string;
  host: string;
  port: number;
  grpc_port: number;
  endpoint: string;
  auth_token: string;
  registration_status: string;
  connection_status: string;
  approval_status: string;
  participation_status: string;
  model_status: string;
  sample_count: number;
  last_heartbeat: string | null;
  current_round: number;
}

interface HospitalRegistryProps {
  onOpenHospital: (hospitalId: string) => void;
  onNavigateToSimulate?: () => void;
}

export const HospitalRegistry: React.FC<HospitalRegistryProps> = ({
  onOpenHospital,
  onNavigateToSimulate,
}) => {
  const [hospitals, setHospitals] = useState<HospitalNode[]>([]);
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [loadingAction, setLoadingAction] = useState<string | null>(null);

  // Form fields
  const [formName, setFormName] = useState("");
  const [formId, setFormId] = useState("");
  const [formHost, setFormHost] = useState("127.0.0.1");
  const [formPort, setFormPort] = useState(8081);
  const [formType, setFormType] = useState("remote");
  const [formToken, setFormToken] = useState("");

  const loadHospitals = () => {
    fetch("http://127.0.0.1:8000/api/hospitals")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setHospitals(data);
      })
      .catch(() => {});
  };

  useEffect(() => {
    loadHospitals();
    const interval = setInterval(loadHospitals, 2500);
    return () => clearInterval(interval);
  }, []);

  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formName.trim()) return;

    setLoadingAction("registering");
    try {
      const res = await fetch("http://127.0.0.1:8000/api/hospitals/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formName.trim(),
          hospital_id: formId.trim() || undefined,
          host: formHost.trim(),
          port: Number(formPort),
          type: formType,
          auth_token: formToken.trim() || undefined,
        }),
      });
      if (res.ok) {
        setShowRegisterModal(false);
        setFormName("");
        setFormId("");
        loadHospitals();
      } else {
        const err = await res.json();
        alert(err.detail || "Registration failed");
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingAction(null);
    }
  };

  const handleApprove = async (hid: string) => {
    setLoadingAction(`approve-${hid}`);
    try {
      await fetch(`http://127.0.0.1:8000/api/hospitals/${hid}/approve`, { method: "POST" });
      loadHospitals();
    } finally {
      setLoadingAction(null);
    }
  };

  const handleReject = async (hid: string) => {
    setLoadingAction(`reject-${hid}`);
    try {
      await fetch(`http://127.0.0.1:8000/api/hospitals/${hid}/reject`, { method: "POST" });
      loadHospitals();
    } finally {
      setLoadingAction(null);
    }
  };

  const handleConnect = async (hid: string) => {
    setLoadingAction(`connect-${hid}`);
    try {
      await fetch(`http://127.0.0.1:8000/api/hospitals/${hid}/connect`, { method: "POST" });
      loadHospitals();
    } finally {
      setLoadingAction(null);
    }
  };

  const handleDelete = async (hid: string) => {
    if (!confirm(`Are you sure you want to remove hospital ${hid}?`)) return;
    setLoadingAction(`delete-${hid}`);
    try {
      await fetch(`http://127.0.0.1:8000/api/hospitals/${hid}`, { method: "DELETE" });
      loadHospitals();
    } finally {
      setLoadingAction(null);
    }
  };

  return (
    <div className="p-8 space-y-8 text-slate-200">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-wide">Hospital Registry</h2>
          <p className="text-sm text-slate-400">
            Centrally manage remote hospital authorization, network identities, and federation quorum
          </p>
        </div>

        <div className="flex items-center space-x-3">
          {onNavigateToSimulate && (
            <button
              onClick={onNavigateToSimulate}
              className="flex items-center space-x-2 bg-navy-800 hover:bg-navy-700 text-slate-300 px-4 py-2 rounded-lg border border-navy-700 text-xs font-semibold transition"
            >
              <Box className="w-4 h-4 text-blue-400" />
              <span>Simulate Remote Hospital</span>
            </button>
          )}

          <button
            onClick={() => setShowRegisterModal(true)}
            className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-xs font-semibold transition shadow-sm"
          >
            <Plus className="w-4 h-4" />
            <span>Register Hospital</span>
          </button>
        </div>
      </div>

      {/* Hospital Table or Empty State */}
      <div className="bg-navy-800 border border-navy-700 rounded-xl overflow-hidden shadow-sm">
        {hospitals.length === 0 ? (
          <div className="p-12 text-center space-y-4">
            <div className="w-14 h-14 bg-navy-900 border border-navy-700 rounded-full flex items-center justify-center mx-auto text-slate-400">
              <Users className="w-7 h-7" />
            </div>
            <h3 className="text-lg font-bold text-white">No Hospitals Registered</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              There are currently zero hospitals enrolled in the federation. Register a remote hospital node to configure identities and request approval.
            </p>
            <button
              onClick={() => setShowRegisterModal(true)}
              className="bg-blue-600 hover:bg-blue-500 text-white px-5 py-2.5 rounded-lg text-xs font-semibold inline-flex items-center space-x-1.5 transition"
            >
              <Plus className="w-4 h-4" />
              <span>Register Hospital</span>
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-navy-900/60 text-slate-400 uppercase text-[11px] font-semibold border-b border-navy-700">
                <tr>
                  <th className="px-5 py-4">Hospital Node</th>
                  <th className="px-4 py-4">Endpoint / Type</th>
                  <th className="px-4 py-4">Connection</th>
                  <th className="px-4 py-4">Approval</th>
                  <th className="px-4 py-4">Participation</th>
                  <th className="px-4 py-4">Model State</th>
                  <th className="px-4 py-4">Volumes</th>
                  <th className="px-5 py-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-700/60">
                {hospitals.map((h) => {
                  const isApproved = h.approval_status === "APPROVED";
                  const isPending = h.approval_status === "PENDING";
                  const isConnected = h.connection_status === "CONNECTED";

                  return (
                    <tr key={h.hospital_id} className="hover:bg-navy-700/20 transition-colors">
                      {/* Name & ID */}
                      <td className="px-5 py-4">
                        <div className="font-bold text-white text-sm">{h.name}</div>
                        <div className="text-[11px] text-slate-400 font-mono">{h.hospital_id}</div>
                      </td>

                      {/* Endpoint & Type */}
                      <td className="px-4 py-4">
                        <div className="font-mono text-[11px] text-blue-300">{h.endpoint}</div>
                        <span className="inline-block mt-0.5 px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-navy-900 text-slate-400 border border-navy-700">
                          {h.type}
                        </span>
                      </td>

                      {/* Connection */}
                      <td className="px-4 py-4">
                        <span
                          className={`inline-flex items-center space-x-1 px-2.5 py-1 rounded text-[11px] font-semibold ${
                            isConnected
                              ? "bg-emerald-500/20 text-emerald-400"
                              : "bg-slate-700/40 text-slate-400"
                          }`}
                        >
                          <span
                            className={`w-1.5 h-1.5 rounded-full mr-1 ${
                              isConnected ? "bg-emerald-400" : "bg-slate-400"
                            }`}
                          />
                          {h.connection_status}
                        </span>
                      </td>

                      {/* Approval Status */}
                      <td className="px-4 py-4">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-bold ${
                            isApproved
                              ? "bg-emerald-500/20 text-emerald-300"
                              : isPending
                              ? "bg-amber-500/20 text-amber-300"
                              : "bg-red-500/20 text-red-300"
                          }`}
                        >
                          {h.approval_status}
                        </span>
                      </td>

                      {/* Participation */}
                      <td className="px-4 py-4">
                        <span className="font-mono text-slate-300 font-medium">
                          {h.participation_status}
                        </span>
                      </td>

                      {/* Model State */}
                      <td className="px-4 py-4">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase ${
                            h.model_status === "SUBMITTED"
                              ? "bg-purple-500/20 text-purple-300"
                              : h.model_status === "TRAINED"
                              ? "bg-blue-500/20 text-blue-300"
                              : h.model_status === "READY"
                              ? "bg-emerald-500/20 text-emerald-300"
                              : "bg-slate-700/30 text-slate-400"
                          }`}
                        >
                          {h.model_status}
                        </span>
                      </td>

                      {/* Samples */}
                      <td className="px-4 py-4 font-mono text-slate-300">
                        {h.sample_count} vols
                      </td>

                      {/* Actions */}
                      <td className="px-5 py-4 text-right">
                        <div className="flex items-center justify-end space-x-2">
                          {/* Approve / Reject buttons if pending */}
                          {isPending && (
                            <>
                              <button
                                onClick={() => handleApprove(h.hospital_id)}
                                disabled={!!loadingAction}
                                className="bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-300 border border-emerald-500/40 px-2 py-1 rounded text-[11px] font-semibold transition"
                                title="Approve hospital for federation"
                              >
                                Approve
                              </button>
                              <button
                                onClick={() => handleReject(h.hospital_id)}
                                disabled={!!loadingAction}
                                className="bg-red-600/30 hover:bg-red-600/50 text-red-300 border border-red-500/40 px-2 py-1 rounded text-[11px] font-semibold transition"
                                title="Reject hospital"
                              >
                                Reject
                              </button>
                            </>
                          )}

                          {/* Connect button if offline */}
                          {!isConnected && (
                            <button
                              onClick={() => handleConnect(h.hospital_id)}
                              disabled={!!loadingAction}
                              className="text-blue-400 hover:text-blue-300 p-1 rounded hover:bg-navy-700"
                              title="Establish Connection"
                            >
                              <Network className="w-4 h-4" />
                            </button>
                          )}

                          {/* Open Hospital-Side UI */}
                          <button
                            onClick={() => onOpenHospital(h.hospital_id)}
                            className="bg-blue-600 hover:bg-blue-500 text-white px-2.5 py-1 rounded text-[11px] font-semibold flex items-center space-x-1 transition shadow-sm"
                            title="Open private hospital environment"
                          >
                            <span>Open</span>
                            <ExternalLink className="w-3 h-3" />
                          </button>

                          {/* Delete */}
                          <button
                            onClick={() => handleDelete(h.hospital_id)}
                            disabled={!!loadingAction}
                            className="text-slate-500 hover:text-red-400 p-1 rounded hover:bg-navy-700 transition"
                            title="Remove hospital"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Register Hospital Modal */}
      {showRegisterModal && (
        <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4">
          <div className="bg-navy-800 border border-navy-700 rounded-xl max-w-lg w-full p-6 shadow-2xl relative space-y-5">
            <div className="flex justify-between items-center border-b border-navy-700 pb-3">
              <div>
                <h3 className="text-lg font-bold text-white">Register Hospital Node</h3>
                <p className="text-xs text-slate-400">Enroll a remote or local hospital client in FedMed</p>
              </div>
              <button
                onClick={() => setShowRegisterModal(false)}
                className="text-slate-400 hover:text-white p-1"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleRegisterSubmit} className="space-y-4 text-xs">
              <div>
                <label className="block text-slate-300 font-semibold mb-1">
                  Hospital Name <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Johns Hopkins Neuro Imaging"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-300 font-semibold mb-1">Hospital ID (Optional)</label>
                  <input
                    type="text"
                    placeholder="e.g. johns_hopkins"
                    value={formId}
                    onChange={(e) => setFormId(e.target.value)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-slate-300 font-semibold mb-1">Hospital Environment</label>
                  <select
                    value={formType}
                    onChange={(e) => setFormType(e.target.value)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                  >
                    <option value="remote">Remote Hospital</option>
                    <option value="docker">Docker Hospital (Demo)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-300 font-semibold mb-1">Host / IP</label>
                  <input
                    type="text"
                    value={formHost}
                    onChange={(e) => setFormHost(e.target.value)}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-slate-300 font-semibold mb-1">Port</label>
                  <input
                    type="number"
                    value={formPort}
                    onChange={(e) => setFormPort(Number(e.target.value))}
                    className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-300 font-semibold mb-1">Authentication Secret (Optional)</label>
                <input
                  type="text"
                  placeholder="Auto-generated secure token if left blank"
                  value={formToken}
                  onChange={(e) => setFormToken(e.target.value)}
                  className="w-full bg-navy-900 border border-navy-700 rounded-lg px-3 py-2 text-white font-mono focus:outline-none focus:border-blue-500"
                />
              </div>

              <div className="pt-3 border-t border-navy-700 flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowRegisterModal(false)}
                  className="px-4 py-2 rounded-lg bg-navy-900 text-slate-400 hover:text-white transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loadingAction === "registering"}
                  className="px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-bold flex items-center space-x-1.5 transition"
                >
                  {loadingAction === "registering" && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                  <span>Register Hospital</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
