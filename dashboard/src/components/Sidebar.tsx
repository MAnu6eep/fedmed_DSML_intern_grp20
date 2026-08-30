
import {
  Activity,
  ShieldCheck,
  Server,
  Settings,
  Database,
} from "lucide-react";

export const Sidebar: React.FC = () => {
  const navItems = [
    { icon: Activity, label: "Overview", active: true },
    { icon: Server, label: "Hospital Nodes", active: false },
    { icon: ShieldCheck, label: "Privacy & Security", active: false },
    { icon: Database, label: "Dataset & Baselines", active: false },
    { icon: Settings, label: "Configuration", active: false },
  ];

  return (
    <aside className="w-64 bg-slate-900 text-slate-200 min-h-screen p-4 flex flex-col border-r border-slate-800">
      <div className="flex items-center gap-3 px-2 py-4 mb-6 border-b border-slate-800">
        <div className="bg-emerald-500/20 p-2 rounded-lg text-emerald-400 font-bold text-xl">
          FM
        </div>

        <div>
          <h1 className="font-semibold text-white">
            FedMed Core
          </h1>
          <p className="text-xs text-slate-400">
            PPML Medical Vision
          </p>
        </div>
      </div>

      <nav className="space-y-1">
        {navItems.map((item, idx) => {
          const Icon = item.icon;

          return (
            <button
              key={idx}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                item.active
                  ? "bg-emerald-600 text-white"
                  : "text-slate-400 hover:bg-slate-800 hover:text-white"
              }`}
            >
              <Icon className="w-4 h-4" />
              {item.label}
            </button>
          );
        })}
      </nav>
    </aside>
  );
};