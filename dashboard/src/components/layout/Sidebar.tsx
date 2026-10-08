import React, { useState } from "react";
import { 
  Activity, 
  Users, 
  Box, 
  Server, 
  Database, 
  History, 
  FlaskConical, 
  ChevronLeft, 
  ChevronRight,
  Info 
} from "lucide-react";
import type { ViewState } from "../../App";

interface SidebarProps {
  currentView: ViewState;
  executionMode: "standby" | "docker";
  onViewChange: (view: ViewState) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  currentView, 
  executionMode, 
  onViewChange 
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [width, setWidth] = useState(260); // Resizable width in px

  const navItems = [
    { id: "dashboard", label: "Central Dashboard", icon: Activity },
    { id: "registry", label: "Hospital Registry", icon: Users },
    { id: "simulate_remote", label: "Simulate Remote Hospital", icon: Box },
    { id: "federation", label: "Federation of Models", icon: Server },
    { id: "model_registry", label: "Global Model Registry", icon: Database },
    { id: "activity", label: "Hospital Activity", icon: History },
    // Testing & Experiments at the VERY END as strictly required
    { id: "testing_experiments", label: "Testings & Experimenting", icon: FlaskConical },
  ];

  const isDockerMode = executionMode === "docker";

  return (
    <div
      style={{ width: isCollapsed ? 76 : width }}
      className="bg-navy-800 border-r border-navy-700 flex flex-col h-full relative transition-[width] duration-150 select-none flex-shrink-0"
    >
      {/* Header */}
      <div className="p-4 border-b border-navy-700/60 flex items-center justify-between overflow-hidden">
        {!isCollapsed && (
          <div className="min-w-0 pr-2">
            <h1 className="text-xl font-bold text-white tracking-wide truncate">FedMed</h1>
            <p className="text-[11px] text-beige-100 opacity-70 uppercase tracking-wider truncate">
              Central Control Hub
            </p>
          </div>
        )}
        {isCollapsed && (
          <div className="w-full flex justify-center py-1">
            <span className="font-extrabold text-base text-white tracking-widest">FM</span>
          </div>
        )}

        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-navy-700 transition flex-shrink-0"
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation Items (Single line, strictly ordered) */}
      <nav className="flex-1 px-2.5 py-4 space-y-1.5 overflow-y-auto overflow-x-hidden">
        {navItems.map((item) => {
          const isActive = currentView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id as ViewState)}
              title={isCollapsed ? item.label : undefined}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-colors whitespace-nowrap text-left ${
                isActive
                  ? "bg-navy-700 text-white font-semibold shadow-sm"
                  : "text-slate-300 hover:bg-navy-700/60 hover:text-white font-medium"
              }`}
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              {!isCollapsed && (
                <span className="text-xs truncate tracking-wide">{item.label}</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* About Project Button (Bottom of Sidebar) */}
      <div className="px-2.5 py-2 border-t border-navy-700/60">
        <button
          onClick={() => onViewChange("about")}
          title={isCollapsed ? "About Project (Overview & Architecture)" : undefined}
          className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors whitespace-nowrap text-left ${
            currentView === "about"
              ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
              : "text-cyan-400 hover:bg-cyan-500/10 hover:text-cyan-300 font-medium"
          }`}
        >
          <Info className="w-4 h-4 flex-shrink-0" />
          {!isCollapsed && (
            <span className="text-xs truncate tracking-wide">About Project</span>
          )}
        </button>
      </div>

      {/* Footer & Mode Indicator */}
      {!isCollapsed ? (
        <div className="p-3 border-t border-navy-700/60 space-y-1.5">
          <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">
            Execution Mode
          </div>
          <div
            className={`flex items-center space-x-2 px-2.5 py-1.5 rounded-lg text-[11px] font-bold ${
              isDockerMode
                ? "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full flex-shrink-0 ${
                isDockerMode ? "bg-blue-400" : "bg-emerald-400"
              }`}
            />
            <span className="truncate">
              {isDockerMode ? "Docker Simulation" : "Standby / Native Mode"}
            </span>
          </div>
          <div className="text-[10px] text-slate-500 text-center pt-0.5 truncate">
            FedMed Central Machine
          </div>
        </div>
      ) : (
        <div className="p-2 text-center border-t border-navy-700/60">
          <span
            className={`inline-block w-2.5 h-2.5 rounded-full ${
              isDockerMode ? "bg-blue-400" : "bg-emerald-400"
            }`}
            title={`Execution Mode: ${
              isDockerMode ? "Docker Remote Hospital Simulation" : "Standby / Native Mode"
            }`}
          />
        </div>
      )}

      {/* Drag Resize Handle (right border) */}
      {!isCollapsed && (
        <div
          onMouseDown={(e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startWidth = width;
            const onMouseMove = (moveEvent: MouseEvent) => {
              const newWidth = Math.max(220, Math.min(420, startWidth + (moveEvent.clientX - startX)));
              setWidth(newWidth);
            };
            const onMouseUp = () => {
              window.removeEventListener("mousemove", onMouseMove);
              window.removeEventListener("mouseup", onMouseUp);
            };
            window.addEventListener("mousemove", onMouseMove);
            window.addEventListener("mouseup", onMouseUp);
          }}
          className="absolute right-0 top-0 bottom-0 w-1.5 hover:w-2 hover:bg-blue-500/50 cursor-col-resize transition-colors"
          title="Drag to resize sidebar width"
        />
      )}
    </div>
  );
};
