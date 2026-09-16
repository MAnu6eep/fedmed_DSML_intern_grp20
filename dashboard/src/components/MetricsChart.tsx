import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

import type { FederationMetricPoint } from "../types/metrics";

interface MetricsChartProps {
  data: FederationMetricPoint[];
}

export const MetricsChart: React.FC<MetricsChartProps> = ({ data }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-white">
          Federated Training Metrics
        </h3>

        <p className="text-sm text-slate-400">
          Live training metrics across federation rounds
        </p>
      </div>

      <div className="w-full h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />

            <XAxis
              dataKey="round"
              stroke="#94a3b8"
              label={{
                value: "Federation Round",
                position: "insideBottom",
                offset: -5,
              }}
            />

            <YAxis
              stroke="#94a3b8"
              domain={[0, 1]}
            />

            <Tooltip />
            <Legend />

            <Line
              type="monotone"
              dataKey="trainingLoss"
              name="Training Loss"
              stroke="#ef4444"
              strokeWidth={2}
              dot
              connectNulls={false}
            />

            <Line
              type="monotone"
              dataKey="validationLoss"
              name="Validation Loss"
              stroke="#f97316"
              strokeWidth={2}
              dot
              connectNulls={false}
            />

            <Line
              type="monotone"
              dataKey="diceScore"
              name="Dice Score"
              stroke="#22c55e"
              strokeWidth={2}
              dot
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};