import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 5000,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface HealthResponse {
  status: string;
  timestamp: string;
  active_strategy: string;
  nodes_connected: number;
}

export interface Hospital {
  hospital_id: string;
  name: string;
  status: "online" | "training" | "offline" | "error";
  port: number;
  samples: number;
  loss: number;
  dice: number;
}

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>("/api/health");
  return response.data;
};

export const getHospitals = async (): Promise<Hospital[]> => {
  const response = await apiClient.get<Hospital[]>("/api/hospitals");
  return response.data;
};

export const getMetrics = async () => {
  const response = await apiClient.get("/api/metrics");
  return response.data;
};

export default apiClient;