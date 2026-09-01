import { create } from "zustand";
import { getHospitals } from "../api/client";
import type { Hospital } from "../api/client";

interface HospitalStore {
  hospitals: Hospital[];
  loading: boolean;
  error: string | null;

  fetchHospitals: () => Promise<void>;

  updateHospitalStatus: (
    hospitalId: string,
    status: Hospital["status"]
  ) => void;
}

const MOCK_HOSPITALS: Hospital[] = [
  {
    hospital_id: "hospital_a",
    name: "General Hospital Neuro",
    status: "online",
    port: 8081,
    samples: 150,
    loss: 0.2841,
    dice: 0.912,
  },
  {
    hospital_id: "hospital_b",
    name: "St. Jude Imaging",
    status: "training",
    port: 8082,
    samples: 120,
    loss: 0.3152,
    dice: 0.887,
  },
  {
    hospital_id: "hospital_c",
    name: "Metro Health Oncology",
    status: "online",
    port: 8083,
    samples: 180,
    loss: 0.2617,
    dice: 0.924,
  },
];

export const useHospitalStore = create<HospitalStore>((set) => ({
  hospitals: MOCK_HOSPITALS,
  loading: false,
  error: null,

  fetchHospitals: async () => {
    set({
      loading: true,
      error: null,
    });

    try {
      const hospitals = await getHospitals();

      set({
        hospitals,
        loading: false,
      });
    } catch {
      set({
        hospitals: MOCK_HOSPITALS,
        loading: false,
        error: "Backend unavailable. Using mock telemetry.",
      });
    }
  },

  updateHospitalStatus: (hospitalId, status) => {
    set((state) => ({
      hospitals: state.hospitals.map((hospital) =>
        hospital.hospital_id === hospitalId
          ? {
              ...hospital,
              status,
            }
          : hospital
      ),
    }));
  },
}));
