/**
 * demoData.ts
 *
 * Central data repository for the FedMed Interactive Demo & Visual Explanation System.
 * Maps exact repository architecture, parameters, benchmark results, and illustrative demo data.
 */

export interface HospitalNodeInfo {
  id: string;
  name: string;
  host: string;
  port: number;
  samples: number;
  status: "online" | "training" | "offline" | "aggregating";
  datasetName: string;
  tumorDistribution: {
    enhancingTumor: number; // ET %
    tumorCore: number;      // TC %
    wholeTumor: number;     // WT %
  };
  loss: number;
  dice: number;
}

export interface UNetLayerInfo {
  name: string;
  type: "input" | "encoder" | "bottleneck" | "decoder" | "output";
  shape: string;
  channels: number;
  resolution: string;
  description: string;
}

export interface PrivacyConfigInfo {
  tensealCKKS: {
    polyModulusDegree: number;
    coeffModBitSizes: number[];
    globalScale: string;
    scheme: string;
    selectiveEncryption: string;
  };
  secaggPlus: {
    threshold: number;
    totalClients: number;
    quantizationBits: number;
    modulusRange: string;
    clippingBound: number;
  };
  differentialPrivacy: {
    enabled: boolean;
    maxGradNorm: number;
    noiseMultiplier: number;
    epsilon: number;
    delta: string;
  };
}

export interface StrategyInfo {
  id: "fedavg" | "fedprox" | "scaffold";
  name: string;
  fullName: string;
  description: string;
  formula: string;
  hyperparameters: Record<string, string | number>;
  useCase: string;
}

export interface BenchmarkResult {
  numClients: number;
  plaintextFedAvgTimeSec: number;
  plaintextFedAvgPayloadKB: number;
  secaggPlusTimeSec: number;
  secaggPlusPayloadKB: number;
  communicationOverheadPercent: number;
  timeOverheadPercent: number;
}

// ------------------------------------------------------------------
// 1. HOSPITAL NODES (Matches hospitals/ folder & API)
// ------------------------------------------------------------------
export const HOSPITALS_DATA: HospitalNodeInfo[] = [
  {
    id: "hospital_a",
    name: "General Hospital Neuro",
    host: "hospital-a",
    port: 8081,
    samples: 150,
    status: "online",
    datasetName: "BraTS2020_Neuro_CohortA",
    tumorDistribution: {
      enhancingTumor: 35,
      tumorCore: 45,
      wholeTumor: 85,
    },
    loss: 0.2841,
    dice: 0.912,
  },
  {
    id: "hospital_b",
    name: "St. Jude Imaging",
    host: "hospital-b",
    port: 8082,
    samples: 120,
    status: "training",
    datasetName: "BraTS2020_StJude_CohortB",
    tumorDistribution: {
      enhancingTumor: 20,
      tumorCore: 35,
      wholeTumor: 70,
    },
    loss: 0.3152,
    dice: 0.887,
  },
  {
    id: "hospital_c",
    name: "Metro Health Oncology",
    host: "hospital-c",
    port: 8083,
    samples: 180,
    status: "online",
    datasetName: "BraTS2020_Metro_CohortC",
    tumorDistribution: {
      enhancingTumor: 50,
      tumorCore: 60,
      wholeTumor: 92,
    },
    loss: 0.2617,
    dice: 0.924,
  },
];

// ------------------------------------------------------------------
// 2. MONAI PREPROCESSING STEPS (Matches fedmed/data/loader.py)
// ------------------------------------------------------------------
export const PREPROCESSING_PIPELINE = [
  {
    step: 1,
    name: "LoadImaged",
    title: "Raw NIfTI Volume Ingestion",
    description: "Loads 4-channel 3D NIfTI (.nii.gz) MRI volumes (FLAIR, T1, T1ce, T2).",
    outputShape: "155 × 240 × 240 (Raw Voxels)",
  },
  {
    step: 2,
    name: "Orientationd",
    title: "Anatomical Reorientation (RAS)",
    description: "Standardizes 3D spatial axes to Right-Anterior-Superior orientation.",
    outputShape: "155 × 240 × 240 (RAS Standardized)",
  },
  {
    step: 3,
    name: "Spacingd",
    title: "Isotropic Resampling (1.0mm³)",
    description: "Resamples voxel spacing to uniform 1.0mm × 1.0mm × 1.0mm isotropic grid.",
    outputShape: "1.0mm × 1.0mm × 1.0mm Isotropic",
  },
  {
    step: 4,
    name: "NormalizeIntensityd",
    title: "Z-Score Intensity Normalization",
    description: "Applies zero-mean, unit-variance intensity normalization per channel on non-zero brain voxels.",
    outputShape: "Non-zero Channel-wise Normalized",
  },
  {
    step: 5,
    name: "RandCropByPosNegLabeld",
    title: "Spatial Patch Crop & Tensor Assembly",
    description: "Extracts balanced positive/negative spatial ROI crops into 3D PyTorch tensors.",
    outputShape: "(4, 64, 64, 32) Input Tensor",
  },
];

// ------------------------------------------------------------------
// 3. 3D U-NET LAYERS (Matches fedmed/core/model.py FedMedUNet3D)
// ------------------------------------------------------------------
export const UNET_LAYERS: UNetLayerInfo[] = [
  {
    name: "Input Volumetric Tensor",
    type: "input",
    shape: "(4, 64, 64, 32)",
    channels: 4,
    resolution: "64×64×32",
    description: "4-Channel MRI Tensor: FLAIR, T1, T1ce, T2.",
  },
  {
    name: "Encoder Stage 1 (16 Channels)",
    type: "encoder",
    shape: "(16, 32, 32, 16)",
    channels: 16,
    resolution: "32×32×16",
    description: "3D Convolution + Batch Normalization + PReLU + Residual Unit 1.",
  },
  {
    name: "Encoder Stage 2 (32 Channels)",
    type: "encoder",
    shape: "(32, 16, 16, 8)",
    channels: 32,
    resolution: "16×16×8",
    description: "Stride-2 Downsampling + 3D Feature Extraction.",
  },
  {
    name: "Encoder Stage 3 (64 Channels)",
    type: "encoder",
    shape: "(64, 8, 8, 4)",
    channels: 64,
    resolution: "8×8×4",
    description: "Deep Spatial Context Aggregation + Residual Unit 3.",
  },
  {
    name: "Encoder Stage 4 (128 Channels)",
    type: "encoder",
    shape: "(128, 4, 4, 2)",
    channels: 128,
    resolution: "4×4×2",
    description: "High-level Abstract Feature Encoding.",
  },
  {
    name: "Bottleneck (256 Channels)",
    type: "bottleneck",
    shape: "(256, 2, 2, 1)",
    channels: 256,
    resolution: "2×2×1",
    description: "Deepest Latent Representation (Dropout = 0.1).",
  },
  {
    name: "Decoder Stage 4 + Skip Connection",
    type: "decoder",
    shape: "(128, 4, 4, 2)",
    channels: 128,
    resolution: "4×4×2",
    description: "3D Transpose Conv + Concatenation of Encoder Stage 4 Features.",
  },
  {
    name: "Decoder Stage 3 + Skip Connection",
    type: "decoder",
    shape: "(64, 8, 8, 4)",
    channels: 64,
    resolution: "8×8×4",
    description: "Spatial Up-sampling & High-Resolution Feature Fusion.",
  },
  {
    name: "Decoder Stage 2 + Skip Connection",
    type: "decoder",
    shape: "(32, 16, 16, 8)",
    channels: 32,
    resolution: "16×16×8",
    description: "Multi-scale Reconstruction.",
  },
  {
    name: "Decoder Stage 1 + Skip Connection",
    type: "decoder",
    shape: "(16, 32, 32, 16)",
    channels: 16,
    resolution: "32×32×16",
    description: "Final Spatial Detail Reconstruction.",
  },
  {
    name: "Output Segmentation Mask",
    type: "output",
    shape: "(1, 64, 64, 32)",
    channels: 1,
    resolution: "64×64×32",
    description: "1x1x1 3D Conv -> Voxel-wise Tumor Mask Probability.",
  },
];

// ------------------------------------------------------------------
// 4. PRIVACY & SECURITY SPECS (Matches fedmed/privacy/)
// ------------------------------------------------------------------
export const PRIVACY_SPECS: PrivacyConfigInfo = {
  tensealCKKS: {
    polyModulusDegree: 8192,
    coeffModBitSizes: [60, 40, 40, 60],
    globalScale: "2^40 (1,099,511,627,776)",
    scheme: "CKKS Homomorphic Encryption",
    selectiveEncryption: "Top-k% gradient parameters encrypted; remaining aggregated via plaintext SecAgg+",
  },
  secaggPlus: {
    threshold: 2,
    totalClients: 3,
    quantizationBits: 16,
    modulusRange: "2^31 (2,147,483,648)",
    clippingBound: 10.0,
  },
  differentialPrivacy: {
    enabled: true,
    maxGradNorm: 1.0,
    noiseMultiplier: 1.0,
    epsilon: 2.45,
    delta: "1e-5",
  },
};

// ------------------------------------------------------------------
// 5. FEDERATED STRATEGIES (Matches fedmed/federation/server.py)
// ------------------------------------------------------------------
export const STRATEGIES_DATA: StrategyInfo[] = [
  {
    id: "fedavg",
    name: "FedAvg",
    fullName: "Federated Averaging",
    description: "Standard example-weighted model parameter aggregation.",
    formula: "w_{t+1} = \\sum_{k=1}^K \\frac{n_k}{N} w_{k, t+1}",
    hyperparameters: {
      fraction_fit: 1.0,
      min_fit_clients: 3,
      local_epochs: 1,
    },
    useCase: "Ideal for homogeneous IID hospital data distributions.",
  },
  {
    id: "fedprox",
    name: "FedProx",
    fullName: "Federated Proximal Optimization",
    description: "Adds a proximal regularization term to restrict local model drift under non-IID conditions.",
    formula: "\\min_{w} h_k(w; w^t) = f_k(w) + \\frac{\\mu}{2} \\|w - w^t\\|^2",
    hyperparameters: {
      proximal_mu: 0.01,
      fraction_fit: 1.0,
      min_fit_clients: 3,
    },
    useCase: "Restricts client model divergence when hospitals have heterogeneous scanner hardware.",
  },
  {
    id: "scaffold",
    name: "SCAFFOLD",
    fullName: "Stochastic Controlled Averaging",
    description: "Uses client and server control variates to correct gradient direction bias.",
    formula: "y_{i, k} = y_{i, k-1} - \\eta \\left(g_i(y_{i, k-1}) - c_i + c\\right)",
    hyperparameters: {
      server_learning_rate: 1.0,
      client_learning_rate: 1e-4,
      local_epochs: 1,
    },
    useCase: "Achieves faster convergence and eliminates client drift on extreme Non-IID Dirichlet splits.",
  },
];

// ------------------------------------------------------------------
// 6. REAL BENCHMARK RESULTS (From docs/secagg_benchmark_results.json)
// ------------------------------------------------------------------
export const REAL_BENCHMARK_RESULTS: BenchmarkResult = {
  numClients: 5,
  plaintextFedAvgTimeSec: 0.0105,
  plaintextFedAvgPayloadKB: 200,
  secaggPlusTimeSec: 0.1466,
  secaggPlusPayloadKB: 4970.7,
  communicationOverheadPercent: 2385.36,
  timeOverheadPercent: 1298.25,
};

// ------------------------------------------------------------------
// 7. ILLUSTRATIVE FEDERATED ROUNDS METRICS (Demo / Experiment history)
// ------------------------------------------------------------------
export const DEMO_ROUND_HISTORY = [
  {
    round: 1,
    trainLoss: 1.6563,
    valLoss: 1.6561,
    valDice: 0.2222,
    etDice: 0.1850,
    tcDice: 0.2100,
    wtDice: 0.2716,
    hd95: 18.4,
    payloadSizeMB: 2.8,
    roundTimeSec: 14.2,
    activeHospitals: 3,
  },
  {
    round: 2,
    trainLoss: 1.1240,
    valLoss: 1.0980,
    valDice: 0.4850,
    etDice: 0.4210,
    tcDice: 0.4680,
    wtDice: 0.5660,
    hd95: 12.1,
    payloadSizeMB: 2.8,
    roundTimeSec: 13.8,
    activeHospitals: 3,
  },
  {
    round: 3,
    trainLoss: 0.6840,
    valLoss: 0.6420,
    valDice: 0.7120,
    etDice: 0.6720,
    tcDice: 0.7010,
    wtDice: 0.7630,
    hd95: 8.6,
    payloadSizeMB: 2.8,
    roundTimeSec: 14.1,
    activeHospitals: 3,
  },
  {
    round: 4,
    trainLoss: 0.4120,
    valLoss: 0.3890,
    valDice: 0.8450,
    etDice: 0.8140,
    tcDice: 0.8350,
    wtDice: 0.8860,
    hd95: 5.4,
    payloadSizeMB: 2.8,
    roundTimeSec: 13.9,
    activeHospitals: 3,
  },
  {
    round: 5,
    trainLoss: 0.2840,
    valLoss: 0.2610,
    valDice: 0.9120,
    etDice: 0.8840,
    tcDice: 0.9020,
    wtDice: 0.9350,
    hd95: 4.2,
    payloadSizeMB: 2.8,
    roundTimeSec: 14.5,
    activeHospitals: 3,
  },
];

// ------------------------------------------------------------------
// 8. 14 PIPELINE STAGES FOR PRESENTATION MODE
// ------------------------------------------------------------------
export interface StageConfig {
  id: number;
  slug: string;
  title: string;
  subtitle: string;
  shortDesc: string;
}

export const PIPELINE_STAGES: StageConfig[] = [
  {
    id: 1,
    slug: "problem",
    title: "1. The Privacy Problem",
    subtitle: "Centralized Data Transfer vs. Federated Isolation",
    shortDesc: "Traditional cloud ML forces hospitals to expose sensitive patient 3D MRI scans. FedMed keeps raw data isolated within local firewalls.",
  },
  {
    id: 2,
    slug: "data-layer",
    title: "2. MONAI Data Ingestion",
    subtitle: "Multi-Modal MRI Preprocessing Pipeline",
    shortDesc: "Raw FLAIR, T1, T1ce, and T2 scans are orientation-aligned (RAS), isotropically resampled (1.0mm³), normalized, and cropped into (4,64,64,32) tensors.",
  },
  {
    id: 3,
    slug: "hospital-nodes",
    title: "3. Federated Hospital Nodes",
    subtitle: "Distributed Clinical Sites",
    shortDesc: "Three distinct hospital nodes (Hospital A, B, and C) manage local MRI datasets and maintain isolated local 3D U-Net models.",
  },
  {
    id: 4,
    slug: "unet-arch",
    title: "4. MONAI 3D U-Net Model",
    subtitle: "FedMedUNet3D Neural Network Architecture",
    shortDesc: "Volumetric 3D U-Net featuring 4 encoder levels (16-128 channels), bottleneck (256 channels), skip connections, and voxel-wise tumor segmentation output.",
  },
  {
    id: 5,
    slug: "local-training",
    title: "5. Concurrent Local Training",
    subtitle: "On-Premise Model Parameter Optimization",
    shortDesc: "Hospitals train the 3D U-Net locally for E epochs. Only local model updates (Δw) are generated; raw MRI scans never leave local node boundaries.",
  },
  {
    id: 6,
    slug: "secure-comm",
    title: "6. Secure Communication & Privacy",
    subtitle: "TenSEAL CKKS, SecAgg+, and DP-SGD",
    shortDesc: "Model updates are protected using TenSEAL CKKS homomorphic encryption, SecAgg+ zero-sum secret sharing, and Differential Privacy noise.",
  },
  {
    id: 7,
    slug: "fed-server",
    title: "7. Federated Server & Strategies",
    subtitle: "FedAvg, FedProx, and SCAFFOLD Aggregation",
    shortDesc: "The central server aggregates encrypted/masked parameter updates using weighted averaging (FedAvg), proximal regularization (FedProx), or control variates (SCAFFOLD).",
  },
  {
    id: 8,
    slug: "fed-rounds",
    title: "8. Federated Round Progression",
    subtitle: "Iterative Convergence Timeline",
    shortDesc: "The global model undergoes multiple communication rounds. Each round improves global Dice segmentation score while reducing global loss.",
  },
  {
    id: 9,
    slug: "data-partition",
    title: "9. Data Partitioning",
    subtitle: "IID Split vs. Non-IID Dirichlet Split",
    shortDesc: "Simulates realistic hospital heterogeneity using Dirichlet distribution (α parameter) to benchmark robustness under non-IID clinical distributions.",
  },
  {
    id: 10,
    slug: "evaluation",
    title: "10. Model Evaluation & Metrics",
    subtitle: "Voxel-wise Dice, IoU, and Hausdorff Distance",
    shortDesc: "Evaluates global model performance using Dice Similarity Coefficient (DSC), Intersection over Union (IoU), and 95th percentile Hausdorff Distance (HD95).",
  },
  {
    id: 11,
    slug: "comparison",
    title: "11. Centralized vs. Federated Benchmark",
    subtitle: "Empirical Performance & Privacy Overhead",
    shortDesc: "Compares plaintext FedAvg against SecAgg+ / CKKS encryption, highlighting exact execution time ratios and communication payload sizes.",
  },
  {
    id: 12,
    slug: "resilience",
    title: "12. Node Resilience & Dropout",
    subtitle: "Fault-Tolerant Dynamic Node Management",
    shortDesc: "Monitors hospital heartbeats. If a hospital node drops offline during a round, the federation continues seamlessly with the minimum 2/3 threshold.",
  },
  {
    id: 13,
    slug: "segmentation",
    title: "13. 3D Tumor Segmentation Output",
    subtitle: "Volumetric Brain Tumor Mask Prediction",
    shortDesc: "Generates precise 3D predictions for Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT) overlaying multi-sequence MRI slices.",
  },
  {
    id: 14,
    slug: "complete-pipeline",
    title: "14. Complete FedMed System Flow",
    subtitle: "End-to-End Architectural Integration",
    shortDesc: "Full system diagram linking raw 3D MRI ingestion to local training, secure aggregation, global update, and final tumor segmentation.",
  },
];
