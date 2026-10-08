# FedMed Project Manager Visual Explanation Website & Architecture Specification

## 1. Discovered Repository Architecture

The FedMed project implements a hybrid, privacy-preserving federated learning system tailored for 3D brain tumor MRI segmentation. Thorough inspection of the repository revealed the following authoritative modules:

| Component Category | Implementation Module | Authoritative Code Details |
|---|---|---|
| **3D Neural Architecture** | `fedmed/core/model.py` | `FedMedUNet3D` using MONAI `UNet` (`spatial_dims=3`, `in_channels=4`, `out_channels=1`, `channels=(16, 32, 64, 128, 256)`, `strides=(2, 2, 2, 2)`, `num_res_units=2`, `dropout=0.1`). |
| **Volumetric Preprocessing** | `fedmed/data/loader.py` | MONAI Compose dictionary transforms: `LoadImaged`, `Orientationd(axcodes="RAS")`, `Spacingd(pixdim=(1.0, 1.0, 1.0))`, `NormalizeIntensityd(nonzero=True, channel_wise=True)`, `RandCropByPosNegLabeld(roi_size=(64, 64, 32))`. |
| **Data Partitioning** | `fedmed/data/partitioner.py` | Uniform `partition_iid` and Non-IID Dirichlet `partition_dirichlet` parameterized by concentration $\alpha$. |
| **Federated Strategies** | `fedmed/federation/server.py` | Flower strategies: `FedAvg`, `FedProx` ($\mu=0.01$ proximal penalty), and `SCAFFOLD` (`scaffold.py` control variates $c, c_i$). |
| **Homomorphic Encryption** | `fedmed/privacy/tenseal_engine.py` | TenSEAL CKKS engine (`poly_modulus_degree=8192`, `coeff_mod_bit_sizes=[60, 40, 40, 60]`, `global_scale=2^40`). |
| **Secure Aggregation** | `fedmed/privacy/secagg_config.py` | SecAgg+ Shamir secret sharing ($k=2/3$ threshold, 16-bit quantization, $2^{31}$ modulus). |
| **Differential Privacy** | `fedmed/privacy/differential_privacy.py` | DP-SGD L2 gradient clipping ($C=1.0$) + Gaussian noise ($\sigma=1.0$) + $(\epsilon, \delta)$ accounting. |
| **Node Resilience** | `fedmed/resilience/dropout.py` | Heartbeat monitoring with dynamic dropout tolerance ($2/3$ quorum minimum). |
| **Evaluation Metrics** | `fedmed/core/evaluation.py` | Dice Similarity Coefficient (DSC), IoU / Jaccard Index, Hausdorff Distance 95% (HD95), Loss. |
| **Telemetry & API** | `api/main.py` & `api/telemetry.py` | FastAPI backend with WebSocket endpoint `/ws/telemetry` & REST endpoints `/api/hospitals`, `/api/metrics`, `/api/segmentation`. |

---

## 2. What the Website Represents

The website acts as an interactive visual explanation and control center designed specifically for project manager reviews and system demonstrations.

It visualizes the complete end-to-end data lifecycle:
1. **Raw Multi-Modal MRI Ingestion**: 4 input sequences (FLAIR, T1, T1ce, T2).
2. **MONAI Preprocessing Pipeline**: Transformation into a standardized 3D tensor $(4, 64, 64, 32)$.
3. **On-Premise Hospital Isolation**: Raw MRI scans remain strictly inside hospital firewalls.
4. **Concurrent Local 3D U-Net Training**: Generating local model weight updates ($\Delta w$).
5. **Cryptographic Protection Layers**: TenSEAL CKKS Homomorphic Encryption, SecAgg+ zero-sum masking, and DP-SGD noise.
6. **Server Aggregation & Global Model Update**: Flower FedAvg, FedProx, or SCAFFOLD strategy fusion.
7. **Round Progression & Convergence**: Metric evolution across communication rounds.
8. **3D Tumor Segmentation Output**: Volumetric predictions for Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT).

---

## 3. Actual Repository Features vs. Illustrative Demo Values

| Feature / Value | Category | Verification Source |
|---|---|---|
| **3D U-Net Structure & Channel Sizes** | **Actual Feature** | `fedmed/core/model.py` (`channels=(16,32,64,128,256)`) |
| **MONAI Preprocessing Transforms** | **Actual Feature** | `fedmed/data/loader.py` |
| **FedAvg, FedProx, SCAFFOLD Strategies** | **Actual Feature** | `fedmed/federation/server.py` |
| **TenSEAL CKKS Encryption Parameters** | **Actual Feature** | `fedmed/privacy/tenseal_engine.py` |
| **SecAgg+ 2/3 Threshold & 16-bit Quantization** | **Actual Feature** | `fedmed/privacy/secagg_config.py` |
| **DP-SGD Clipping & Noise Multiplier** | **Actual Feature** | `fedmed/privacy/differential_privacy.py` |
| **SecAgg Benchmark Timing (0.1466s) & Overhead (+2385%)** | **Actual Benchmark Data** | `docs/secagg_benchmark_results.json` |
| **FedAvg vs FedProx Round Loss Logs** | **Actual Benchmark Data** | `tests/results/fedavg_fedprox_comparison.json` |
| **Sample Slice Image Textures** | **Illustrative Demo** | Mock SVG/Canvas rendering in static mode |
| **Local Training Progress Bar Timers** | **Illustrative Demo** | Interactive client simulation in static mode |

---

## 4. How to Run & Start the Website

1. Navigate to the `dashboard` directory:
   ```bash
   cd dashboard
   ```

2. Run the Vite development server:
   ```bash
   npm run dev
   ```

3. Open your browser at `http://localhost:5173`.

---

## 5. How to Enter and Use Presentation Mode

1. **Via Header Button**: Click the **"PM Review Mode"** or **"Presentation Mode"** tab in the top navigation header.
2. **Via Hero CTA**: Click **"Launch PM Presentation Mode"** on the main hero screen.
3. **Keyboard Controls in Presentation Mode**:
   - `Space`: Pause or resume automated slide progression.
   - `Right Arrow (→)`: Advance to next pipeline stage.
   - `Left Arrow (←)`: Return to previous stage.
   - `Escape (Esc)`: Exit presentation mode back to interactive explorer.

---

## 6. Future Backend & Telemetry Integration Readiness

The website is structured cleanly so that real-time FastAPI & WebSocket integration can be enabled without changing the UI layout:

- **Telemetry Client**: `src/api/telemetry.ts` connects to `ws://127.0.0.1:8000/ws/telemetry`.
- **Rest API Client**: `src/api/client.ts` calls `http://127.0.0.1:8000/api/hospitals`, `/api/metrics`, and `/api/segmentation`.
- **Zustand State Store**: `src/store/telemetryStore.ts` automatically maps WebSocket events (`ROUND_STARTED`, `TRAINING_COMPLETED`, `AGGREGATION_COMPLETED`, `CLIENT_DISCONNECTED`) into live hospital node statuses and chart points.
- **Graceful Fallback**: If the FastAPI backend is not running, the website automatically falls back to static demo mode with pre-populated benchmark logs.
