# FedMed: Privacy-Preserving Federated 3D Medical Image Segmentation & Benchmark Platform

**FedMed** is an end-to-end framework for running secure, multi-institutional federated learning experiments on 3D medical images (like MRI scans). It integrates deep learning for segmentation (**MONAI / PyTorch**), federated orchestration (**Flower**), privacy mechanisms (**TenSEAL Homomorphic Encryption, Differential Privacy**), and a full interactive web suite (**FastAPI / React / Docker**) to simulate and monitor real-world remote hospital deployments.

---

## 🚀 Key Features

* **3D Medical Segmentation**: Uses a 3D UNet architecture via MONAI to process NIfTI (`.nii.gz`) MRI volumes.
* **Multiple FL Strategies**: Built-in support for **FedAvg**, **FedProx** (for non-IID data), and **SCAFFOLD** (control variates).
* **Military-Grade Privacy**: 
  * **Secure Aggregation (SecAgg)**: Model updates are encrypted using CKKS Homomorphic Encryption (via TenSEAL) before leaving the hospital. The central server averages them *without ever decrypting* individual hospital weights.
  * **Differential Privacy (DP)**: Injects noise to prevent model inversion attacks.
* **Isolated Docker Architecture**: Simulates remote hospitals locally using fully isolated Docker containers with read-only data mounts.
* **Interactive Live Dashboard**: A modern web dashboard built with React and Tailwind CSS to track global models, hospital health, incoming updates, and live training progress metrics.

---

## 🏗️ Architecture

FedMed simulates a real-world Hub-and-Spoke topology:
- **Central FedMed Server (The Hub)**: Orchestrates federation rounds, aggregates encrypted weights, and serves the UI dashboard telemetry. Raw patient data *never* reaches the hub.
- **Hospital Clients (The Spokes)**: Each hospital operates as an isolated entity (simulated via Docker container). Hospitals load local private datasets, train the global model locally, encrypt the weight updates, and send them back to the hub.

---

## 💻 Prerequisites

Ensure you have the following installed on your machine:
* Python 3.10 or 3.11
* Node.js (v18+) & npm
* Docker Desktop (Required for multi-hospital container simulation)

---

## ⚙️ Installation & Setup

**1. Clone & Setup Python Environment**
```powershell
cd fedmed_DSML_intern_grp20
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

**2. Generate Isolated Synthetic 3D Datasets**
To simulate real hospital data, generate isolated synthetic 3D BraTS NIfTI image volumes for all registered hospital nodes.
```powershell
python scripts/setup_data.py
```
*(This generates data inside `hospitals/hospital_a/data/`, `hospitals/hospital_b/data/`, etc.)*

**3. Setup Dashboard Frontend**
Open a new terminal and prepare the React frontend.
```powershell
cd dashboard
npm install
```

---

## 🏃‍♂️ Running the Interactive Demo (UI & Docker)

To see FedMed in action visually, follow these steps using three separate terminal windows:

### Terminal 1: Start Central API & Telemetry
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload --port 8000
```
*(Access Swagger Docs at: `http://localhost:8000/docs`)*

### Terminal 2: Start Web Dashboard
```powershell
cd dashboard
npm run dev
```
*(Access Dashboard UI at: `http://localhost:5173`)*

### Terminal 3: Start Hospital Containers
Ensure Docker Desktop is running on your machine.
```powershell
# Launch 3 simulated hospital environments
docker-compose --profile hospital-a --profile hospital-b --profile hospital-c up -d --build
```
Navigate to the Web Dashboard. Click on **Hospital Registry** -> **Docker Hospital Control** to see your containers. Click **Open Hospital** to view the private interface of a specific hospital and track its training loop.

---

## 🔬 Running FL Experiments via CLI

You can bypass the UI and directly run headless federated simulations to evaluate model convergence and benchmark algorithms.

```powershell
# Run baseline Federated Averaging
python run_experiment.py --config experiments/fedavg.yaml

# Run Federated Proximal (optimized for heterogeneous Non-IID data)
python run_experiment.py --config experiments/fedprox.yaml

# Run SCAFFOLD (uses control variates to correct client drift)
python run_experiment.py --config experiments/scaffold.yaml

# Run Centralized Baseline (Upper-bound evaluation pooling all data together)
python run_experiment.py --config experiments/centralized.yaml
```
Output model checkpoints and metrics are saved dynamically as rounds progress.

---

## 🧪 Running the Test Suite

FedMed includes a massive 187-test verification suite covering ML mathematics, privacy bounds, and Docker isolation.

```powershell
.\.venv\Scripts\Activate.ps1

# Run the complete project test suite
pytest tests/ -v

# Run just the Docker isolation and profile verification tests
pytest tests/test_phase4_docker_hospital_environment.py -v
```

---

## 📂 Project Structure

```text
C:\fedmed_DSML_intern_grp20\
├── api/                  # FastAPI Central Server & Telemetry WebSockets
├── dashboard/            # React/Vite/Tailwind Frontend UI
├── docker/               # Dockerfile configurations for isolated nodes
├── experiments/          # YAML configurations for FedAvg, FedProx, SCAFFOLD
├── fedmed/               # Core Python Package (ML, Privacy, Federation)
│   ├── core/             # MONAI 3D UNet Models
│   ├── data/             # NIfTI Preprocessing pipelines
│   ├── federation/       # Flower FL Strategies
│   ├── security/         # SecAgg (TenSEAL CKKS) & Differential Privacy
│   └── nodes/            # Hospital Client Node Manager
├── hospitals/            # Isolated directories for Hospital A, B, C, D, E
├── scripts/              # Data generation utilities
└── tests/                # 180+ Automated Pytest verification tests
```