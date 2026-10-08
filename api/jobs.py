"""
api/jobs.py

On-demand job runner for FedMed experiments and test suites.
Decouples experiments and tests from project startup, allowing explicit user execution from the dashboard.
"""

import os
import sys
import threading
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/jobs", tags=["Jobs & Experiments"])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_BIN = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


class JobManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.current_job: Dict[str, Any] = {
            "id": None,
            "type": None,
            "name": None,
            "status": "idle",
            "command": [],
            "start_time": None,
            "end_time": None,
            "return_code": None,
            "logs": [],
        }

    def _reader_thread(self, proc: subprocess.Popen, job_id: str):
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                clean_line = line.rstrip("\r\n")
                with self.lock:
                    if self.current_job["id"] == job_id:
                        self.current_job["logs"].append(clean_line)
                        if len(self.current_job["logs"]) > 2000:
                            self.current_job["logs"] = self.current_job["logs"][-2000:]
            proc.stdout.close()
            ret_code = proc.wait()
            with self.lock:
                if self.current_job["id"] == job_id:
                    self.current_job["return_code"] = ret_code
                    if self.current_job["status"] == "running":
                        self.current_job["status"] = "completed" if ret_code == 0 else "failed"
                    self.current_job["end_time"] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            with self.lock:
                if self.current_job["id"] == job_id:
                    self.current_job["logs"].append(f"[Internal Error] {str(e)}")
                    self.current_job["status"] = "failed"
                    self.current_job["end_time"] = datetime.now(timezone.utc).isoformat()

    def start_job(self, job_type: str, name: str, cmd: List[str]) -> Dict[str, Any]:
        with self.lock:
            if self.current_job["status"] == "running" and self.process and self.process.poll() is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"A job ({self.current_job['name']}) is already running. Please stop it first."
                )

            job_id = f"job_{int(time.time())}"
            self.current_job = {
                "id": job_id,
                "type": job_type,
                "name": name,
                "status": "running",
                "command": cmd,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": None,
                "return_code": None,
                "logs": [f"=== Starting {job_type.upper()}: {name} ===", f"Command: {' '.join(cmd)}"],
            }

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=str(PROJECT_ROOT),
                    env=os.environ.copy()
                )
            except Exception as e:
                self.current_job["status"] = "failed"
                self.current_job["end_time"] = datetime.now(timezone.utc).isoformat()
                self.current_job["logs"].append(f"Failed to spawn process: {e}")
                raise HTTPException(status_code=500, detail=str(e))

            thread = threading.Thread(
                target=self._reader_thread,
                args=(self.process, job_id),
                daemon=True
            )
            thread.start()

            return self.get_status()

    def stop_job(self) -> Dict[str, Any]:
        with self.lock:
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                    time.sleep(0.5)
                    if self.process.poll() is None:
                        self.process.kill()
                    self.current_job["status"] = "stopped"
                    self.current_job["end_time"] = datetime.now(timezone.utc).isoformat()
                    self.current_job["logs"].append("=== Process stopped by user ===")
                except Exception as e:
                    self.current_job["logs"].append(f"Error terminating process: {e}")
            else:
                if self.current_job["status"] == "running":
                    self.current_job["status"] = "stopped"
            return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            # Check if still running
            if self.current_job["status"] == "running" and self.process:
                poll = self.process.poll()
                if poll is not None:
                    self.current_job["return_code"] = poll
                    self.current_job["status"] = "completed" if poll == 0 else "failed"
                    if not self.current_job["end_time"]:
                        self.current_job["end_time"] = datetime.now(timezone.utc).isoformat()

            return {
                "id": self.current_job["id"],
                "type": self.current_job["type"],
                "name": self.current_job["name"],
                "status": self.current_job["status"],
                "command": self.current_job["command"],
                "start_time": self.current_job["start_time"],
                "end_time": self.current_job["end_time"],
                "return_code": self.current_job["return_code"],
                "log_count": len(self.current_job["logs"]),
                "logs": self.current_job["logs"][-200:],  # Return latest 200 lines for UI efficiency
            }

    def clear_logs(self):
        with self.lock:
            self.current_job["logs"] = []
            return {"status": "cleared"}


job_manager = JobManager()


# ---------------------------------------------------------------------------
# API Models & Endpoints
# ---------------------------------------------------------------------------

class RunExperimentRequest(BaseModel):
    config_file: str  # e.g., "fedavg.yaml"


class RunTestRequest(BaseModel):
    suite_id: str  # e.g., "phase4_docker", "ml", "privacy", "all"


@router.get("/status")
def get_job_status():
    """Get the current or latest job status, metrics, and terminal logs."""
    return job_manager.get_status()


@router.post("/stop")
def stop_active_job():
    """Halt any active running experiment or test execution."""
    return job_manager.stop_job()


@router.post("/clear-logs")
def clear_job_logs():
    """Clear terminal output buffer."""
    return job_manager.clear_logs()


@router.get("/experiments")
def list_available_experiments():
    """List all available YAML experiments in experiments/ with parsed parameters."""
    exp_dir = PROJECT_ROOT / "experiments"
    configs = []
    if exp_dir.exists():
        for file in sorted(exp_dir.glob("*.yaml")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                configs.append({
                    "filename": file.name,
                    "strategy": data.get("strategy", "unknown"),
                    "rounds": data.get("rounds", data.get("num_rounds", 1)),
                    "local_epochs": data.get("local_epochs", 1),
                    "batch_size": data.get("batch_size", 2),
                    "learning_rate": data.get("learning_rate", 0.0001),
                    "partition": data.get("partition", {}),
                    "extra": {k: v for k, v in data.items() if k not in ["strategy", "rounds", "num_rounds", "local_epochs", "batch_size", "learning_rate", "partition"]}
                })
            except Exception as e:
                configs.append({"filename": file.name, "error": str(e)})
    return configs


@router.post("/run-experiment")
def run_experiment(req: RunExperimentRequest):
    """Launch a federated experiment on-demand using run_experiment.py."""
    config_path = PROJECT_ROOT / "experiments" / req.config_file
    if not config_path.is_file():
        raise HTTPException(status_code=404, detail=f"Experiment config '{req.config_file}' not found.")

    cmd = [
        PYTHON_BIN,
        str(PROJECT_ROOT / "run_experiment.py"),
        "--config",
        str(config_path)
    ]
    return job_manager.start_job(
        job_type="experiment",
        name=req.config_file,
        cmd=cmd
    )


TEST_SUITES = {
    "phase4_docker": {
        "title": "Docker Hospital Isolation & Profiles",
        "description": "Verifies isolated read-only data mounts, hospital identities, and compose profiles.",
        "path": "tests/test_phase4_docker_hospital_environment.py"
    },
    "privacy": {
        "title": "Privacy & Cryptography (SecAgg + DP)",
        "description": "Evaluates TenSEAL CKKS homomorphic aggregation and differential privacy noise.",
        "path": "tests/test_ckks_addition.py tests/test_secure_aggregation.py tests/test_differential_privacy.py"
    },
    "ml_pipeline": {
        "title": "Core ML & Segmentation Pipeline",
        "description": "Tests 3D UNet model tensor shapes, preprocessing consistency, and slice extraction.",
        "path": "tests/test_model.py tests/test_preprocessing_consistency.py tests/test_slice_extraction.py"
    },
    "resilience": {
        "title": "Node Resilience & Dropout Recovery",
        "description": "Evaluates quorum adaptation when hospital nodes disconnect or recover.",
        "path": "tests/test_dropout.py tests/test_recovery.py tests/test_heartbeat.py"
    },
    "all": {
        "title": "Complete Test Suite",
        "description": "Executes all 39 test files across the repository.",
        "path": "tests/"
    }
}


@router.get("/test-suites")
def list_available_test_suites():
    """List predefined test suites available for user execution."""
    return [
        {
            "id": suite_id,
            "title": info["title"],
            "description": info["description"],
            "path": info["path"]
        }
        for suite_id, info in TEST_SUITES.items()
    ]


@router.post("/run-test")
def run_test_suite(req: RunTestRequest):
    """Launch a specific pytest verification suite on-demand."""
    if req.suite_id not in TEST_SUITES:
        raise HTTPException(status_code=404, detail=f"Test suite '{req.suite_id}' not found.")

    suite_info = TEST_SUITES[req.suite_id]
    args = suite_info["path"].split()
    
    cmd = [PYTHON_BIN, "-m", "pytest", "-v"] + args

    return job_manager.start_job(
        job_type="test",
        name=suite_info["title"],
        cmd=cmd
    )
