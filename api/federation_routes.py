"""
api/federation_routes.py

Dynamic Federation Control, Hospital Registration, Model Lifecycle & System Telemetry API.
Connects real backend state with the FedMed dashboard.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.state_manager import state_manager, get_real_system_resources, get_docker_containers_detailed

router = APIRouter(tags=["Federation Control & Hospital Lifecycle"])


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class RegisterHospitalRequest(BaseModel):
    name: str
    hospital_id: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8080
    grpc_port: Optional[int] = None
    endpoint: Optional[str] = None
    auth_token: Optional[str] = None
    type: str = "remote"  # "remote" or "docker"
    container_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConfigureDataRequest(BaseModel):
    mode: str = "demo"  # "demo" or "local"
    local_path: Optional[str] = None


class TrainRequest(BaseModel):
    epochs: int = 1


class StartFederationRequest(BaseModel):
    strategy: str = "FedAvg"


class InitiateContainerRequest(BaseModel):
    name: str
    hospital_id: Optional[str] = None
    port: Optional[int] = None
    data_mode: str = "demo"


# ---------------------------------------------------------------------------
# Hospital Registry Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/hospitals")
def get_registered_hospitals():
    """Return all legitimately registered hospital nodes."""
    return state_manager.get_all_hospitals()


@router.post("/api/hospitals/register")
def register_hospital(req: RegisterHospitalRequest):
    """Register a new remote or Docker hospital node."""
    try:
        return state_manager.register_hospital(
            name=req.name,
            hospital_id=req.hospital_id,
            host=req.host,
            port=req.port,
            grpc_port=req.grpc_port,
            endpoint=req.endpoint,
            auth_token=req.auth_token,
            hosp_type=req.type,
            container_id=req.container_id,
            metadata=req.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/approve")
def approve_hospital(hospital_id: str):
    """Approve a hospital for federation participation."""
    try:
        return state_manager.approve_hospital(hospital_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/reject")
def reject_hospital(hospital_id: str):
    """Reject hospital participation."""
    try:
        return state_manager.reject_hospital(hospital_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/revoke")
def revoke_hospital(hospital_id: str):
    """Revoke a hospital's federation credentials."""
    try:
        return state_manager.revoke_hospital(hospital_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/api/hospitals/{hospital_id}")
def delete_hospital(hospital_id: str):
    """Remove a hospital from central registry."""
    try:
        return state_manager.remove_hospital(hospital_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/connect")
def connect_hospital(hospital_id: str):
    """Establish secure connection with hospital node."""
    try:
        return state_manager.connect_hospital(hospital_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/init-project")
def init_hospital_project(hospital_id: str):
    """Initialize remote hospital client project configuration."""
    try:
        return state_manager.init_project(hospital_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Hospital-Side Private Lifecycle Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/hospitals/{hospital_id}/configure-data")
def configure_hospital_data(hospital_id: str, req: ConfigureDataRequest):
    """Configure hospital local MRI dataset (local path or demo). Raw data never transfers."""
    try:
        return state_manager.configure_data(
            hospital_id=hospital_id,
            mode=req.mode,
            local_path=req.local_path
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/init-model")
def init_hospital_model(hospital_id: str):
    """Instantiate global model into hospital local memory."""
    try:
        return state_manager.init_model(hospital_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/request-approval")
def request_federation_approval(hospital_id: str):
    """Hospital requests approval from central administrator to join federation."""
    try:
        return state_manager.request_approval(hospital_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/train")
def start_local_training(hospital_id: str, req: TrainRequest):
    """Execute local training inside hospital environment. Requires approval."""
    try:
        return state_manager.start_training(hospital_id=hospital_id, epochs=req.epochs)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/submit-model")
def submit_trained_model(hospital_id: str):
    """Submit permitted trained model update to central server. No raw data is transferred."""
    try:
        return state_manager.submit_model_update(hospital_id)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Federation of Models & Aggregation
# ---------------------------------------------------------------------------

@router.get("/api/federation/incoming-models")
def get_incoming_models():
    """Retrieve list of model updates submitted by hospitals awaiting aggregation."""
    return state_manager.get_incoming_models()


@router.post("/api/federation/start")
def start_federation(req: StartFederationRequest):
    """Trigger aggregation of incoming model updates to produce a new global model."""
    try:
        return state_manager.start_federation(strategy=req.strategy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/federation/global-model")
def get_global_model():
    """Retrieve persistent global model version and history."""
    return state_manager.get_global_model_info()


# ---------------------------------------------------------------------------
# Activity Logs
# ---------------------------------------------------------------------------

@router.get("/api/activity/logs")
def get_activity_logs():
    """Retrieve persistent chronological federation and hospital activity logs."""
    return state_manager.get_activity_logs()


# ---------------------------------------------------------------------------
# System Resources & Simulate Remote Hospital
# ---------------------------------------------------------------------------

@router.get("/api/system/resources")
def get_system_resources():
    """Retrieve real host RAM, Disk, and Docker daemon operational metrics."""
    return get_real_system_resources()


@router.get("/api/docker/simulated-hospitals")
def get_simulated_hospitals():
    """Retrieve all simulated Docker hospitals."""
    all_h = state_manager.get_all_hospitals()
    return [h for h in all_h if h.get("type") == "docker"]


@router.get("/api/docker/containers")
def get_docker_containers():
    """Retrieve real active and stopped Docker containers grouped by status."""
    return get_docker_containers_detailed()


@router.post("/api/docker/containers/{container_id}/start")
def start_docker_container(container_id: str):
    """Start a stopped Docker container."""
    try:
        return state_manager.start_docker_container(container_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/docker/containers/{container_id}/stop")
def stop_docker_container(container_id: str):
    """Stop an active Docker container."""
    try:
        return state_manager.stop_docker_container(container_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/docker/containers/{container_id}/restart")
def restart_docker_container(container_id: str):
    """Restart a Docker container."""
    try:
        return state_manager.restart_docker_container(container_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/docker/containers/{container_id}")
def remove_docker_container(container_id: str):
    """Remove a Docker container."""
    try:
        return state_manager.remove_docker_container(container_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/docker/start")
def start_hospital_docker(hospital_id: str):
    """Start or spin up Docker container specifically for this hospital."""
    try:
        return state_manager.start_docker_container(hospital_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/docker/stop")
def stop_hospital_docker(hospital_id: str):
    """Stop Docker container specifically for this hospital."""
    try:
        return state_manager.stop_docker_container(hospital_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/hospitals/{hospital_id}/docker/restart")
def restart_hospital_docker(hospital_id: str):
    """Restart Docker container specifically for this hospital."""
    try:
        return state_manager.restart_docker_container(hospital_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/docker/initiate")
def initiate_simulated_container(req: InitiateContainerRequest):
    """Initiate a simulated hospital Docker container with system resource checks."""
    try:
        hosp = state_manager.initiate_simulated_hospital(
            name=req.name,
            hospital_id=req.hospital_id,
            port=req.port,
            data_mode=req.data_mode
        )
        return hosp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/hospitals/{hospital_id}/training-metrics")
def get_hospital_training_metrics(hospital_id: str):
    """Retrieve historical telemetry metrics for a hospital's local training runs."""
    hosp = state_manager.get_hospital(hospital_id)
    if not hosp:
        raise HTTPException(status_code=404, detail=f"Hospital '{hospital_id}' not found.")
    return hosp.get("training_metrics", [])


class SetExecutionModeRequest(BaseModel):
    mode: str  # "standby" or "docker"


@router.get("/api/execution-mode")
def get_current_execution_mode():
    """Retrieve current federation execution mode (standby vs docker) and daemon state."""
    return state_manager.get_execution_mode()


@router.post("/api/execution-mode")
def set_current_execution_mode(req: SetExecutionModeRequest):
    """Switch execution mode between Standby/Native and Docker Simulation."""
    try:
        return state_manager.set_execution_mode(req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/docker/status")
def get_docker_status():
    """Check if Docker Desktop is running and return container status."""
    res = get_real_system_resources()
    docker_running = res.get("docker", {}).get("running", False)
    return {
        "docker_running": docker_running,
        "execution_mode": state_manager.execution_mode,
        "ram": res.get("ram", {}),
        "disk": res.get("disk", {}),
        "active_containers": res.get("docker", {}).get("active_containers", 0),
        "status_text": "Docker daemon is active" if docker_running else "Docker Desktop is currently not running",
    }


@router.post("/api/docker/start-simulation")
def start_docker_simulation_infra():
    """Invoke existing Docker Compose infrastructure to start configured hospital nodes."""
    try:
        result = state_manager.start_docker_simulation()
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

