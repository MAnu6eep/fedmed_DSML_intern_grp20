"""
fedmed/security/grpc_tls.py

TLS/mTLS helpers for FedMed gRPC communication.
Generated certificates and private keys remain local.
"""

from concurrent import futures
from pathlib import Path

import grpc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CERTS_ROOT = PROJECT_ROOT / "certs"

CA_CERT = CERTS_ROOT / "ca" / "ca.crt"


def _read_file(path: Path) -> bytes:
    """Read a local TLS artifact as bytes."""
    if not path.is_file():
        raise FileNotFoundError(f"TLS file not found: {path}")

    return path.read_bytes()


def certificate_paths(node_id: str) -> tuple[Path, Path]:
    """Return certificate and private-key paths for a node."""
    certificate_path = CERTS_ROOT / node_id / f"{node_id}.crt"
    private_key_path = CERTS_ROOT / node_id / f"{node_id}.key"

    return certificate_path, private_key_path


def load_server_credentials(
    certificate_path: Path,
    private_key_path: Path,
) -> grpc.ServerCredentials:
    """
    Create mTLS server credentials for a specific node.

    The node presents its own certificate and requires clients
    to present a certificate signed by the trusted FedMed CA.
    """
    ca_cert = _read_file(CA_CERT)
    certificate = _read_file(certificate_path)
    private_key = _read_file(private_key_path)

    return grpc.ssl_server_credentials(
        ((private_key, certificate),),
        root_certificates=ca_cert,
        require_client_auth=True,
    )


def load_client_credentials(
    certificate_path: Path,
    private_key_path: Path,
) -> grpc.ChannelCredentials:
    """
    Create mTLS client credentials for a specific node.

    The client verifies the peer against the FedMed CA and
    presents its own certificate and private key.
    """
    ca_cert = _read_file(CA_CERT)
    certificate = _read_file(certificate_path)
    private_key = _read_file(private_key_path)

    return grpc.ssl_channel_credentials(
        root_certificates=ca_cert,
        private_key=private_key,
        certificate_chain=certificate,
    )


def hospital_certificate_paths(
    hospital_id: str,
) -> tuple[Path, Path]:
    """Return certificate and private-key paths for a hospital."""
    return certificate_paths(hospital_id)


def add_identity_service(
    server: grpc.Server,
    identity: str,
) -> None:
    """
    Register a minimal gRPC identity endpoint.

    This endpoint verifies that the mTLS handshake succeeds and
    that the authenticated node can reach the gRPC service.
    """

    def get_identity(request, context):
        return identity.encode("utf-8")

    handler = grpc.unary_unary_rpc_method_handler(
        get_identity,
        request_deserializer=lambda value: value,
        response_serializer=lambda value: value,
    )

    service = grpc.method_handlers_generic_handler(
        "fedmed.Security",
        {
            "GetIdentity": handler,
        },
    )

    server.add_generic_rpc_handlers((service,))


def create_secure_grpc_server(
    port: int,
    identity: str,
) -> grpc.Server:
    """
    Create an mTLS-enabled gRPC server for a specific node.

    The node uses its own certificate/private key and requires
    clients to present certificates signed by the FedMed CA.
    """
    certificate_path, private_key_path = certificate_paths(identity)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4)
    )

    add_identity_service(server, identity)

    server_credentials = load_server_credentials(
        certificate_path=certificate_path,
        private_key_path=private_key_path,
    )

    bound_port = server.add_secure_port(
        f"[::]:{port}",
        server_credentials,
    )

    if bound_port != port:
        raise RuntimeError(
            f"Failed to bind secure gRPC server to port {port}"
        )

    return server
