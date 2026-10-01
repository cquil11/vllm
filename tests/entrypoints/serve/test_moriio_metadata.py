# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware
from vllm.entrypoints.serve.moriio.api_router import attach_router


@pytest.fixture
def app():
    app = FastAPI()
    app.state.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MoRIIOConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"host_ip": "10.0.0.1"},
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            nnodes=1,
            distributed_executor_backend="mp",
        ),
    )
    attach_router(app)
    return app


@pytest.mark.parametrize(
    "role,read_mode", [("kv_producer", False), ("kv_consumer", True)]
)
def test_metadata_without_proxy_settings(app, role, read_mode):
    """Static routing receives role, transfer mode and the actual base ports."""
    kv = app.state.vllm_config.kv_transfer_config
    kv.kv_role = role
    kv.kv_connector_extra_config.update(
        read_mode=read_mode, handshake_port="6302", notify_port="61006"
    )
    response = TestClient(app).get("/v1/moriio/metadata")
    assert response.status_code == 200
    assert response.json() == {
        "type": "P" if role == "kv_producer" else "D",
        "http_address": "testserver",
        "zmq_address": "host:10.0.0.1,handshake:6302,notify:61006",
        "dp_size": 1,
        "tp_size": 4,
        "transfer_mode": "READ" if read_mode else "WRITE",
    }


def test_metadata_defaults_and_authentication(app):
    app.add_middleware(AuthenticationMiddleware, tokens=["secret"])
    client = TestClient(app)
    assert client.get("/v1/moriio/metadata").status_code == 401
    response = client.get(
        "/v1/moriio/metadata", headers={"Authorization": "Bearer secret"}
    )
    assert response.status_code == 200
    assert response.json()["zmq_address"] == (
        "host:10.0.0.1,handshake:6301,notify:61005"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("nnodes", 2),
        ("distributed_executor_backend", "ray"),
    ],
)
def test_metadata_rejects_topologies_with_ambiguous_addresses(app, field, value):
    setattr(app.state.vllm_config.parallel_config, field, value)
    assert TestClient(app).get("/v1/moriio/metadata").status_code == 400


def test_metadata_unavailable_for_other_connectors(app):
    app.state.vllm_config.kv_transfer_config.kv_connector = "NixlConnector"
    assert TestClient(app).get("/v1/moriio/metadata").status_code == 404
