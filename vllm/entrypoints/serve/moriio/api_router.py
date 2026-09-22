# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, HTTPException, Request

router = APIRouter()


@router.get("/v1/moriio/metadata")
async def moriio_metadata(request: Request):
    """Advertise MoRI-IO control addresses to a statically configured router."""
    config = request.app.state.vllm_config
    kv_config = config.kv_transfer_config
    if kv_config is None or kv_config.kv_connector != "MoRIIOConnector":
        raise HTTPException(status_code=404, detail="MoRIIOConnector is not enabled")

    pc = config.parallel_config
    if (
        pc.data_parallel_size != 1
        or pc.pipeline_parallel_size != 1
        or pc.nnodes != 1
        or pc.distributed_executor_backend not in ("uni", "mp")
    ):
        raise HTTPException(
            status_code=400,
            detail="Static MoRI-IO metadata supports single-node uni/mp workers "
            "with DP=1 and PP=1 only; use service discovery for other topologies",
        )
    if kv_config.kv_role not in ("kv_producer", "kv_consumer"):
        raise HTTPException(
            status_code=400,
            detail="Static MoRI-IO metadata requires kv_producer or kv_consumer",
        )

    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
        MoRIIOConstants,
        get_moriio_mode,
        resolve_host_ip,
    )

    extra = kv_config.kv_connector_extra_config
    host = resolve_host_ip(extra)
    handshake = int(
        extra.get("handshake_port") or MoRIIOConstants.DEFAULT_HANDSHAKE_PORT
    )
    notify = int(extra.get("notify_port") or MoRIIOConstants.DEFAULT_NOTIFY_PORT)
    return {
        "type": "P" if kv_config.kv_role == "kv_producer" else "D",
        "http_address": request.url.netloc,
        "zmq_address": f"host:{host},handshake:{handshake},notify:{notify}",
        "dp_size": pc.data_parallel_size,
        "tp_size": pc.tensor_parallel_size,
        "transfer_mode": get_moriio_mode(kv_config).name,
    }


def attach_router(app: FastAPI):
    app.include_router(router)
