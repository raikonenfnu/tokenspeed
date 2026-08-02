"""TokenSpeed-owned adapter for the bundled SMG gRPC engine.

The TokenSpeed protobuf has no first-class seed field.  Newer SMG routers can
transport extension sampling fields in ``SamplingParams.custom_params``; the
currently pinned router omits the OpenAI seed entirely.  Keep both compatibility
shims at this boundary so core runtime sampling remains gateway-neutral.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from google.protobuf.json_format import MessageToDict
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2
from smg_grpc_servicer.tokenspeed import server as smg_server
from smg_grpc_servicer.tokenspeed.__main__ import main
from smg_grpc_servicer.tokenspeed.servicer import (
    HEALTH_CHECK_TIMEOUT,
    TokenSpeedSchedulerServicer as _SMGTokenSpeedSchedulerServicer,
)


def _custom_sampling_params(params) -> dict[str, Any]:
    """Decode SMG's optional sampling-extension struct."""
    if not params.HasField("custom_params"):
        return {}
    value = MessageToDict(params.custom_params)
    return value if isinstance(value, dict) else {}


def _server_seed_was_explicit() -> bool:
    """Return whether this engine process was launched with ``--seed``."""
    return any(arg == "--seed" or arg.startswith("--seed=") for arg in sys.argv[1:])


class TokenSpeedSchedulerServicer(_SMGTokenSpeedSchedulerServicer):
    """Preserve SMG extension fields needed by TokenSpeed sampling."""

    async def HealthCheck(self, request, context):
        """Probe scheduler liveness without injecting an inference request.

        SMG's default deep probe generates one token. Periodic router probes
        can therefore change a live decode batch's graph padding and logits.
        A load round-trip exercises the same scheduler control channel without
        entering model execution.
        """
        del request, context
        if self.async_llm.gracefully_exit:
            return tokenspeed_scheduler_pb2.HealthCheckResponse(
                healthy=False,
                message="Server is shutting down",
            )
        try:
            await asyncio.wait_for(
                self.async_llm.get_load(),
                timeout=HEALTH_CHECK_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - health RPC reports failure.
            return tokenspeed_scheduler_pb2.HealthCheckResponse(
                healthy=False,
                message=f"Scheduler health check failed: {exc}",
            )
        return tokenspeed_scheduler_pb2.HealthCheckResponse(
            healthy=True,
            message="Health check passed (scheduler load round-trip)",
        )

    @staticmethod
    def _sampling_params_from_proto(
        params,
        *,
        reasoning_parser: str | None = None,
    ) -> dict[str, Any]:
        out = _SMGTokenSpeedSchedulerServicer._sampling_params_from_proto(
            params,
            reasoning_parser=reasoning_parser,
        )
        custom = _custom_sampling_params(params)
        seed = custom.pop("seed", None)
        if seed is not None:
            out["seed"] = int(seed)
        if custom:
            out["custom_params"] = custom
        return out

    def _build_generate_req(self, request):
        """Build a request and recover a seed omitted by the pinned router.

        An explicit seed carried in ``custom_params`` always wins.  Otherwise,
        use the engine seed when one was configured.  This makes a deployment
        with ``--seed`` reproducible without changing sampling semantics for a
        deployment that leaves the server seed unset.
        """
        obj = super()._build_generate_req(request)
        if "seed" not in obj.sampling_params and _server_seed_was_explicit():
            server_seed = getattr(self.server_args, "seed", None)
            if server_seed is not None:
                obj.sampling_params["seed"] = int(server_seed)
        return obj

# ``serve_grpc`` resolves this module global when it constructs the service.
smg_server.TokenSpeedSchedulerServicer = TokenSpeedSchedulerServicer


if __name__ == "__main__":
    main()
