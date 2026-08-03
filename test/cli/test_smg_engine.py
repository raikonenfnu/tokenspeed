"""Tests for TokenSpeed's SMG sampling-parameter adapter."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from smg_grpc_proto.generated import tokenspeed_scheduler_pb2
from smg_grpc_servicer.tokenspeed.servicer import (
    TokenSpeedSchedulerServicer as SMGTokenSpeedSchedulerServicer,
)

from tokenspeed.cli.smg_engine import TokenSpeedSchedulerServicer


def test_health_check_does_not_inject_generation() -> None:
    servicer = object.__new__(TokenSpeedSchedulerServicer)
    servicer.async_llm = SimpleNamespace(
        gracefully_exit=False,
        get_load=AsyncMock(return_value=[]),
    )

    response = asyncio.run(
        servicer.HealthCheck(
            tokenspeed_scheduler_pb2.HealthCheckRequest(),
            SimpleNamespace(),
        )
    )

    assert response.healthy
    assert response.message == "Health check passed (scheduler load round-trip)"
    servicer.async_llm.get_load.assert_awaited_once_with()


def test_sampling_adapter_preserves_seed_and_other_custom_params() -> None:
    params = tokenspeed_scheduler_pb2.SamplingParams(
        temperature=1.0,
        top_p=0.95,
    )
    params.custom_params.update({"seed": 42, "reasoning_effort": "high"})

    actual = TokenSpeedSchedulerServicer._sampling_params_from_proto(params)

    assert actual["seed"] == 42
    assert actual["custom_params"] == {"reasoning_effort": "high"}


def test_sampling_adapter_leaves_absent_extensions_absent() -> None:
    params = tokenspeed_scheduler_pb2.SamplingParams(temperature=0.0)

    actual = TokenSpeedSchedulerServicer._sampling_params_from_proto(params)

    assert "seed" not in actual
    assert "custom_params" not in actual


def test_generate_request_falls_back_to_configured_server_seed() -> None:
    servicer = object.__new__(TokenSpeedSchedulerServicer)
    servicer.server_args = SimpleNamespace(seed=42)
    generated = SimpleNamespace(sampling_params={"temperature": 1.0})

    with (
        patch.object(
            SMGTokenSpeedSchedulerServicer,
            "_build_generate_req",
            return_value=generated,
        ),
        patch("tokenspeed.cli.smg_engine.sys.argv", ["smg-engine", "--seed", "42"]),
    ):
        actual = servicer._build_generate_req(SimpleNamespace())

    assert actual.sampling_params["seed"] == 42


def test_generate_request_keeps_explicit_wire_seed() -> None:
    servicer = object.__new__(TokenSpeedSchedulerServicer)
    servicer.server_args = SimpleNamespace(seed=42)
    generated = SimpleNamespace(sampling_params={"seed": 7})

    with patch.object(
        SMGTokenSpeedSchedulerServicer,
        "_build_generate_req",
        return_value=generated,
    ):
        actual = servicer._build_generate_req(SimpleNamespace())

    assert actual.sampling_params["seed"] == 7


def test_generate_request_does_not_use_implicit_random_server_seed() -> None:
    servicer = object.__new__(TokenSpeedSchedulerServicer)
    # ServerArgs resolves an omitted --seed to a process-random default.
    servicer.server_args = SimpleNamespace(seed=123456)
    generated = SimpleNamespace(sampling_params={})

    with (
        patch.object(
            SMGTokenSpeedSchedulerServicer,
            "_build_generate_req",
            return_value=generated,
        ),
        patch("tokenspeed.cli.smg_engine.sys.argv", ["smg-engine"]),
    ):
        actual = servicer._build_generate_req(SimpleNamespace())

    assert "seed" not in actual.sampling_params
