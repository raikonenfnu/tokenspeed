"""Tests for scheduler request-burst admission."""

from __future__ import annotations

from types import SimpleNamespace

import zmq

from tokenspeed.runtime.engine.io_struct import (
    FlushCacheReqInput,
    FlushCacheReqOutput,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler


def _generate_req(seed: int | None, token: int = 0) -> TokenizedGenerateReqInput:
    req = object.__new__(TokenizedGenerateReqInput)
    req.sampling_params = SimpleNamespace(seed=seed)
    req.input_ids = [token]
    return req


class _BurstSocket:
    def __init__(self, *groups: list[object]) -> None:
        self._ready = list(groups[0])
        self._later = [list(group) for group in groups[1:]]
        self.poll_calls = 0

    def recv_pyobj(self, flags: int) -> object:
        assert flags == zmq.NOBLOCK
        if not self._ready:
            raise zmq.Again()
        return self._ready.pop(0)

    def poll(self, timeout: int, flags: int) -> int:
        assert timeout > 0
        assert flags == zmq.POLLIN
        self.poll_calls += 1
        if not self._later:
            return 0
        self._ready.extend(self._later.pop(0))
        return zmq.POLLIN


def _handler(socket: _BurstSocket) -> RequestHandler:
    handler = object.__new__(RequestHandler)
    handler.attn_tp_rank = 0
    handler.attn_tp_size = 1
    handler.attn_tp_cpu_group = None
    handler.recv_func = socket
    return handler


def test_seeded_generate_burst_is_drained_without_reordering() -> None:
    first = [_generate_req(42, 15), _generate_req(42, 0)]
    second = [_generate_req(42, token) for token in range(14, 0, -1)]
    socket = _BurstSocket(first, second)

    actual = _handler(socket).recv_reqs()

    assert [req.input_ids for req in actual] == [
        [15],
        [0],
        *[[token] for token in range(14, 0, -1)],
    ]
    assert socket.poll_calls == 2


def test_unseeded_requests_keep_nonblocking_admission() -> None:
    first = [_generate_req(None)]
    socket = _BurstSocket(first, [_generate_req(None)])

    actual = _handler(socket).recv_reqs()

    assert actual == first
    assert socket.poll_calls == 0


def test_flush_cache_invokes_scheduler_callback_before_acknowledging() -> None:
    replies = []
    calls = []
    handler = object.__new__(RequestHandler)
    handler.flush_cache_fn = lambda: calls.append("flush") or True
    handler.send_func = SimpleNamespace(send_pyobj=replies.append)

    actual = handler.process_requests([FlushCacheReqInput()])

    assert actual == ([], [], [], [])
    assert calls == ["flush"]
    assert len(replies) == 1
    assert isinstance(replies[0], FlushCacheReqOutput)
    assert replies[0].success is True
