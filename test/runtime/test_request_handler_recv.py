"""Tests for scheduler-owned prefix-cache flushing."""

from __future__ import annotations

from types import SimpleNamespace

from tokenspeed.runtime.engine.io_struct import (
    FlushCacheReqInput,
    FlushCacheReqOutput,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler


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
