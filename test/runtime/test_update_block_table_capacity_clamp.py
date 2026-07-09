"""Regression tests for ``update_block_table`` page-row mirroring.

The scheduler owns the authoritative logical page row for each request.
``req_to_page`` is the Python/device mirror used by cache and attention kernels.

Two failure modes are covered here:

* If the scheduler rewrites earlier logical pages while also appending a tail
  page, applying only ``new_occupied_pages`` as an append delta leaves stale
  mirror entries and can alias one physical page into multiple logical ranges.
* If a request exceeds the per-request page table width, the mirror update must
  clamp the copied row instead of raising and tearing down the whole engine.

Tests use a lightweight ``SimpleNamespace`` stand-in for ``forward_op`` so we
don't depend on the C++ scheduler binding. The ``update_req_to_page`` kernel
is itself stubbed (we assert what arguments it receives), keeping the test
CPU-only and GPU-free.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch


def _make_forward_op(
    begins: list[int],
    sizes: list[int],
    new_occupied_pages: list[list[int]] | None = None,
    occupied_pages: list[list[int]] | None = None,
    request_ids: list[str] | None = None,
    request_pool_indices: list[int] | None = None,
) -> SimpleNamespace:
    """Build a minimal forward_op stand-in with just the fields the function reads."""
    if new_occupied_pages is None:
        new_occupied_pages = [list(range(s)) for s in sizes]
    if occupied_pages is None:
        occupied_pages = [
            [-(i + 1) * 1000 - j for j in range(begin)] + list(new_pages)
            for i, (begin, new_pages) in enumerate(zip(begins, new_occupied_pages))
        ]
    if request_ids is None:
        request_ids = [f"req-{i}" for i in range(len(begins))]
    if request_pool_indices is None:
        request_pool_indices = list(range(len(begins)))
    return SimpleNamespace(
        begins=list(begins),
        sizes=list(sizes),
        new_occupied_pages=new_occupied_pages,
        occupied_pages=occupied_pages,
        request_ids=request_ids,
        request_pool_indices=request_pool_indices,
    )


def test_update_block_table_does_not_raise_on_overflow(monkeypatch):
    """Per-request overflow used to ``raise RuntimeError`` and kill the engine.

    Now it must clamp the offending request's ``size`` and proceed without
    raising, so the rest of the batch survives.
    """
    from tokenspeed.runtime.execution import cache_loc_kernel

    # max_pages=513 (the value from the real crash). req[1] is the offender:
    # its authoritative occupied row has 514 pages.
    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(
        begins=[400, 513, 100],
        sizes=[2, 1, 3],
        occupied_pages=[
            list(range(402)),
            list(range(514)),
            list(range(103)),
        ],
    )

    captured: dict = {}

    def fake_update_req_to_page(
        req_to_page,
        req_pool_indices,
        new_occupied_pages,
        new_occupied_pages_num,
        pages_copy_starts,
    ):
        captured["num"] = new_occupied_pages_num.tolist()
        captured["starts"] = pages_copy_starts.tolist()
        captured["pages"] = new_occupied_pages.tolist()

    monkeypatch.setattr(cache_loc_kernel, "update_req_to_page", fake_update_req_to_page)

    # Must not raise.
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    # All rows refresh from position 0; the offender is clamped to table width.
    assert captured["num"] == [402, 513, 103]
    assert captured["starts"] == [0, 0, 0]
    assert len(captured["pages"]) == 402 + 513 + 103


def test_update_block_table_refreshes_authoritative_scheduler_row(monkeypatch):
    """A scheduler row update can include earlier logical page replacements,
    so the device mirror must copy the full occupied_pages row from page 0."""
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(
        begins=[2],
        sizes=[1],
        new_occupied_pages=[[12]],
        occupied_pages=[[90, 91, 12]],
    )

    captured: dict = {}

    def fake_update_req_to_page(
        req_to_page,
        req_pool_indices,
        new_occupied_pages,
        new_occupied_pages_num,
        pages_copy_starts,
    ):
        captured["num"] = new_occupied_pages_num.tolist()
        captured["starts"] = pages_copy_starts.tolist()
        captured["pages"] = new_occupied_pages.tolist()

    monkeypatch.setattr(cache_loc_kernel, "update_req_to_page", fake_update_req_to_page)
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    assert captured["starts"] == [0]
    assert captured["num"] == [3]
    assert captured["pages"] == [90, 91, 12]


def test_update_block_table_does_not_apply_append_delta_when_prefix_changed(
    monkeypatch,
):
    """Copying only begin=2,new_tail would leave the stale prefix in req_to_page.

    The bad old mirror could become [old_prefix, page1, old_prefix] if the old
    prefix page was reused as the tail. The fixed path copies the scheduler's
    authoritative row [cached_prefix, page1, reused_tail] from logical page 0.
    """
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(
        begins=[2],
        sizes=[1],
        new_occupied_pages=[[10]],
        occupied_pages=[[99, 11, 10]],
    )

    captured: dict = {}

    def fake_update_req_to_page(
        req_to_page,
        req_pool_indices,
        new_occupied_pages,
        new_occupied_pages_num,
        pages_copy_starts,
    ):
        captured["num"] = new_occupied_pages_num.tolist()
        captured["starts"] = pages_copy_starts.tolist()
        captured["pages"] = new_occupied_pages.tolist()

    monkeypatch.setattr(cache_loc_kernel, "update_req_to_page", fake_update_req_to_page)
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    assert captured == {
        "num": [3],
        "starts": [0],
        "pages": [99, 11, 10],
    }


def test_update_block_table_clamp_partial_overflow(monkeypatch):
    """If the authoritative scheduler row is too wide, clamp to table width."""
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    # The authoritative row has 516 pages, so it clamps to the table width.
    forward_op = _make_forward_op(
        begins=[512],
        sizes=[4],
        new_occupied_pages=[[700, 701, 702, 703]],
        occupied_pages=[list(range(516))],
    )

    captured: dict = {}

    def fake_update_req_to_page(
        req_to_page,
        req_pool_indices,
        new_occupied_pages,
        new_occupied_pages_num,
        pages_copy_starts,
    ):
        captured["num"] = new_occupied_pages_num.tolist()
        captured["pages"] = new_occupied_pages.tolist()

    monkeypatch.setattr(cache_loc_kernel, "update_req_to_page", fake_update_req_to_page)
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )

    assert captured["num"] == [513]
    assert captured["pages"] == list(range(513))


def test_update_block_table_zero_total_returns_early(monkeypatch):
    """If every size is 0 the function must short-circuit (no kernel call)."""
    from tokenspeed.runtime.execution import cache_loc_kernel

    req_to_page = torch.zeros(8, 513, dtype=torch.int32)
    forward_op = _make_forward_op(begins=[100, 200], sizes=[0, 0])

    called = {"v": False}

    def fake_update_req_to_page(**kwargs):
        called["v"] = True

    monkeypatch.setattr(cache_loc_kernel, "update_req_to_page", fake_update_req_to_page)
    cache_loc_kernel.update_block_table(
        forward_op, device="cpu", req_to_page=req_to_page
    )
    assert called["v"] is False


def test_update_block_table_logs_warning_on_clamp():
    """Engine survives, but the clamp must be loud (logger.warning) so the
    upstream length-bound bug remains visible. cache_loc_kernel uses a
    non-propagating colorful logger, so caplog can't see it; attach a direct
    capturing handler instead."""
    import logging

    from tokenspeed.runtime.execution import cache_loc_kernel

    captured_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_records.append(record)

    handler = _Capture(level=logging.WARNING)
    cache_loc_kernel.logger.addHandler(handler)
    try:
        req_to_page = torch.zeros(8, 513, dtype=torch.int32)
        forward_op = _make_forward_op(
            begins=[513],
            sizes=[1],
            request_ids=["my-bad-req"],
        )
        with mock.patch.object(
            cache_loc_kernel, "update_req_to_page", lambda **kw: None
        ):
            cache_loc_kernel.update_block_table(
                forward_op, device="cpu", req_to_page=req_to_page
            )
    finally:
        cache_loc_kernel.logger.removeHandler(handler)

    msgs = [r.getMessage() for r in captured_records]
    assert any("my-bad-req" in m for m in msgs), msgs
    assert any("page copy would exceed req_to_page capacity" in m for m in msgs), msgs
