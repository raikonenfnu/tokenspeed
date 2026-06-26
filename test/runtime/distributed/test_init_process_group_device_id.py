# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Regression tests for issue #2: TP=8 Iris fd-passing deadlock.

Root cause: ``dist.init_process_group`` was called without ``device_id``, so
NCCL guesses each rank's GPU from its global rank; at the TP=8 rank->GPU layout
that guess is wrong and ``all_gather_object`` (Iris fd exchange) deadlocks.

These tests are deterministic and require no GPUs: they assert that an explicit
``device_id`` (the rank's own GPU) is threaded all the way into
``torch.distributed.init_process_group``. They FAIL on the pre-fix code (which
passed no ``device_id``) and PASS with the fix.
"""

from types import SimpleNamespace
from unittest import mock

import torch

from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager


def test_init_distributed_forwards_explicit_device_id():
    """``ProcessGroupManager.init_distributed`` must forward ``device_id`` to
    ``dist.init_process_group`` (not fall back to NCCL's global-rank guess)."""
    mapping = SimpleNamespace(world_size=8, rank=3)
    pg = ProcessGroupManager()
    dev = torch.device("cuda", 3)

    with mock.patch(
        "tokenspeed.runtime.distributed.process_group_manager.dist"
    ) as mdist:
        mdist.is_initialized.return_value = False
        pg.init_distributed(
            mapping,
            distributed_init_method="tcp://127.0.0.1:12345",
            backend="nccl",
            timeout=1800,
            device_id=dev,
        )

    mdist.init_process_group.assert_called_once()
    _, kwargs = mdist.init_process_group.call_args
    assert "device_id" in kwargs, (
        "init_process_group called without device_id -> NCCL will guess the "
        "GPU from global rank and deadlock all_gather_object at TP=8."
    )
    assert kwargs["device_id"] == dev


def test_distributed_initializer_binds_rank_local_device():
    """End-to-end through the real caller: ``DistributedInitializer.initialize``
    must pass ``device_id = torch.device(device, gpu_id)`` for this rank."""
    from tokenspeed.runtime.execution import distributed_initializer as di

    gpu_id = 5
    config = di.DistributedConfig(
        device="cuda",
        gpu_id=gpu_id,
        world_size=1,  # >1 would trigger a cross-rank memory check we don't mock
        global_rank=gpu_id,
        local_rank=gpu_id,
        attn_tp_rank=0,
        attn_tp_size=1,
        dp_size=1,
        dense_tp_size=1,
        moe_ep_size=1,
        moe_ep_rank=0,
        nccl_port=12345,
        mapping=mock.MagicMock(),
    )

    with mock.patch.object(di, "pg_manager") as mpg, mock.patch.object(
        di.torch, "get_device_module"
    ) as mdev, mock.patch.object(
        di, "maybe_set_numa_aware_cpu_affinity"
    ), mock.patch.object(
        di, "get_available_gpu_memory", return_value=10.0
    ):
        mdev.return_value = mock.MagicMock()  # set_device(...) becomes a no-op
        di.DistributedInitializer.initialize(config)

    mpg.init_distributed.assert_called_once()
    _, kwargs = mpg.init_distributed.call_args
    assert kwargs.get("device_id") == torch.device("cuda", gpu_id), (
        "DistributedInitializer must bind the default process group to this "
        "rank's own GPU via device_id; otherwise TP=8 deadlocks in Iris "
        "fd-passing."
    )
